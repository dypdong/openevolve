#!/usr/bin/env python3
"""opencode_client.py —— 面向批处理的 opencode 同步客户端（V1 / V2 双协议）。
 
为什么要分版本
    opencode 2.x（V2）改了服务端路由与字段：
        V1: POST /session                    -> POST /session/{id}/message
        V2: POST /api/session                -> POST /api/session/{id}/prompt
    V2 的模型在**创建 session 时**确定（{"providerID": ..., "id": ...}），
    prompt 只接受 {"text": "..."}；且 /api/* 强制 Basic Auth。
    V1 则相反：prompt body 里带 parts/model/agent 等字段。
    两者字段名还不一样（modelID vs id），不能靠改路径了事，所以这里做成适配器。
 
版本选择
    OpencodeClient(api="auto")  探测 /api/session，命中即 V2，否则回退 V1
    api="v1" / "v2"             跳过探测，强制使用
    探测失败（服务未起）时 wait_ready() 会继续轮询，不会提前抛错。
 
依赖
    pip install httpx
 
库内使用
    from opencode_client import OpencodeServer, OpencodeClient, Model
 
    with OpencodeServer(cwd="/path/to/project") as server:
        with OpencodeClient(server.base_url, api="auto",
                            default_model=Model("deepseek", "deepseek-flash")) as c:
            print(c.api)                       # 实际协商到的版本：v1 / v2
            print(c.ask("统计 feature/ 下的空值率").text)
 
    # 批量：每条独立 session，单条失败不影响整批
    for item in c.batch(["任务A", "任务B"], workers=4):
        print(item.index, item.ok, item.error)
 
连接已有服务
    with OpencodeClient("http://127.0.0.1:4096", password="xxx") as c:
        ...
config.yaml 用法
llm:
  provider: "opencode"
  api_base: "http://127.0.0.1:4096"
  api_key: "${OPENCODE_SERVER_PASSWORD}"
  models:
    - name: "deepseek/deepseek-flash"
      weight: 0.5
    - name: "deepseek/deepseek-v4-pro"
      weight: 0.5      
 
CLI
    python opencode_client.py ask "统计空值率" --serve --cwd . \
        --model deepseek/deepseek-flash --api auto
    python opencode_client.py models --base-url http://127.0.0.1:4096
"""
 
from __future__ import annotations
 
import argparse
import asyncio
import contextlib
import json
import logging
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
 
import httpx

from openevolve.llm.base import LLMInterface
 
__all__ = [
    "Model",
    "ToolCall",
    "TurnResult",
    "BatchItem",
    "OpencodeError",
    "TransportError",
    "ApiError",
    "AgentError",
    "ServerStartError",
    "OpencodeClient",
    "OpencodeServer",
    "OpencodeLLM",
    "init_opencode_client",
    "API_VERSIONS",
]
 
LOG = logging.getLogger("opencode_client")
 
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4096
DEFAULT_TIMEOUT = 1200.0          # agent 带工具调用时可能跑很久
DEFAULT_RETRIES = 2
DEFAULT_BACKOFF = 1.0
RETRY_STATUS = frozenset({429, 502, 503, 504})
API_VERSIONS = ("auto", "v1", "v2")
 
# 批处理下需要预先放行的工具；具体名字以 `GET /agent` 或 /doc 为准。
DEFAULT_ALLOWED_TOOLS: tuple[str, ...] = (
    "read", "write", "edit", "bash", "glob", "grep", "list", "patch",
)

# OpencodeLLM 后端默认值（配置暂内置于此，验证通过后再挪进 config.py）。
OPENCODE_DEFAULT_BASE_URL = "http://127.0.0.1:4096"
OPENCODE_DEFAULT_API = "v2"
OPENCODE_DEFAULT_MODEL = "deepseek/deepseek-flash"
OPENCODE_DEFAULT_TIMEOUT = 1200.0
OPENCODE_DEFAULT_RETRIES = 3
OPENCODE_DEFAULT_RETRY_DELAY = 5.0
OPENCODE_DEFAULT_USERNAME = "opencode"
OPENCODE_DEFAULT_AGENT: str | None = None
 
 
# --------------------------------------------------------------------------- #
# 异常
# --------------------------------------------------------------------------- #
class OpencodeError(RuntimeError):
    """本模块所有异常的基类。"""
 
 
class TransportError(OpencodeError):
    """连接/超时等传输层故障。"""
 
 
class ApiError(OpencodeError):
    """服务端返回 4xx/5xx。"""
 
    def __init__(self, status: int, method: str, url: str, body: Any) -> None:
        self.status = status
        self.method = method
        self.url = url
        self.body = body
        snippet = json.dumps(body, ensure_ascii=False) if isinstance(body, (dict, list)) else str(body)
        super().__init__(f"{method} {url} -> HTTP {status}: {snippet[:500]}")
 
    @property
    def not_found(self) -> bool:
        return self.status == 404
 
 
class AgentError(OpencodeError):
    """HTTP 200，但 assistant message 里带 error（模型/provider/tool 失败）。"""
 
    def __init__(self, error: Any) -> None:
        self.error = error
        super().__init__(f"agent error: {error}")
 
 
class ServerStartError(OpencodeError):
    """opencode serve 起不来或健康检查超时。"""
 
 
def _unauthorized_hint(
    username: str,
    password: str | None,
    resp: httpx.Response | None = None,
    api: str = "auto",
) -> str:
    """401 有三类来源：V2 的 /api 鉴权、V1 的 Basic Auth、本机请求被代理截走。"""
    lines = [
        "服务端返回 401（Unauthorized）。按这个顺序排查：",
        "  1. V2（2.x）的 /api/* 强制 Basic Auth；没有密码时也会拒绝，",
        "     确认 password / OPENCODE_SERVER_PASSWORD 与启动服务端时用的是同一个",
        "  2. 若是 V1，检查 password 与用户名 OPENCODE_SERVER_USERNAME"
        f"（当前 username={username!r}，password={'已设置' if password else '未设置'}）",
        "  3. 是不是代理：本机请求不该经过 HTTP_PROXY/ALL_PROXY，",
        "     用 `curl --noproxy '*' -i <url>` 对照一次；curl 通、客户端不通就是代理",
        "  4. 版本是否配错：显式传 api='v1' 或 api='v2' 排除探测偏差",
    ]
    if resp is not None:
        detail = [
            f"  {name}: {resp.headers[name]}"
            for name in ("www-authenticate", "server", "via", "x-proxy-id")
            if name in resp.headers
        ]
        if detail:
            lines.append("响应头：")
            lines.extend(detail)
        if "via" in resp.headers or "proxy" in (resp.headers.get("server") or "").lower():
            lines.append("  ↑ 这个 401 来自代理，不是 opencode")
    return "\n".join(lines)
 
 
# --------------------------------------------------------------------------- #
# 值对象
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Model:
    provider_id: str
    model_id: str
 
    @classmethod
    def parse(cls, spec: str) -> "Model":
        """接受 'provider/model' 或 'provider:model'。"""
        for sep in ("/", ":"):
            if sep in spec:
                provider, model = spec.split(sep, 1)
                if provider and model:
                    return cls(provider, model)
        raise ValueError(f"无法解析模型标识 {spec!r}，应为 'provider/model'")
 
    def to_json(self) -> dict[str, str]:
        """V1 形状。"""
        return {"providerID": self.provider_id, "modelID": self.model_id}
 
    def to_v2(self) -> dict[str, str]:
        """V2 形状：注意键名是 id，不是 modelID。"""
        return {"providerID": self.provider_id, "id": self.model_id}
 
    def __str__(self) -> str:
        return f"{self.provider_id}/{self.model_id}"
 
 
@dataclass
class ToolCall:
    name: str
    call_id: str | None = None
    status: str | None = None
    input: Any = None
    output: Any = None
    error: Any = None
 
 
@dataclass
class TurnResult:
    session_id: str
    message_id: str | None
    text: str
    reasoning: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    cost: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
 
    @property
    def empty(self) -> bool:
        return not self.text.strip()
 
 
@dataclass
class BatchItem:
    index: int
    prompt: str
    result: TurnResult | None = None
    error: BaseException | None = None
    elapsed: float = 0.0
 
    @property
    def ok(self) -> bool:
        return self.error is None
 
 
# --------------------------------------------------------------------------- #
# 协议适配器
# --------------------------------------------------------------------------- #
class _Adapter:
    """把两个版本的路径与字段差异收敛到一处。"""
 
    name = ""
 
    # 会话
    session_prefix = ""
 
    def create_path(self) -> str:
        raise NotImplementedError
 
    def create_body(
        self,
        *,
        title: str | None,
        model: Model | None,
        agent: str | None,
        allow_tools: Sequence[str],
        parent_id: str | None,
    ) -> dict[str, Any]:
        raise NotImplementedError
 
    def prompt_path(self, session_id: str) -> str:
        raise NotImplementedError
 
    def prompt_body(
        self,
        *,
        text: str,
        model: Model | None,
        agent: str | None,
        system: str | None,
        extra_parts: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        raise NotImplementedError
 
    def delete_path(self, session_id: str) -> str:
        return f"{self.session_prefix}/{session_id}"
 
    def abort_path(self, session_id: str) -> str:
        return f"{self.session_prefix}/{session_id}/abort"
 
    # 发现
    def health_path(self) -> str:
        raise NotImplementedError
 
    def models_path(self) -> str:
        return "/config/providers"
 
    def agents_path(self) -> str:
        return "/agent"
 
 
class _ApiV1(_Adapter):
    """opencode < 2.0：/session + /session/{id}/message。"""
 
    name = "v1"
    session_prefix = "/session"
 
    def create_path(self) -> str:
        return "/session"
 
    def create_body(self, *, title, model, agent, allow_tools, parent_id):
        body: dict[str, Any] = {}
        if title:
            body["title"] = title
        if parent_id:
            body["parentID"] = parent_id
        if model:
            body["model"] = model.to_json()
        if agent:
            body["agent"] = agent
        if allow_tools:
            body["permission"] = {
                "rules": [{"toolName": t, "action": "allow"} for t in allow_tools]
            }
        return body
 
    def prompt_path(self, session_id: str) -> str:
        return f"/session/{session_id}/message"
 
    def prompt_body(self, *, text, model, agent, system, extra_parts):
        body: dict[str, Any] = {"parts": [{"type": "text", "text": text}, *extra_parts]}
        if model:
            body["model"] = model.to_json()
        if agent:
            body["agent"] = agent
        if system:
            body["system"] = system
        return body
 
    def health_path(self) -> str:
        return "/global/health"
 
 
class _ApiV2(_Adapter):
    """opencode >= 2.0：/api/session + /api/session/{id}/prompt。
 
    V2 的模型在创建 session 时确定，prompt 只接受 {"text": ...}。
    permissions 为 [{action, resource, effect}]；空 allow_tools = 禁用全部工具，被拒时自动降级。
    """
 
    name = "v2"
    session_prefix = "/api/session"
 
    def create_path(self) -> str:
        return "/api/session"
 
    def create_body(self, *, title, model, agent, allow_tools, parent_id):
        body: dict[str, Any] = {}
        if title:
            body["title"] = title
        if parent_id:
            body["parentID"] = parent_id
        if model:
            body["model"] = model.to_v2()
        if agent:
            body["agent"] = agent
        # V2 权限：Permission.Ruleset = [{action, resource, effect}]。
        # 空 allow_tools = 禁用全部工具（纯文本生成）；否则 deny 全部再放行指定工具。
        rules = [{"action": "*", "resource": "*", "effect": "deny"}]
        rules += [{"action": t, "resource": "*", "effect": "allow"} for t in allow_tools]
        body["permissions"] = rules
        return body
 
    def prompt_path(self, session_id: str) -> str:
        return f"/api/session/{session_id}/prompt"
 
    def prompt_body(self, *, text, model, agent, system, extra_parts):
        # 模型与 agent 已在建 session 时确定；这里只传 text。
        if system:
            LOG.warning("V2 的 prompt 不接受 system 字段，已前置到文本")
            text = f"{system}\n\n{text}"
        if extra_parts:
            LOG.warning("V2 的 prompt 不接受额外 parts，已忽略 %d 个", len(extra_parts))
        return {"text": text}
 
    def health_path(self) -> str:
        return "/api/session"
 
    def models_path(self) -> str:
        return "/api/model"
 
    def agents_path(self) -> str:
        return "/api/agent"
 
 
_ADAPTERS: dict[str, type[_Adapter]] = {"v1": _ApiV1, "v2": _ApiV2}
 
 
# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #
class OpencodeClient:
    """同步客户端。线程安全：内部 httpx.Client 可被多线程共享，并发单位是 session。"""
 
    def __init__(
        self,
        base_url: str = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}",
        *,
        api: str = "auto",
        password: str | None = None,
        username: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        connect_timeout: float = 10.0,
        retries: int = DEFAULT_RETRIES,
        backoff: float = DEFAULT_BACKOFF,
        default_model: Model | None = None,
        default_agent: str | None = None,
        allow_tools: Sequence[str] | None = DEFAULT_ALLOWED_TOOLS,
        transport: httpx.BaseTransport | None = None,
        trust_env: bool = False,
    ) -> None:
        if api not in API_VERSIONS:
            raise ValueError(f"api 必须是 {API_VERSIONS} 之一，收到 {api!r}")
        self.base_url = base_url.rstrip("/")
        self.api = api
        self.default_model = default_model
        self.default_agent = default_agent
        self.allow_tools = tuple(allow_tools) if allow_tools else ()
        self._retries = max(0, retries)
        self._backoff = max(0.0, backoff)
        self.timeout = timeout
        self._adapter: _Adapter | None = _ADAPTERS[api]() if api != "auto" else None
 
        # 环境变量兜底：opencode serve 用 OPENCODE_SERVER_PASSWORD 开启鉴权，
        # 调用方常常只设了环境变量而没传参数，此时不应静默地发无凭证请求。
        username = username or os.environ.get("OPENCODE_SERVER_USERNAME", "opencode")
        password = password if password is not None else os.environ.get("OPENCODE_SERVER_PASSWORD")
        self.username = username
        self.password = password or None
        auth = (username, password) if password else None
 
        self._http = httpx.Client(
            base_url=self.base_url,
            auth=auth,
            transport=transport,
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            # 本机服务不该走系统代理。开着 trust_env 时，HTTP_PROXY/ALL_PROXY
            # 会把 127.0.0.1 的请求转发出去，表现就是莫名其妙的 401/403。
            trust_env=trust_env,
        )
 
    # -- 生命周期 ---------------------------------------------------------
    def close(self) -> None:
        self._http.close()
 
    def __enter__(self) -> "OpencodeClient":
        return self
 
    def __exit__(self, *exc: object) -> None:
        self.close()
 
    # -- 版本协商 ---------------------------------------------------------
    @property
    def adapter(self) -> _Adapter:
        if self._adapter is None:
            self.api = self.detect_api()
            self._adapter = _ADAPTERS[self.api]()
        return self._adapter
 
    def detect_api(self, timeout: float = 2.0) -> str:
        """V2 认 /api/session；V1 认 /global/health。
 
        V2 的 /api/* 要鉴权，所以 401 也是"存在"的证据。
        """
        try:
            resp = self._http.get("/api/session", timeout=timeout)
        except httpx.TransportError as exc:
            raise TransportError(f"连接 {self.base_url} 失败: {exc}") from exc
        if resp.status_code in (200, 204, 400, 401, 405):
            LOG.debug("探测到 V2（/api/session -> %s）", resp.status_code)
            return "v2"
 
        try:
            resp = self._http.get("/global/health", timeout=timeout)
        except httpx.TransportError as exc:
            raise TransportError(f"连接 {self.base_url} 失败: {exc}") from exc
        if resp.status_code in (200, 401):
            LOG.debug("探测到 V1（/global/health -> %s）", resp.status_code)
            return "v1"
 
        raise OpencodeError(
            "无法识别服务端版本：/api/session 与 /global/health 都不可用。"
            "请显式传 api='v1' 或 api='v2'，并确认服务已启动。"
        )
 
    # -- 底层请求 ---------------------------------------------------------
    def _request(self, method: str, path: str, *, idempotent: bool = True, **kwargs: Any) -> httpx.Response:
        """带重试的请求。
 
        idempotent=False 时不重试传输层错误（POST prompt 会重复执行 agent），
        但 429/5xx 仍然重试——这类响应意味着服务端没有处理请求。
        """
        delay = self._backoff
        last_exc: httpx.TransportError | None = None
 
        for attempt in range(1, self._retries + 2):
            try:
                resp = self._http.request(method, path, **kwargs)
            except httpx.TransportError as exc:
                last_exc = exc
                if not idempotent or attempt > self._retries:
                    raise TransportError(f"{method} {path} 失败: {exc}") from exc
                LOG.warning("传输错误 %s %s（第 %d 次）: %s", method, path, attempt, exc)
                time.sleep(delay)
                delay *= 2
                continue
 
            if resp.status_code in RETRY_STATUS and attempt <= self._retries:
                LOG.warning("%s %s -> HTTP %s，%.1fs 后重试", method, path, resp.status_code, delay)
                time.sleep(delay)
                delay *= 2
                continue
 
            if resp.status_code == 401:
                raise ApiError(401, method, str(resp.request.url), _safe_json(resp))
            if resp.status_code >= 400:
                raise ApiError(resp.status_code, method, str(resp.request.url), _safe_json(resp))
            return resp
 
        raise TransportError(f"{method} {path} 重试耗尽: {last_exc}")
 
    # -- 健康与发现 -------------------------------------------------------
    def _probe(self, timeout: float = 2.0) -> tuple[str, httpx.Response | None]:
        """返回 (ok / unauthorized / down, 响应)。探测阶段失败视为 down，由调用方重试。"""
        try:
            adapter = self.adapter
        except (OpencodeError, httpx.HTTPError):
            return "down", None
        try:
            resp = self._http.get(adapter.health_path(), timeout=timeout)
        except httpx.TransportError:
            return "down", None
        if resp.status_code in (200, 204):
            return "ok", resp
        if resp.status_code == 401:
            return "unauthorized", resp
        return "down", resp
 
    def health(self, timeout: float = 5.0) -> bool:
        return self._probe(timeout)[0] == "ok"
 
    def wait_ready(self, timeout: float = 30.0, interval: float = 0.2) -> "OpencodeClient":
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state, resp = self._probe()
            if state == "ok":
                return self
            if state == "unauthorized":
                raise ServerStartError(
                    _unauthorized_hint(self.username, self.password, resp, api=self.api)
                )
            time.sleep(interval)
        raise ServerStartError(
            f"{self.base_url} 在 {timeout:.0f}s 内未就绪"
            f"（探测版本={self.api}，检查服务是否启动、端口与版本配置）"
        )
 
    def list_models(self) -> list[Model]:
        """返回结构随版本略有差异，这里做宽松提取；接口不存在时返回空列表。"""
        try:
            payload = self._request("GET", self.adapter.models_path()).json()
        except ApiError as exc:
            if exc.status in (404, 405):
                LOG.debug("%s 未提供模型列表接口", self.api)
                return []
            raise
        if not isinstance(payload, Mapping):
            return []

        # V2: {"location": ..., "data": [Model.Info{providerID, modelID, ...}, ...]}
        data = payload.get("data")
        if isinstance(data, list):
            models: list[Model] = []
            for entry in data:
                if not isinstance(entry, Mapping):
                    continue
                pid = entry.get("providerID") or entry.get("provider")
                mid = entry.get("modelID") or entry.get("id")
                if pid and mid:
                    models.append(Model(pid, mid))
            return models

        # V1: {"providers": [...]} 或 {"all": [...]}，每个 provider 下挂 models。
        providers = payload.get("providers") or payload.get("all") or []
        models = []
        for provider in providers:
            pid = provider.get("id") or provider.get("providerID")
            raw = provider.get("models") or {}
            if isinstance(raw, Mapping):
                ids = list(raw.keys())
            else:
                ids = [m.get("id") for m in raw if isinstance(m, Mapping)]
            models.extend(Model(pid, mid) for mid in ids if pid and mid)
        return models
 
    def list_agents(self) -> list[dict[str, Any]]:
        try:
            payload = self._request("GET", self.adapter.agents_path()).json()
        except ApiError as exc:
            if exc.status in (404, 405):
                return []
            raise
        return payload if isinstance(payload, list) else payload.get("agents", [])
 
    # -- 会话 -------------------------------------------------------------
    def create_session(
        self,
        *,
        title: str | None = None,
        model: Model | None = None,
        agent: str | None = None,
        allow_tools: Sequence[str] | None = None,
        parent_id: str | None = None,
    ) -> str:
        adapter = self.adapter
        tools = tuple(allow_tools) if allow_tools is not None else self.allow_tools
        body = adapter.create_body(
            title=title,
            model=model or self.default_model,
            agent=agent or self.default_agent,
            allow_tools=tools,
            parent_id=parent_id,
        )
 
        # 可选字段被拒时逐级降级，避免一个未验证的字段让整批任务失败。
        optional = ["permissions", "agent"]
        while True:
            try:
                return self._create_session_call(adapter, body)
            except ApiError as exc:
                if exc.status not in (400, 422):
                    raise
                victim = next((key for key in optional if key in body), None)
                if victim is None:
                    raise
                LOG.warning("%s 的 %s 字段被拒绝（HTTP %s），去掉后重试",
                            adapter.name, victim, exc.status)
                body.pop(victim, None)
                optional.remove(victim)
 
    def _create_session_call(self, adapter: _Adapter, body: Mapping[str, Any]) -> str:
        payload = self._request("POST", adapter.create_path(), json=dict(body)).json()
        session_id = _extract_session_id(payload)
        if not session_id:
            raise OpencodeError(f"创建会话返回异常: {payload}")
        LOG.debug("session created: %s (%s)", session_id, adapter.name)
        return session_id
 
    def delete_session(self, session_id: str) -> None:
        self._request("DELETE", self.adapter.delete_path(session_id))
 
    @contextlib.contextmanager
    def session(self, **kwargs: Any) -> Iterator[str]:
        session_id = self.create_session(**kwargs)
        try:
            yield session_id
        finally:
            self._quiet_delete(session_id)
 
    def _quiet_delete(self, session_id: str) -> None:
        try:
            self.delete_session(session_id)
        except OpencodeError as exc:  # 清理失败不应覆盖主流程的异常
            LOG.debug("删除会话 %s 失败: %s", session_id, exc)
 
    def abort(self, session_id: str) -> None:
        self._request("POST", self.adapter.abort_path(session_id))
 
    # -- 主接口 -----------------------------------------------------------
    def ask(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        model: Model | None = None,
        agent: str | None = None,
        system: str | None = None,
        extra_parts: Sequence[Mapping[str, Any]] = (),
        ephemeral: bool = True,
        timeout: float | None = None,
    ) -> TurnResult:
        """发一条消息并等待完整回复（阻塞）。
 
        session_id 为空时新建临时会话，ephemeral=True 时结束后删除。
        复用已有会话会累积上下文，批量任务建议保持默认的隔离行为。
        """
        adapter = self.adapter
        own_session = session_id is None
        sid = session_id or self.create_session(model=model, agent=agent)
        try:
            body = adapter.prompt_body(
                text=prompt,
                model=model or self.default_model,
                agent=agent or self.default_agent,
                system=system,
                extra_parts=tuple(extra_parts),
            )
            resp = self._request(
                "POST", adapter.prompt_path(sid),
                idempotent=False,
                json=body,
                timeout=timeout or httpx.USE_CLIENT_DEFAULT,
            )
            if self.api == "v2":
                # V2 的 prompt 是异步投递：立即返回 user 消息回执，
                # assistant 回复需轮询 GET /api/session/{id}/message。
                deadline = time.monotonic() + (timeout or self.timeout or DEFAULT_TIMEOUT)
                return self._collect_v2_turn(sid, deadline)
            result = _parse_turn(sid, resp)
            if result.empty and resp.status_code in (202, 204):
                LOG.warning(
                    "prompt 返回 %s 且无文本，可能需要在 /api/session/%s 轮询结果；"
                    "请用 --json 或 result.raw 查看原始响应",
                    resp.status_code, sid,
                )
            return result
        finally:
            if own_session and ephemeral:
                self._quiet_delete(sid)

    @staticmethod
    def _assistant_text(msg: Mapping[str, Any]) -> str:
        """拼接 assistant 消息里所有 text 部分的文本（流式/工具调用消息可能为空）。"""
        texts: list[str] = []
        for part in msg.get("content") or []:
            if isinstance(part, Mapping) and part.get("type") == "text":
                texts.append(part.get("text") or "")
        return "".join(texts)

    def _collect_v2_turn(self, session_id: str, deadline: float) -> TurnResult:
        """V2：prompt 是异步投递，轮询消息直到拿到 assistant 回复，并等 agent 循环
        进入 idle 后再返回。否则 ask() 紧接着删除会话，会与 agent 循环收尾产生竞态，
        服务端对已删除会话的 GET 会报 404。"""
        poll_interval = 0.5
        idle_grace = 5.0  # 拿到 assistant 回复后，给 idle 消息的宽限窗口（秒）
        assistant_msg = None
        idle_grace_until = None
        while True:
            now = time.monotonic()
            if assistant_msg is not None and idle_grace_until is not None and now >= idle_grace_until:
                # 宽限期到仍未 idle，直接返回已拿到的回复（避免无限等待）
                return self._assistant_message_to_turn(session_id, assistant_msg)
            if now >= deadline:
                if assistant_msg is not None:
                    return self._assistant_message_to_turn(session_id, assistant_msg)
                raise TimeoutError(
                    f"opencode V2 未在期限内产出 assistant 回复 (session {session_id})"
                )
            resp = self._request(
                "GET",
                f"/api/session/{session_id}/message",
                params={"order": "desc", "limit": "20"},
                timeout=deadline - now,
            )
            payload = _decode(resp)
            if isinstance(payload, Mapping):
                messages = payload.get("data") or []
            else:
                messages = []
            idle_done = False
            for msg in messages:
                if not isinstance(msg, Mapping):
                    continue
                mtype = msg.get("type")
                if mtype == "assistant":
                    if msg.get("error"):
                        err = msg["error"]
                        detail = err.get("message") if isinstance(err, Mapping) else err
                        raise RuntimeError(f"opencode V2 assistant error: {detail}")
                    if assistant_msg is None and (msg.get("finish") or (msg.get("time") or {}).get("completed")) and self._assistant_text(msg):
                        assistant_msg = msg
                elif mtype == "idle":
                    outcome = msg.get("outcome")
                    if outcome == "succeeded":
                        idle_done = True
                    elif outcome in ("failed", "interrupted"):
                        err = msg.get("error") or {}
                        detail = err.get("message") if isinstance(err, Mapping) else err
                        raise RuntimeError(f"opencode V2 turn {outcome}: {detail}")
            if assistant_msg is not None:
                if idle_done:
                    return self._assistant_message_to_turn(session_id, assistant_msg)
                if idle_grace_until is None:
                    idle_grace_until = now + idle_grace
            time.sleep(poll_interval)

    def _assistant_message_to_turn(
        self, session_id: str, msg: Mapping[str, Any]
    ) -> TurnResult:
        """把 V2 assistant 消息（content 数组）转成 TurnResult。"""
        texts: list[str] = []
        reasoning: list[str] = []
        tools: list[ToolCall] = []
        for part in msg.get("content") or []:
            if not isinstance(part, Mapping):
                continue
            kind = part.get("type")
            if kind == "text":
                texts.append(part.get("text") or "")
            elif kind == "reasoning":
                reasoning.append(part.get("text") or "")
            elif kind == "tool":
                state = part.get("state") or {}
                tools.append(ToolCall(
                    name=part.get("tool") or "",
                    call_id=part.get("callID") or part.get("callId"),
                    status=state.get("status"),
                    input=state.get("input"),
                    output=state.get("output"),
                    error=state.get("error"),
                ))
        usage = msg.get("tokens") or {}
        return TurnResult(
            session_id=session_id,
            message_id=msg.get("id"),
            text="".join(texts),
            reasoning="".join(reasoning),
            tool_calls=tools,
            usage=dict(usage) if isinstance(usage, Mapping) else {},
            cost=msg.get("cost") if isinstance(msg.get("cost"), (int, float)) else None,
            raw=dict(msg),
        )
 
    def batch(
        self,
        prompts: Sequence[str],
        *,
        workers: int = 4,
        model: Model | None = None,
        agent: str | None = None,
        system: str | None = None,
        timeout: float | None = None,
        on_done: Any = None,
    ) -> list[BatchItem]:
        """并发执行多条独立任务，返回与输入同序的结果列表。
 
        失败被收敛到 BatchItem.error，不会中断其它任务。
        on_done: 可选回调 (BatchItem) -> None，用于进度上报。
        """
        items = [BatchItem(index=i, prompt=p) for i, p in enumerate(prompts)]
 
        def work(item: BatchItem) -> BatchItem:
            started = time.monotonic()
            try:
                item.result = self.ask(
                    item.prompt, model=model, agent=agent,
                    system=system, timeout=timeout,
                )
            except Exception as exc:  # noqa: BLE001 —— 批量场景需要兜住一切
                item.error = exc
                LOG.warning("任务 #%d 失败: %s", item.index, exc)
            item.elapsed = time.monotonic() - started
            if on_done is not None:
                on_done(item)
            return item
 
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(work, items))
        return items
 
 
# --------------------------------------------------------------------------- #
# 响应解析
# --------------------------------------------------------------------------- #
def _extract_session_id(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    for key in ("id", "sessionID", "sessionId"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    nested = payload.get("session") or payload.get("data")
    if isinstance(nested, Mapping):
        return _extract_session_id(nested)
    return None
 
 
def _decode(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return resp.text
 
 
def _parse_turn(session_id: str, resp: httpx.Response) -> TurnResult:
    """兼容 V1（{info, parts}）与 V2（可能是 {text} 或 {message:{...}}）的返回。"""
    payload = _decode(resp)
    if isinstance(payload, str):
        return TurnResult(session_id=session_id, message_id=None,
                          text=payload, reasoning="", raw={"text": payload})
    if not isinstance(payload, Mapping):
        return TurnResult(session_id=session_id, message_id=None,
                          text=str(payload), reasoning="", raw={"payload": payload})
 
    # 部分实现会包一层 {data: {...}}
    if "data" in payload and isinstance(payload["data"], Mapping) \
            and not {"info", "parts", "text", "content"} & set(payload):
        payload = payload["data"]
 
    info: Mapping[str, Any] = {}
    for key in ("info", "message", "result"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            info = value
            break
    if info.get("error"):
        raise AgentError(info["error"])
 
    parts = payload.get("parts")
    if parts is None and isinstance(payload.get("content"), list):
        parts = payload["content"]
 
    texts: list[str] = []
    reasoning: list[str] = []
    tools: list[ToolCall] = []
 
    for part in parts or []:
        if isinstance(part, str):
            texts.append(part)
            continue
        if not isinstance(part, Mapping):
            continue
        kind = part.get("type")
        if kind == "text" or (kind is None and "text" in part):
            texts.append(part.get("text") or "")
        elif kind == "reasoning":
            reasoning.append(part.get("text") or "")
        elif kind == "tool":
            state = part.get("state") or {}
            tools.append(ToolCall(
                name=part.get("tool") or "",
                call_id=part.get("callID") or part.get("callId"),
                status=state.get("status"),
                input=state.get("input"),
                output=state.get("output"),
                error=state.get("error"),
            ))
        elif "text" in part and isinstance(part["text"], str):
            texts.append(part["text"])
 
    # 顶层或 info 里直接带文本（V2 简化形态）
    if not texts:
        for source in (payload, info):
            for key in ("text", "output", "content"):
                value = source.get(key)
                if isinstance(value, str) and value:
                    texts.append(value)
                    break
            if texts:
                break
 
    usage = info.get("tokens") or payload.get("usage") or info.get("usage") or {}
    cost = info.get("cost", payload.get("cost"))
    # V2 的 prompt 响应可能直接是 {id, text}，此时顶层 id 即消息 id
    message_id = info.get("id") or payload.get("messageID") or payload.get("id")
 
    return TurnResult(
        session_id=session_id,
        message_id=message_id if isinstance(message_id, str) else None,
        text="".join(texts),
        reasoning="".join(reasoning),
        tool_calls=tools,
        usage=dict(usage) if isinstance(usage, Mapping) else {},
        cost=cost if isinstance(cost, (int, float)) else None,
        raw=dict(payload),
    )
 
 
def _safe_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return resp.text[:2000]
 
 
# --------------------------------------------------------------------------- #
# 服务端托管
# --------------------------------------------------------------------------- #
class OpencodeServer:
    """以子进程方式托管 `opencode serve`。
 
    cwd 决定项目根，也就决定了 agent 能读写的文件范围。
    """
 
    def __init__(
        self,
        *,
        cwd: str | os.PathLike[str] | None = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        password: str | None = None,
        username: str | None = None,
        executable: str = "opencode",
        log_path: str | os.PathLike[str] | None = None,
        startup_timeout: float = 30.0,
        extra_args: Sequence[str] = (),
    ) -> None:
        self.cwd = os.fspath(cwd) if cwd is not None else None
        self.host = host
        self.port = port
        # 与子进程环境保持一致：显式参数优先，其次读环境变量。
        # 否则会出现"服务端继承了密码、客户端却没带"的 401 死循环。
        self.password = password if password is not None else os.environ.get("OPENCODE_SERVER_PASSWORD")
        self.username = username or os.environ.get("OPENCODE_SERVER_USERNAME", "opencode")
        self.executable = executable
        self.log_path = os.fspath(log_path) if log_path is not None else None
        self.startup_timeout = startup_timeout
        self.extra_args = tuple(extra_args)
        self.process: subprocess.Popen[bytes] | None = None
        self.base_url = f"http://{host}:{port}"
        self._log_handle: Any = None
 
    # -- 上下文管理 -------------------------------------------------------
    def start(self) -> "OpencodeServer":
        if self.process is not None:
            return self
        if self.port == 0:
            self.port = _free_port(self.host)
            self.base_url = f"http://{self.host}:{self.port}"
 
        env = os.environ.copy()
        if self.password:
            env["OPENCODE_SERVER_PASSWORD"] = self.password
 
        cmd = [
            self.executable, "serve",
            "--hostname", self.host,
            "--port", str(self.port),
            *self.extra_args,
        ]
 
        if self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)) or ".", exist_ok=True)
            self._log_handle = open(self.log_path, "wb")
            stdout: Any = self._log_handle
        else:
            stdout = subprocess.DEVNULL
 
        LOG.info("启动 %s", " ".join(cmd))
        self.process = subprocess.Popen(
            cmd, cwd=self.cwd, env=env, stdout=stdout, stderr=subprocess.STDOUT,
        )
        try:
            self._wait_healthy()
        except Exception:
            self.stop()
            raise
        return self
 
    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        auth = (self.username, self.password) if self.password else None
        # 显式关掉 trust_env：探活是本机请求，不能交给 HTTP_PROXY 处理
        with httpx.Client(auth=auth, timeout=2.0, trust_env=False) as probe:
            while time.monotonic() < deadline:
                assert self.process is not None
                if self.process.poll() is not None:
                    raise ServerStartError(
                        f"opencode serve 提前退出（code={self.process.returncode}）\n{self._tail()}"
                    )
                try:
                    # V2 没有 /global/health，用 /api/session 探活；两者任一 200 即可。
                    for path in ("/global/health", "/api/session"):
                        resp = probe.get(f"{self.base_url}{path}")
                        if resp.status_code in (200, 204):
                            LOG.info("opencode 就绪: %s（%s）", self.base_url, path)
                            return
                        if resp.status_code == 401:
                            raise ServerStartError(
                                _unauthorized_hint(self.username, self.password, resp)
                            )
                except httpx.HTTPError:
                    pass
                time.sleep(0.2)
        raise ServerStartError(f"健康检查超时（{self.startup_timeout:.0f}s）\n{self._tail()}")
 
    def stop(self) -> None:
        proc, self.process = self.process, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                LOG.warning("opencode 未在 10s 内退出，强制结束")
                proc.kill()
                proc.wait(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
 
    def _tail(self, lines: int = 30) -> str:
        if not self.log_path:
            return "（未设置 log_path，无日志）"
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as fh:
                return "".join(fh.readlines()[-lines:])
        except OSError as exc:
            return f"（读取日志失败: {exc}）"
 
    def client(self, **kwargs: Any) -> OpencodeClient:
        kwargs.setdefault("password", self.password)
        kwargs.setdefault("username", self.username)
        return OpencodeClient(self.base_url, **kwargs)
 
    # -- 语法糖 -----------------------------------------------------------
    def __enter__(self) -> "OpencodeServer":
        return self.start()
 
    def __exit__(self, *exc: object) -> None:
        self.stop()
 
 
def _free_port(host: str = DEFAULT_HOST) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


# --------------------------------------------------------------------------- #
# OpenEvolve LLM 后端适配层
# --------------------------------------------------------------------------- #
# 连接一个"手工启动"的 `opencode serve` 服务（不负责启动子进程），
# 把 OpencodeClient.ask()（同步）桥接成 LLMInterface 要求的异步接口。
# 配置暂以内置默认值兜底，验证通过后再挪进 config.py。


def _cfg_value(model_cfg: Any, attr: str, default: Any) -> Any:
    """字段存在但为 None 时也回落到默认值（LLMModelConfig 多数字段默认即 None）。"""
    value = getattr(model_cfg, attr, None)
    return default if value is None else value


def _normalize_usage(usage: Any, model_spec: str) -> Dict[str, Any] | None:
    """把 opencode 的 info.tokens 映射成 OpenAILLM 风格的 token 统计（尽力而为）。"""
    if not isinstance(usage, Mapping):
        return None

    def first(*keys: str) -> int | float:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                return value
        return 0

    prompt = first("input", "prompt", "prompt_tokens", "input_tokens")
    completion = first("output", "completion", "completion_tokens", "output_tokens")
    total = first("total", "total_tokens")
    if not total:
        total = prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "model": model_spec,
    }


class OpencodeLLM(LLMInterface):
    """通过一个已启动的 `opencode serve`（V2 协议）生成文本的 LLM 后端。

    不负责启动/管理 opencode 子进程——调用方需先手工运行：
        opencode serve --hostname 127.0.0.1 --port 4096

    Usage in config.yaml:
        llm:
          provider: "opencode"
          api_base: "http://127.0.0.1:4096"
          api_key: "${OPENCODE_SERVER_PASSWORD}"
          models:
            - name: "deepseek/deepseek-flash"
              weight: 1.0

    Or inject programmatically:
        from openevolve.llm.opencode_cli2 import init_opencode_client
        for model_cfg in config.llm.models:
            model_cfg.init_client = init_opencode_client

    base_url 由 api_base 承接（等于 OpenAI 默认地址时回落 opencode 默认地址），
    密码由 api_key 承接（支持 ${VAR}，未设置时读 OPENCODE_SERVER_PASSWORD 环境变量）。
    temperature / top_p / max_tokens 在 V2 协议下无对应字段，被忽略。
    """

    def __init__(self, model_cfg=None):
        spec = _cfg_value(model_cfg, "name", OPENCODE_DEFAULT_MODEL)
        self.model = str(spec)  # 字符串，供 LLMEnsemble 日志与兼容
        self._model = Model.parse(self.model)

        self.system_message = getattr(model_cfg, "system_message", None)
        # base_url 用 api_base 承接；api_base 在 LLMConfig 有 OpenAI 默认值并会下发，
        # 因此把"等于 https://api.openai.com/v1"视作未设置，回落到 opencode 默认地址。
        api_base = getattr(model_cfg, "api_base", None)
        self.base_url = (
            api_base
            if api_base and api_base.rstrip("/") != "https://api.openai.com/v1"
            else OPENCODE_DEFAULT_BASE_URL
        )
        self.api = _cfg_value(model_cfg, "opencode_api", OPENCODE_DEFAULT_API)
        self.username = _cfg_value(model_cfg, "opencode_username", OPENCODE_DEFAULT_USERNAME)
        # 密码用 api_key 承接（api_key 已支持 ${VAR} 解析）；未设置时读环境变量兜底。
        self.password = _cfg_value(
            model_cfg, "api_key", os.environ.get("OPENCODE_SERVER_PASSWORD")
        )
        self.timeout = _cfg_value(model_cfg, "timeout", OPENCODE_DEFAULT_TIMEOUT)
        self.retries = _cfg_value(model_cfg, "retries", OPENCODE_DEFAULT_RETRIES)
        self.retry_delay = _cfg_value(model_cfg, "retry_delay", OPENCODE_DEFAULT_RETRY_DELAY)
        self.agent = _cfg_value(model_cfg, "opencode_agent", OPENCODE_DEFAULT_AGENT)
        self.allow_tools = _cfg_value(model_cfg, "opencode_allow_tools", ())
        self.last_usage: Dict[str, Any] | None = None

        # 不启动子进程；httpx 惰性连接，此处不会真的建连。
        # retries=0：HTTP/生成重试统一交给本类的外层循环，避免双重放大。
        self.client = OpencodeClient(
            base_url=self.base_url,
            api=self.api,
            password=self.password,
            username=self.username,
            timeout=self.timeout,
            retries=0,
            allow_tools=self.allow_tools,
            default_model=self._model,
            default_agent=self.agent,
        )
        LOG.info(
            "Initialized OpencodeLLM with model: %s (api=%s, base_url=%s)",
            self.model, self.api, self.base_url,
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        sys_msg = kwargs.pop("system_message", self.system_message) or ""
        return await self.generate_with_context(
            system_message=sys_msg,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        # 与 claude_code / codex_app_server 一致：只保留 user 消息，
        # assistant 历史在 opencode V2 下没有等价的结构化传参。
        user_content = "\n\n".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )

        timeout = kwargs.get("timeout", self.timeout)
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        model = kwargs.get("model") or self._model
        agent = kwargs.get("agent", self.agent)

        loop = asyncio.get_running_loop()
        for attempt in range(retries + 1):
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(
                        None, lambda: self._ask(user_content, system_message, model, agent)
                    ),
                    timeout=timeout + 30,
                )
            except asyncio.TimeoutError:
                if attempt < retries:
                    LOG.warning(
                        "OpencodeLLM timeout on attempt %d/%d. Retrying...",
                        attempt + 1, retries + 1,
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    LOG.error("All %d attempts failed with timeout", retries + 1)
                    raise
            except Exception as exc:
                if attempt < retries:
                    LOG.warning(
                        "OpencodeLLM error on attempt %d/%d: %s. Retrying...",
                        attempt + 1, retries + 1, exc,
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    LOG.error("All %d attempts failed with error: %s", retries + 1, exc)
                    raise

    def _ask(
        self, user_content: str, system_message: str, model: Model, agent: str | None
    ) -> str:
        """同步执行一次 ask（在线程池中运行）。"""
        try:
            result = self.client.ask(
                user_content,
                model=model,
                agent=agent,
                system=system_message or None,
                ephemeral=True,  # 每次一个临时 session，用完即删，隔离演化上下文
            )
        except TransportError as exc:
            raise TransportError(
                f"{exc} (is `opencode serve` running at {self.base_url}?)"
            ) from exc

        if result.empty:
            raise RuntimeError("Empty response from opencode")

        self.last_usage = _normalize_usage(result.usage, self.model) if result.usage else None
        LOG.info("Response model: %s usage%s",self.model, self.last_usage)
        LOG.info("Response text: %s ",result.text)
        return result.text


def init_opencode_client(model_cfg):
    """Factory 函数，兼容 OpenEvolve 的 init_client 配置钩子。"""
    return OpencodeLLM(model_cfg)
 
 
# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opencode_client",
        description="opencode 同步客户端（V1/V2，面向批处理）",
    )
    sub = parser.add_subparsers(dest="command", required=True)
 
    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--base-url", default="http://127.0.0.1:4096", help="已有服务地址，如 http://127.0.0.1:4096")
        p.add_argument("--serve", action="store_true", help="自行拉起 opencode serve 子进程")
        p.add_argument("--cwd", default=None, help="项目根（决定 agent 可见文件范围）")
        p.add_argument("--api", choices=API_VERSIONS, default=os.environ.get("OPENCODE_API", "auto"),
                       help="协议版本，默认自动探测")
        p.add_argument("--password", default=None, help="OPENCODE_SERVER_PASSWORD（缺省时从环境变量读取）")
        p.add_argument("--agent", default="build")
        p.add_argument("--model", default="deepseek/deepseek-flash", help="provider/model")
        p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
        p.add_argument("--json", action="store_true", help="以 JSON 输出结构化结果")
 
    ask = sub.add_parser("ask", help="单次提问")
    ask.add_argument("prompt")
    ask.add_argument("--system", default=None)
    common(ask)
 
    models = sub.add_parser("models", help="列出可用模型")
    common(models)
 
    ping = sub.add_parser("ping", help="探活并打印协商到的协议版本")
    common(ping)
    return parser
 
 
def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("OPENCODE_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = _build_parser().parse_args(argv)
    model = Model.parse(args.model) if args.model else None
    server: OpencodeServer | None = None
 
    try:
        if args.base_url:
            client = OpencodeClient(
                args.base_url, api=args.api, password=args.password, timeout=args.timeout,
            )
        elif args.serve:
            server = OpencodeServer(
                cwd=args.cwd, password=args.password, port=0,
                log_path=os.environ.get("OPENCODE_LOG_PATH"),
            ).start()
            client = server.client(api=args.api, timeout=args.timeout)
        else:
            raise SystemExit("需要 --base-url 或 --serve 之一")
 
        with client:
            client.wait_ready()
            if args.command == "ping":
                print(json.dumps({"base_url": client.base_url, "api": client.api}, ensure_ascii=False))
                return 0
            if args.command == "models":
                for m in client.list_models():
                    print(m)
                return 0
 
            result = client.ask(
                args.prompt, model=model, agent=args.agent,
                system=args.system,
            )
            if args.json:
                print(json.dumps(
                    {
                        "session_id": result.session_id,
                        "message_id": result.message_id,
                        "api": client.api,
                        "text": result.text,
                        "reasoning": result.reasoning,
                        "tool_calls": [tc.__dict__ for tc in result.tool_calls],
                        "usage": result.usage,
                        "cost": result.cost,
                        "raw": result.raw,
                    },
                    ensure_ascii=False, indent=2, default=str,
                ))
            else:
                print(result.text)
                LOG.info(
                    "api=%s session=%s tokens=%s cost=%s tools=%d",
                    client.api, result.session_id, result.usage,
                    result.cost, len(result.tool_calls),
                )
            return 0
    except OpencodeError as exc:
        LOG.error("%s", exc)
        return 2
    finally:
        if server is not None:
            server.stop()
 
 
if __name__ == "__main__":
    raise SystemExit(main())