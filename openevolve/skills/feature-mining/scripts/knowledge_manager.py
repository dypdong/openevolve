#!/usr/bin/env python3
"""File/directory knowledge synchronizer for the feature-mining skill.

Markdown files are the source of truth. The generated JSON indexes are derived
artifacts and are replaced atomically after validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    yaml = None
    YAML_IMPORT_ERROR = str(exc)
else:
    YAML_IMPORT_ERROR = None

LOGGER = logging.getLogger("feature_mining.knowledge")
INDEX_NAMES = ("index_by_id.json", "index_by_tag.json", "index_by_trigger.json", "manifest.json")
REQUIRED = ("id", "title", "domain", "type", "status", "version", "tags", "confidence")
VALID_STATUS = {"draft", "published", "deprecated", "archived"}
VALID_MODES = {"check", "draft", "publish"}


def emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def content_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def parse_document(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError("missing YAML front matter")
    match = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
    if not match:
        raise ValueError("invalid YAML front matter")
    metadata = parse_front_matter(match.group(1))
    if not isinstance(metadata, dict):
        raise ValueError("front matter must be a mapping")
    return metadata, match.group(2)


def parse_front_matter(text: str) -> dict[str, Any]:
    """Parse YAML when available; use a small scalar/list fallback for local use."""
    if yaml is not None:
        return yaml.safe_load(text) or {}
    result: dict[str, Any] = {}
    current_list: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- ") and current_list:
            result[current_list].append(line[2:].strip().strip("'\""))
            continue
        if ":" not in line:
            raise ValueError(f"invalid front matter line without YAML support: {raw}")
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if not value:
            result[key] = []
            current_list = key
        else:
            current_list = None
            if value.isdigit():
                result[key] = int(value)
            elif value.lower() in {"true", "false"}:
                result[key] = value.lower() == "true"
            elif value.startswith("[") and value.endswith("]"):
                result[key] = [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
            else:
                result[key] = value.strip("'\"")
    return result


def find_root(source: Path) -> Path:
    start = source if source.is_dir() else source.parent
    for candidate in (start, *start.parents):
        if (candidate / "index_by_id.json").exists() or (candidate / "tag_synonyms.yaml").exists():
            return candidate
    return start


def discover(source: Path) -> tuple[Path, str, list[Path]]:
    source = source.expanduser().resolve()
    if not source.exists():
        raise ValueError(f"path does not exist: {source}")
    if source.is_file():
        if source.suffix.lower() not in {".md", ".markdown"}:
            raise ValueError("file input must be Markdown")
        return find_root(source), "file", [source]
    files = sorted(p for p in source.rglob("*.md") if p.is_file() and not any(part.startswith(".") for part in p.relative_to(source).parts))
    return source, "directory", files


def load_index(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def validate_metadata(metadata: dict[str, Any], path: Path) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED:
        if key not in metadata:
            errors.append(f"missing metadata: {key}")
    if metadata.get("status") not in VALID_STATUS:
        errors.append(f"invalid status: {metadata.get('status')}")
    if not isinstance(metadata.get("version"), int) or metadata.get("version", 0) < 1:
        errors.append("version must be a positive integer")
    if not isinstance(metadata.get("tags"), list) or not all(isinstance(tag, str) for tag in metadata.get("tags", [])):
        errors.append("tags must be a list of strings")
    if not isinstance(metadata.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", str(metadata.get("id", ""))):
        errors.append("id must contain only letters, digits, '.', '_' or '-'")
    body = path.read_text(encoding="utf-8")
    if re.search(r"\b(TODO|TBD|PLACEHOLDER)\b", body, re.I):
        errors.append("document contains an unfinished placeholder")
    return errors


def scan(source: Path) -> dict[str, Any]:
    root, scope, files = discover(source)
    documents: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    ids: dict[str, Path] = {}
    for path in files:
        try:
            metadata, _ = parse_document(path)
            item_errors = validate_metadata(metadata, path)
            doc_id = str(metadata.get("id", ""))
            if doc_id in ids:
                item_errors.append(f"duplicate id also used by {ids[doc_id]}")
            else:
                ids[doc_id] = path
            if item_errors:
                errors.append({"path": str(path), "errors": "; ".join(item_errors)})
            documents.append({
                "id": doc_id,
                "path": str(path.relative_to(root)),
                "absolute_path": str(path),
                "title": metadata.get("title"),
                "domain": metadata.get("domain"),
                "type": metadata.get("type"),
                "status": metadata.get("status"),
                "version": metadata.get("version"),
                "tags": metadata.get("tags", []),
                "trigger_terms": metadata.get("trigger_terms", []),
                "confidence": metadata.get("confidence"),
                "hash": content_hash(path),
                "errors": item_errors,
            })
        except (OSError, ValueError, Exception) as exc:
            errors.append({"path": str(path), "errors": str(exc)})
    return {"root": root, "scope": scope, "files": documents, "errors": errors}


def build_indexes(root: Path, docs: list[dict[str, Any]], previous: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    by_id: dict[str, Any] = {}
    by_tag: dict[str, list[str]] = {}
    by_trigger: dict[str, list[str]] = {}
    for doc in docs:
        if doc["errors"] or doc["status"] not in {"published", "deprecated"}:
            continue
        entry = {key: doc[key] for key in ("path", "title", "domain", "type", "status", "version", "tags", "trigger_terms", "confidence", "hash")}
        by_id[doc["id"]] = entry
        if doc["status"] == "published":
            for tag in doc["tags"]:
                by_tag.setdefault(tag, []).append(doc["id"])
            for term in doc["trigger_terms"]:
                by_trigger.setdefault(str(term).lower(), []).append(doc["id"])
    for index in (by_tag, by_trigger):
        for key in index:
            index[key] = sorted(set(index[key]))
    return {"index_by_id": by_id, "index_by_tag": by_tag, "index_by_trigger": by_trigger}


def sync(source: Path, mode: str) -> dict[str, Any]:
    if mode not in VALID_MODES:
        raise ValueError(f"invalid mode: {mode}")
    result = scan(source)
    root: Path = result["root"]
    docs = result["files"]
    errors = result["errors"]
    previous = load_index(root, "index_by_id.json")
    current_ids = {doc["id"] for doc in docs if doc["id"]}
    old_ids = set(previous)
    added = [doc for doc in docs if doc["id"] not in previous and not doc["errors"]]
    updated = [doc for doc in docs if doc["id"] in previous and previous[doc["id"]].get("hash") != doc["hash"] and not doc["errors"]]
    unchanged = [doc for doc in docs if doc["id"] in previous and previous[doc["id"]].get("hash") == doc["hash"] and not doc["errors"]]
    archived = sorted(old_ids - current_ids) if result["scope"] == "directory" else []
    response = {"source": str(source), "root": str(root), "scope": result["scope"], "mode": mode, "status": "failed" if errors else "checked", "added": len(added), "updated": len(updated), "unchanged": len(unchanged), "archived": len(archived), "errors": errors}
    if mode == "check" or errors:
        return response
    # A single-file sync is incremental: preserve unrelated published entries.
    all_docs = docs
    if result["scope"] == "file":
        seen = {doc["id"] for doc in docs}
        for doc_id, old in previous.items():
            if doc_id in seen:
                continue
            all_docs.append({
                "id": doc_id,
                "path": old.get("path", ""),
                "absolute_path": str(root / old.get("path", "")),
                "title": old.get("title"),
                "domain": old.get("domain"),
                "type": old.get("type"),
                "status": old.get("status", "published"),
                "version": old.get("version", 1),
                "tags": old.get("tags", []),
                "trigger_terms": old.get("trigger_terms", []),
                "confidence": old.get("confidence"),
                "hash": old.get("hash"),
                "errors": [],
            })
    indexes = build_indexes(root, all_docs, previous)
    if archived and result["scope"] == "directory":
        for doc_id in archived:
            old = previous[doc_id].copy()
            old["status"] = "archived"
            indexes["index_by_id"][doc_id] = old
    for name, value in indexes.items():
        atomic_json(root / f"{name}.json", value)
    manifest = {"source_directory": str(root), "generated_at": now(), "scope": result["scope"], "document_count": len(indexes["index_by_id"]), "index_version": int(previous.get("_index_version", 0)) + 1}
    atomic_json(root / "manifest.json", manifest)
    response["status"] = "completed"
    response["index_version"] = manifest["index_version"]
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("scan", "validate"):
        p = sub.add_parser(command); p.add_argument("path")
    sync_parser = sub.add_parser("sync"); sync_parser.add_argument("path"); sync_parser.add_argument("--mode", choices=sorted(VALID_MODES), default="check")
    rebuild = sub.add_parser("rebuild-index"); rebuild.add_argument("path")
    args = parser.parse_args(); configure_logging(args.log_level)
    try:
        source = Path(args.path)
        if args.command in {"scan", "validate"}:
            result = scan(source); result["status"] = "failed" if result["errors"] else "valid"; result.pop("files", None) if args.command == "validate" else None; emit(result)
        elif args.command == "sync":
            emit(sync(source, args.mode))
        else:
            emit(sync(source, "publish"))
    except (OSError, ValueError, Exception) as exc:
        LOGGER.error("%s", exc); emit({"status": "failed", "error": str(exc)}); raise SystemExit(2)


if __name__ == "__main__":
    main()
