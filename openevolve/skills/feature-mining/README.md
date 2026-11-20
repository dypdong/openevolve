# Feature Mining Skill

`feature-mining` 用于在 Openevolve 特征挖掘过程中，为大模型提供数据查询和知识检索能力。它围绕参考程序、CSV 数据和领域知识，帮助模型生成可执行的目标程序。

## 能力范围

- 将 CSV 文件导入同目录下的 DuckDB 数据库。
- 根据 CSV 文件名创建数据表，并自动增加 `id` 字段。
- 查询表结构、行数、分布、聚合统计和元数据。
- 安全删除指定数据表。
- 从知识文件或知识目录中扫描、校验和发布知识。
- 根据提示词提取标签，并按标签匹配相关知识。
- 按摘要、方法细节、示例和失败案例逐级展开知识。

## 目录结构

```text
feature-mining/
├── SKILL.md                         # Skill 主规则
├── README.md                        # 使用简介
├── scripts/
│   ├── csv_duckdb.py                # CSV 与 DuckDB 工具
│   ├── knowledge_manager.py         # 知识文件/目录管理工具
│   └── requirements.txt             # 工具依赖
├── references/
│   ├── data-policy.md               # 数据约束
│   ├── knowledge-policy.md          # 知识约束
│   └── tool-contract.md             # 工具接口契约
└── knowledge/                       # 内置知识根目录
    ├── feature-engineering/
    ├── constraints/
    ├── failures/
    ├── experiments/
    ├── drafts/
    ├── archive/
    ├── index_by_id.json
    ├── index_by_tag.json
    ├── index_by_trigger.json
    ├── manifest.json
    └── tag_synonyms.yaml
```

## 安装依赖

```bash
pip install -r openevolve/skills/feature-mining/scripts/requirements.txt
```

依赖包括 DuckDB 和 PyYAML。

## 数据工具

数据工具的职责只有 CSV 导入、表元数据查询、只读 SQL 查询和表删除。每个操作都会根据 CSV 路径，在 CSV 同目录使用 `<csv 文件名>.duckdb`，并在方法内部打开和关闭连接。

### 创建数据库

```bash
python3 scripts/csv_duckdb.py create-database /data/orders.csv
```

`orders.csv` 会对应 `/data/orders.duckdb`。

### 导入 CSV

```bash
python3 scripts/csv_duckdb.py import-csv /data/orders.csv
```

默认表名为 `orders`。也可以指定表名：

```bash
python3 scripts/csv_duckdb.py import-csv /data/orders.csv --table user_orders
```

重复导入同名表会替换原表。

### 查询表结构

```bash
python3 scripts/csv_duckdb.py describe /data/orders.csv orders
```

### 执行统计 SQL

```bash
python3 scripts/csv_duckdb.py query /data/orders.csv \
  "SELECT category, COUNT(*) AS n, AVG(amount) AS avg_amount FROM orders GROUP BY category"
```

查询接口只允许单条 `SELECT` 语句，并限制返回行数。建议优先返回聚合统计、分位数、直方图或小样本，不要把完整大表放入模型上下文。

### 删除数据表

```bash
python3 scripts/csv_duckdb.py drop-table /data/orders.csv orders --confirm
```

删除操作必须显式传入 `--confirm`。

## 知识管理

知识文件使用 Markdown，文件开头必须包含 YAML front matter。至少应包含：

```yaml
---
id: feature.categorical.target_encoding
title: Target Encoding
domain: feature_engineering
type: method
status: published
version: 1
tags:
  - data_type.categorical
  - cardinality.high
trigger_terms:
  - 高基数类别
  - target encoding
confidence: verified
---
```

知识文件是事实来源，JSON 索引是自动生成的检索产物。

### 扫描和校验

扫描单个文件：

```bash
python3 scripts/knowledge_manager.py scan knowledge/feature-engineering/a.md
```

扫描整个知识目录：

```bash
python3 scripts/knowledge_manager.py validate knowledge
```

### 草稿同步

```bash
python3 scripts/knowledge_manager.py sync knowledge --mode draft
```

草稿模式用于检查和生成非运行时索引，不会把知识作为正式检索结果。

### 正式发布

```bash
python3 scripts/knowledge_manager.py sync knowledge --mode publish
```

目录发布会递归扫描 Markdown 文件，识别新增、更新、未变化和归档知识，并原子更新：

```text
index_by_id.json
index_by_tag.json
index_by_trigger.json
manifest.json
```

### 单文件更新

```bash
python3 scripts/knowledge_manager.py sync \
  knowledge/feature-engineering/a.md \
  --mode publish
```

单文件同步是增量操作，不会归档或删除其他知识。目录同步才会识别目录中已不存在的旧知识，并将其标记为归档。

## 推荐运行流程

```text
1. 读取参考程序和任务提示词
2. 导入或确认 CSV 数据
3. describe 获取字段和表规模
4. query 获取缺失率、基数、分布和聚合统计
5. 从提示词提取知识标签
6. 按标签匹配知识摘要
7. 按需展开方法、约束和失败案例
8. 生成候选特征程序
9. 检查字段引用、时间穿越、标签泄漏和资源限制
10. 交给 Openevolve evaluator 评估
```

## 重要约束

- 数据工具只执行只读统计 SQL；建表和删除表由专用命令完成。
- SQL 不允许写操作、DDL、文件读写、扩展加载或多语句执行。
- 数据库默认与 CSV 同目录，数据库名取 CSV 文件名。
- 知识检索先返回标签、摘要和风险，再按需展开详细内容。
- 草稿知识不能直接作为正式知识使用。
- 发布索引由工具生成，不应手工修改。
- 知识更新保留内容哈希和版本信息，避免破坏历史追溯。
