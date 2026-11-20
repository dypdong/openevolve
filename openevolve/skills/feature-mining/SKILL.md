---
name: feature-mining
description: Use when an Openevolve task must discover, design, or validate features using large datasets, SQL statistics, reference programs, or domain knowledge that should be retrieved progressively.
metadata:
  short-description: Data and knowledge assisted feature mining for Openevolve
---

# Feature Mining

Use this skill to help the model produce an executable target program from a reference program, dataset evidence, and selectively retrieved knowledge. Treat the reference program as one evidence source; use data tools to measure facts and knowledge tools to resolve method choice, assumptions, and known failure modes.

## When to use

Activate when any of these are true:

- feature logic depends on distributions, missingness, cardinality, time windows, or label balance;
- the dataset is too large to inspect directly and requires SQL aggregation or sampling;
- a new encoding, aggregation, normalization, interaction, or dimensionality method is being considered;
- leakage, time ordering, evaluation contamination, or domain constraints matter;
- a candidate program must be compared with a baseline using feature statistics.

Do not use it for purely syntactic edits or a local refactor whose feature semantics are unchanged.

## Operating contract

1. Establish the task, dataset identifier and version, label fields, time semantics, evaluation metric, allowed inputs, and resource limits.
2. Inspect the reference program and existing context before invoking tools.
3. If the user supplied a CSV that is not yet loaded, call `data.import_csv`; the table name defaults to the filename and the tool adds an `id` column.
4. Inspect table metadata with `data.describe`, then use `data.query` for read-only statistics. Never treat an assumption as a measured fact.
5. Retrieve method guidance from the knowledge tools progressively, starting with summaries and opening details only when required.
6. Generate a candidate program together with feature provenance, assumptions, query identifiers, knowledge citations, and risks.
7. Run schema, leakage, security, and resource checks before evaluation.
8. Pass the candidate to the existing Openevolve evaluator and use its result as feedback; do not silently change the evaluation protocol.

Read [references/tool-contract.md](references/tool-contract.md) when implementing or calling tools. Read [references/data-policy.md](references/data-policy.md) for CSV import, SQL, schema, and security rules. Read [references/knowledge-policy.md](references/knowledge-policy.md) when adding or querying progressive knowledge.

For knowledge maintenance, use [scripts/knowledge_manager.py](scripts/knowledge_manager.py). Pass either a Markdown file or a knowledge directory. Use `--mode check` for validation only, `--mode draft` for a non-runtime draft index, and `--mode publish` for an atomic published index update. A single-file sync never archives unrelated documents; directory sync may archive documents absent from the directory.

The bundled knowledge root is [knowledge/](knowledge/). Put published Markdown knowledge under its topic directories, drafts under `knowledge/drafts/`, and historical material under `knowledge/archive/`.

## Local reference implementation

The executable reference backend is [scripts/csv_duckdb.py](scripts/csv_duckdb.py). Each operation accepts the CSV path, opens its neighboring DuckDB file using the CSV filename stem (for example, `orders.csv` maps to `orders.duckdb`), and closes the connection internally. Wrap its subcommands as the `data.*` tools rather than exposing shell access to the model:

```text
create-database <csv-path> → import-csv <csv-path> → describe <csv-path> <table> → query <csv-path> <sql> → drop-table <csv-path> <table> (only when deletion is requested)
```

The script also supports explicit table deletion with `drop-table --confirm`. Logs go to stderr while results remain JSON on stdout. It is a local reference implementation; production deployments should add authentication, quotas, path allowlists, and sensitive-column policy.

## Tool selection

| Need | Tool | Expected result |
|---|---|---|
| Import a CSV | `data.import_csv` | DuckDB table name, row count, columns |
| Inspect table metadata | `data.describe` | Column names, types, nullability, row count |
| Calculate statistics or inspect metadata | `data.query` | Bounded read-only SQL result |
| Delete a table | `data.drop_table` | Deletion result after explicit confirmation |
| Browse knowledge | `knowledge.toc` / `knowledge.search` | Hierarchy, summaries, applicability, relevance |
| Resolve a method | `knowledge.open` / `knowledge.expand` | Method details, assumptions, risks, examples |
| Preserve traceability | `knowledge.cite` | Versioned document references |

Use `describe` for schema metadata and `query` for concrete aggregation or metadata SQL. The data utility has four operations: CSV import, table description, read-only SQL query, and explicitly confirmed table deletion.

## Required candidate output

The model must return an executable program plus structured metadata:

```json
{
  "program": "...",
  "features": [
    {
      "name": "feature_name",
      "source_columns": ["..."],
      "logic": "...",
      "time_window": "...",
      "query_ids": ["..."],
      "knowledge_refs": ["..."],
      "assumptions": ["..."]
    }
  ],
  "dataset": {"id": "...", "version": "..."},
  "risks": ["..."],
  "validation_plan": ["..."],
  "program_hash": "..."
}
```

## Hard constraints

- Data queries are read-only `SELECT` statements with bounded result rows and SQL length.
- CSV import creates or replaces the filename-derived table in the configured DuckDB file and adds an `id` column.
- Data SQL is read-only. The model must not create, modify, or drop tables through `data.query`.
- Never return a full large table to the model. Aggregate, sample, paginate, or summarize it.
- Query results include returned columns and truncation status.
- Never use future information, labels, evaluation rows, or post-outcome fields unless the task explicitly defines them as valid inputs.
- Never execute arbitrary shell, network, file-write, extension-loading, or unsafe database operations from a model-generated query or program.
- Validate referenced columns, output schema, NaN/Inf behavior, determinism, memory, and runtime before evaluation.
- Knowledge is advisory evidence: preserve source, version, confidence, applicability, and known failure modes. Do not promote generated text directly into trusted knowledge.
- Treat a replacement import as a new table state; confirm before replacing an existing table.

## Stop conditions

Stop and request missing information when the dataset/version, label semantics, temporal ordering, or output contract cannot be established. Do not compensate by inventing schema or silently selecting a different evaluation protocol.

## Common mistakes

- Sending raw rows or whole documents into context instead of progressive summaries.
- Choosing an algorithm before checking cardinality, missingness, time order, and label leakage.
- Treating a successful SQL query as proof that a feature is valid.
- Omitting data and knowledge provenance from the candidate.
- Adding a feature without a validation plan or failure-mode check.
