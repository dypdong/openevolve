# CSV and DuckDB policy

The data tool has one responsibility: import a user-provided CSV into DuckDB and answer read-only metadata/statistics queries.

## Lifecycle

1. Every operation accepts a CSV path and opens/closes `<csv-directory>/<csv-stem>.duckdb` internally.
2. `create-database <csv-path>` creates or connects to that neighboring database file.
3. `import-csv` reads the header, infers DuckDB types, creates a table named from the filename, adds an `id` column, and imports all rows.
4. `describe` returns the table columns, types, nullability, and row count.
5. `query` executes bounded read-only `SELECT` statements for metadata and statistics.
6. `drop-table` deletes one table and requires explicit confirmation.

The DuckDB file is the single source of truth. Re-importing the same table replaces it explicitly; it does not maintain a separate catalog or version service. `import-csv` derives the same neighboring database path unless `--db` overrides it.

## Safety

- Accept only approved local CSV files; never let model SQL choose a filesystem path.
- Table identifiers are validated and quoted.
- Query accepts one `SELECT` statement only; reject DDL, DML, file I/O, extension loading, and multiple statements.
- Limit result rows and SQL length. Prefer `count`, `min`, `max`, `avg`, quantiles, histograms, and grouped statistics.
- Do not return the full CSV to the model. Use aggregate results or small samples.
- Keep the `id` column stable for one import; treat a replacement import as a new table state.
- Keep operational logs on stderr and machine-readable JSON results on stdout.
- Never route table deletion through `query`; use the dedicated confirmed `drop-table` operation.
