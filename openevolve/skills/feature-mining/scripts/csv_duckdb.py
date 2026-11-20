#!/usr/bin/env python3
"""Minimal CSV-to-DuckDB data tool for the feature-mining skill."""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

try:
    import duckdb
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"duckdb is required: {exc}")

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
BLOCKED = re.compile(
    r"(?:;|\b(?:insert|update|delete|copy|create|drop|alter|attach|detach|install|load|export|pragma|call)\b)",
    re.I,
)
LOGGER = logging.getLogger("feature_mining.data")


def emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, default=str))


def configure_logging(level: str) -> None:
    # Keep machine-readable results on stdout and operational logs on stderr.
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def quote_identifier(value: str) -> str:
    if not IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid identifier: {value}")
    return f'"{value}"'


def database_name(db_path: str) -> str:
    """Use the DuckDB file's stem as its logical database name."""
    name = Path(db_path).stem
    if not name:
        raise ValueError("database file path must include a filename")
    return name


def default_database_path(csv_path: str) -> Path:
    """Place the default DuckDB file beside the input CSV, using its stem."""
    return Path(csv_path).expanduser().resolve().with_suffix(".duckdb")


def connect_database(db_path: str) -> tuple[Any, str, bool]:
    """Open a DuckDB file, creating its parent directory and file if needed."""
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    created = not path.exists()
    con = duckdb.connect(str(path))
    LOGGER.info("%s DuckDB database %s at %s", "Created" if created else "Connected to", database_name(str(path)), path)
    return con, database_name(str(path)), created


def create_database(csv_path: str) -> None:
    """Create or connect to the DuckDB file next to a CSV, then close it."""
    db_path = default_database_path(csv_path)
    con, db_name, created = connect_database(str(db_path))
    try:
        emit({"database": db_name, "path": str(db_path), "created": created})
    finally:
        con.close()
        LOGGER.debug("Closed DuckDB database %s", db_name)


def table_name(csv_path: str, requested: str | None) -> str:
    name = requested or Path(csv_path).stem
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not name or name[0].isdigit():
        name = f"t_{name}"
    return name.lower()


def import_csv(csv_path: str, requested_table: str | None = None) -> None:
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() not in {".csv", ".gz"}:
        raise ValueError("input must be an existing CSV file")
    con, db_name, _ = connect_database(str(default_database_path(str(path))))
    try:
        name = table_name(str(path), requested_table)
        qname = quote_identifier(name)
        escaped = str(path).replace("'", "''")
        LOGGER.info("Importing CSV %s into table %s", path.name, name)
        # row_number adds an import-local ID when no source key is provided.
        con.execute(f"DROP TABLE IF EXISTS {qname}")
        con.execute(f"""
            CREATE TABLE {qname} AS
            SELECT row_number() OVER () AS id, source.*
            FROM read_csv_auto('{escaped}', header=true, sample_size=-1) AS source
        """)
        count = con.execute(f"SELECT count(*) FROM {qname}").fetchone()[0]
        columns = [row[0] for row in con.execute(f"DESCRIBE {qname}").fetchall()]
        LOGGER.info("Imported %s rows and %s columns into %s", count, len(columns), name)
        emit({"database": db_name, "table": name, "source": str(path), "rows": count, "columns": columns})
    finally:
        con.close()
        LOGGER.debug("Closed DuckDB database %s", db_name)


def describe(csv_path: str, table: str) -> None:
    target_db = str(default_database_path(csv_path))
    con, db_name, _ = connect_database(target_db)
    try:
        qname = quote_identifier(table)
        rows = con.execute(f"DESCRIBE {qname}").fetchall()
        LOGGER.info("Read metadata for table %s", table)
        emit({"database": db_name, "table": table, "columns": [{"name": r[0], "type": r[1], "nullable": r[2]} for r in rows], "row_count": con.execute(f"SELECT count(*) FROM {qname}").fetchone()[0]})
    finally:
        con.close()
        LOGGER.debug("Closed DuckDB database %s", db_name)


def drop_table(csv_path: str, table: str) -> None:
    target_db = str(default_database_path(csv_path))
    con, db_name, _ = connect_database(target_db)
    try:
        qname = quote_identifier(table)
        exists = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
            [table],
        ).fetchone()[0]
        if not exists:
            raise ValueError(f"table not found: {table}")
        LOGGER.warning("Dropping table %s from database %s", table, db_name)
        con.execute(f"DROP TABLE {qname}")
        emit({"database": db_name, "table": table, "dropped": True})
    finally:
        con.close()
        LOGGER.debug("Closed DuckDB database %s", db_name)


def query(csv_path: str, sql: str, max_rows: int = 100) -> None:
    if BLOCKED.search(sql) or not re.match(r"^\s*select\b", sql, re.I):
        raise ValueError("only one read-only SELECT statement is allowed")
    if len(sql) > 20000:
        raise ValueError("SQL is too long")
    con, db_name, _ = connect_database(str(default_database_path(csv_path)))
    try:
        result = con.execute(sql)
        columns = [item[0] for item in result.description]
        rows = result.fetchmany(max_rows)
        LOGGER.info("Executed read-only query on %s; returned %s rows", db_name, len(rows))
        emit({"database": db_name, "columns": columns, "rows": rows, "truncated": len(rows) == max_rows})
    finally:
        con.close()
        LOGGER.debug("Closed DuckDB database %s", db_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)
    create_db = sub.add_parser("create-database", help="create or connect to the CSV's neighboring DuckDB database")
    create_db.add_argument("csv_path", help="CSV whose directory and filename determine the default database path")
    imp = sub.add_parser("import-csv", help="create a table from a CSV and import all rows")
    imp.add_argument("csv_path")
    imp.add_argument("--table", help="optional table name; defaults to the CSV filename")
    desc = sub.add_parser("describe", help="show table metadata")
    desc.add_argument("csv_path")
    desc.add_argument("table")
    qry = sub.add_parser("query", help="run a read-only SQL statistic or metadata query")
    qry.add_argument("csv_path")
    qry.add_argument("sql")
    qry.add_argument("--max-rows", type=int, default=100)
    drop = sub.add_parser("drop-table", help="delete a table from DuckDB")
    drop.add_argument("csv_path")
    drop.add_argument("table")
    drop.add_argument("--confirm", action="store_true", help="confirm permanent table deletion")
    args = parser.parse_args()
    configure_logging(args.log_level)
    try:
        if args.command == "create-database":
            create_database(args.csv_path)
        elif args.command == "import-csv":
            import_csv(args.csv_path, args.table)
        elif args.command == "describe":
            describe(args.csv_path, args.table)
        elif args.command == "query":
            query(args.csv_path, args.sql, args.max_rows)
        else:
            if not args.confirm:
                raise ValueError("table deletion requires --confirm")
            drop_table(args.csv_path, args.table)
    except (duckdb.Error, OSError, ValueError) as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        emit({"error": type(exc).__name__, "message": str(exc)})
        raise SystemExit(2)


if __name__ == "__main__":
    main()
