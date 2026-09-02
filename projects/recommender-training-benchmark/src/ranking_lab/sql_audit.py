from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any


def run_sql_audit(
    ratings_csv: Path,
    movies_csv: Path,
    audit_sql: Path,
) -> dict[str, Any]:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE ratings ("
        "user_id INTEGER NOT NULL, movie_id INTEGER NOT NULL, "
        "rating REAL NOT NULL, timestamp INTEGER NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE movies ("
        "movie_id INTEGER NOT NULL, title TEXT NOT NULL, genres TEXT NOT NULL)"
    )

    with ratings_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = (
            (int(row["userId"]), int(row["movieId"]), float(row["rating"]), int(row["timestamp"]))
            for row in reader
        )
        connection.executemany("INSERT INTO ratings VALUES (?, ?, ?, ?)", rows)

    with movies_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = (
            (int(row["movieId"]), row["title"], row["genres"])
            for row in reader
        )
        connection.executemany("INSERT INTO movies VALUES (?, ?, ?)", rows)

    query = audit_sql.read_text(encoding="utf-8")
    cursor = connection.execute(query)
    columns = [description[0] for description in cursor.description]
    values = cursor.fetchone()
    connection.close()
    return dict(zip(columns, values, strict=True))
