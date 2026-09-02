from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ranking_lab.sql_audit import run_sql_audit


class SQLAuditTest(unittest.TestCase):
    def test_audit_detects_duplicate_user_movie_rows(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            ratings = temporary / "ratings.csv"
            movies = temporary / "movies.csv"

            with ratings.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["userId", "movieId", "rating", "timestamp"])
                writer.writerow([1, 10, 4.0, 100])
                writer.writerow([1, 10, 4.5, 101])
                writer.writerow([2, 11, 5.0, 102])
            with movies.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["movieId", "title", "genres"])
                writer.writerow([10, "A", "Drama"])
                writer.writerow([11, "B", "Comedy"])

            audit = run_sql_audit(ratings, movies, root / "sql" / "audit_ratings.sql")

        self.assertEqual(audit["rating_rows"], 3)
        self.assertEqual(audit["duplicate_user_movie_rows"], 1)
        self.assertEqual(audit["out_of_range_ratings"], 0)
        self.assertEqual(audit["invalid_half_star_steps"], 0)


if __name__ == "__main__":
    unittest.main()

