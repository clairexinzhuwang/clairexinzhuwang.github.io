from __future__ import annotations

import csv
import hashlib
import shutil
import ssl
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Rating:
    user_id: int
    movie_id: int
    rating: float
    timestamp: int


@dataclass
class PreparedData:
    user_ids: tuple[int, ...]
    item_ids: tuple[int, ...]
    train_positives: np.ndarray
    holdout_positive_by_user: dict[int, int]
    rated_items_by_user: tuple[frozenset[int], ...]
    source_rating_count: int
    eligible_user_count: int

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


@dataclass(frozen=True)
class DatasetFiles:
    ratings_csv: Path
    movies_csv: Path
    readme: Path
    archive: Path


def file_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - used only to match GroupLens' published checksum.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(
    url: str,
    destination: Path,
    expected_md5: str,
    allow_insecure_download: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ranking-reliability-lab/0.1"},
    )
    context = ssl._create_unverified_context() if allow_insecure_download else None

    try:
        with urllib.request.urlopen(request, context=context, timeout=90) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
    except urllib.error.URLError as exc:
        temporary.unlink(missing_ok=True)
        if not allow_insecure_download:
            raise RuntimeError(
                "Secure download from GroupLens failed. Check the official URL and TLS "
                "certificate, or rerun with --allow-insecure-download only if you accept "
                "the connection risk. The archive is still checked against GroupLens' "
                "published MD5 value."
            ) from exc
        raise

    observed_md5 = file_md5(temporary)
    if observed_md5 != expected_md5:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"MovieLens archive checksum mismatch: expected {expected_md5}, "
            f"observed {observed_md5}."
        )
    temporary.replace(destination)


def download_and_extract(
    root: Path,
    url: str,
    expected_md5: str,
    allow_insecure_download: bool = False,
) -> DatasetFiles:
    raw_dir = root / "data" / "raw"
    archive = raw_dir / "ml-latest-small.zip"
    extracted_dir = raw_dir / "ml-latest-small"

    if archive.exists():
        observed_md5 = file_md5(archive)
        if observed_md5 != expected_md5:
            raise ValueError(
                f"Cached archive checksum mismatch: expected {expected_md5}, "
                f"observed {observed_md5}. Remove {archive} and rerun."
            )
    else:
        _download(url, archive, expected_md5, allow_insecure_download)

    required_members = {
        "ml-latest-small/ratings.csv",
        "ml-latest-small/movies.csv",
        "ml-latest-small/README.txt",
    }
    if not all((raw_dir / member).exists() for member in required_members):
        with zipfile.ZipFile(archive) as zipped:
            available = set(zipped.namelist())
            missing = required_members - available
            if missing:
                raise ValueError(f"MovieLens archive is missing expected files: {sorted(missing)}")
            for member in sorted(required_members):
                target = (raw_dir / member).resolve()
                if raw_dir.resolve() not in target.parents:
                    raise ValueError(f"Unsafe archive path: {member}")
                zipped.extract(member, raw_dir)

    return DatasetFiles(
        ratings_csv=extracted_dir / "ratings.csv",
        movies_csv=extracted_dir / "movies.csv",
        readme=extracted_dir / "README.txt",
        archive=archive,
    )


def load_ratings(path: Path) -> list[Rating]:
    ratings: list[Rating] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"userId", "movieId", "rating", "timestamp"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"Unexpected ratings columns: {reader.fieldnames}")
        for row in reader:
            ratings.append(
                Rating(
                    user_id=int(row["userId"]),
                    movie_id=int(row["movieId"]),
                    rating=float(row["rating"]),
                    timestamp=int(row["timestamp"]),
                )
            )
    return ratings


def load_movie_ids(path: Path) -> list[int]:
    movie_ids: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"movieId", "title", "genres"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"Unexpected movies columns: {reader.fieldnames}")
        for row in reader:
            movie_ids.append(int(row["movieId"]))
    return movie_ids


def prepare_implicit_data(
    ratings: list[Rating],
    movie_ids: list[int],
    positive_rating_threshold: float,
) -> PreparedData:
    ratings_by_user: dict[int, list[Rating]] = {}
    for rating in ratings:
        ratings_by_user.setdefault(rating.user_id, []).append(rating)

    eligible_user_ids = []
    for user_id, user_ratings in ratings_by_user.items():
        positive_count = sum(r.rating >= positive_rating_threshold for r in user_ratings)
        if positive_count >= 2:
            eligible_user_ids.append(user_id)
    eligible_user_ids.sort()

    all_movie_ids = tuple(sorted(set(movie_ids) | {r.movie_id for r in ratings}))
    user_index = {user_id: index for index, user_id in enumerate(eligible_user_ids)}
    item_index = {movie_id: index for index, movie_id in enumerate(all_movie_ids)}

    train_positives: list[tuple[int, int]] = []
    holdout_positive_by_user: dict[int, int] = {}
    rated_items_by_user: list[frozenset[int]] = []

    for user_id in eligible_user_ids:
        encoded_user = user_index[user_id]
        ordered = sorted(
            ratings_by_user[user_id],
            key=lambda value: (value.timestamp, value.movie_id),
        )
        positives = [r for r in ordered if r.rating >= positive_rating_threshold]
        holdout = positives[-1]
        holdout_positive_by_user[encoded_user] = item_index[holdout.movie_id]
        for rating in positives[:-1]:
            train_positives.append((encoded_user, item_index[rating.movie_id]))
        rated_items_by_user.append(
            frozenset(item_index[rating.movie_id] for rating in ordered)
        )

    positive_array = np.asarray(train_positives, dtype=np.int64)
    if positive_array.ndim != 2 or positive_array.shape[1] != 2:
        raise ValueError("Expected at least one eligible positive interaction.")

    return PreparedData(
        user_ids=tuple(eligible_user_ids),
        item_ids=all_movie_ids,
        train_positives=positive_array,
        holdout_positive_by_user=holdout_positive_by_user,
        rated_items_by_user=tuple(rated_items_by_user),
        source_rating_count=len(ratings),
        eligible_user_count=len(eligible_user_ids),
    )


def build_evaluation_candidates(
    data: PreparedData,
    negatives_per_user: int,
    seed: int,
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    all_items = np.arange(data.n_items, dtype=np.int64)
    candidates: dict[int, np.ndarray] = {}

    for user_index in range(data.n_users):
        rated = np.fromiter(data.rated_items_by_user[user_index], dtype=np.int64)
        available = np.setdiff1d(all_items, rated, assume_unique=False)
        if len(available) < negatives_per_user:
            raise ValueError(
                f"User {data.user_ids[user_index]} has only {len(available)} "
                "available evaluation negatives."
            )
        negatives = rng.choice(available, size=negatives_per_user, replace=False)
        positive = data.holdout_positive_by_user[user_index]
        candidates[user_index] = np.concatenate(
            [np.asarray([positive], dtype=np.int64), negatives]
        )
    return candidates
