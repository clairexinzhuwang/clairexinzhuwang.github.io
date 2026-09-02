#!/usr/bin/env python3
"""Deterministic provenance identities for current-code evidence.

The producer-source manifest is deliberately source-only: it is computed before
any gate or experiment output exists, so an output cannot certify itself.  A
separate final-package checksum manifest is created after the audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_EXACT = {
    "FORMAL_GRID.json",
    "requirements.txt",
}
SOURCE_SUFFIXES = {".py", ".sh", ".tex", ".bib"}
SOURCE_SUBDIRS = {"src", "DEVELOPMENT_BENCHMARKS"}
VERSION_SCAN_SUFFIXES = {".py", ".json", ".sh", ".tex", ".bib", ".toml", ".yaml", ".yml", ".cfg"}
VERSION_TOKEN = re.compile(r"v26\.1[a-z]?(?:-[A-Za-z0-9._-]+)?")
OLD_METHOD_VERSION_AUDIT_SENTINEL = "v26.1-selected-support-gaussian-max-monte-carlo-rank"
PREDECLARED_AUDIT_DECISION_RULES_SHA256 = (
    "ba6d0d8898ba7e7393f4a001ecf907149a0b43a2fc7b82abb2bf5e118bae8a88"
)
EXECUTION_PROVENANCE_FIELDS = (
    "producer_source_manifest_sha256",
    "runtime_fingerprint_sha256",
    "source_formal_grid_sha256",
    "execution_grid_sha256",
    "execution_grid_hash_scope",
    "audit_decision_rules_sha256",
    "seed_epoch",
    "formal_seeds_opened",
    "execution_identity_sha256",
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using one stable canonical encoding."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def strict_json_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON values without Python's ``0 == False`` type aliasing."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (
            set(actual) == set(expected)
            and all(strict_json_equal(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            strict_json_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting ambiguous duplicate keys."""
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = child
    return value


def strict_json_loads(text: str) -> Any:
    """Parse standards-compliant JSON, rejecting duplicates and non-finites."""
    return json.loads(
        text,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {token!r}")
        ),
        object_pairs_hook=_reject_duplicate_json_pairs,
    )


def strict_json_load(
    path: str | Path, *, require_object: bool = False, reject_symlink: bool = False,
) -> Any:
    """Strictly parse a JSON file, optionally enforcing evidence-file shape."""
    path = Path(path)
    absolute = path.absolute()
    aliased = reject_symlink and (absolute.is_symlink() or absolute.resolve() != absolute)
    if not path.is_file() or aliased:
        qualifier = "regular non-symlink " if reject_symlink else "regular "
        raise FileNotFoundError(f"required {qualifier}JSON is missing: {path}")
    value = strict_json_loads(path.read_text())
    if require_object and not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def bound_input_manifest(paths: list[str | Path]) -> dict[str, Any]:
    """Bind a verifier/aggregator invocation to the exact sorted input files."""
    resolved = sorted({Path(path).resolve() for path in paths}, key=lambda p: p.as_posix())
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    entries = [
        {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in resolved
    ]
    return {
        "manifest_kind": "exact-input-files",
        "entries": entries,
        "entry_count": len(entries),
        "sha256": canonical_sha256(entries),
    }


def producer_source_paths(root: str | Path = ROOT) -> list[Path]:
    """Return the executable/configuration files that can affect evidence."""
    root = Path(root).resolve()
    paths: set[Path] = set()
    for path in root.iterdir():
        if path.is_file() and (path.name in SOURCE_EXACT or path.suffix in SOURCE_SUFFIXES):
            paths.add(path)
    for subdir in SOURCE_SUBDIRS:
        base = root / subdir
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in SOURCE_SUFFIXES and "__pycache__" not in path.parts:
                paths.add(path)
    return sorted(paths, key=lambda p: p.relative_to(root).as_posix())


def producer_source_manifest(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in producer_source_paths(root)
    ]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "manifest_kind": "producer-source-only",
        "manifest_scope": (
            "all root Python/shell/TeX/BibTeX, src Python, development benchmark "
            "Python, FORMAL_GRID.json, requirements.txt"
        ),
        "entries": entries,
        "entry_count": len(entries),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def declared_version_inventory(root: str | Path = ROOT) -> dict[str, Any]:
    """Inventory version mentions in active code/config/manuscripts (not historical Markdown/patches)."""
    root = Path(root).resolve()
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in VERSION_SCAN_SUFFIXES:
            continue
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(errors="replace")
            tokens = sorted(set(VERSION_TOKEN.findall(text)))
        except OSError:
            continue
        if tokens:
            relative = path.relative_to(root).as_posix()
            sentinel_lines = [
                number for number, line in enumerate(text.splitlines(), 1)
                if relative == "audit_provenance.py"
                and "OLD_METHOD_VERSION_AUDIT_SENTINEL" in line
                and OLD_METHOD_VERSION_AUDIT_SENTINEL in line
            ]
            active_old_lines = [
                number for number, line in enumerate(text.splitlines(), 1)
                if OLD_METHOD_VERSION_AUDIT_SENTINEL in line and number not in sentinel_lines
            ]
            rows.append({
                "path": relative, "version_mentions": tokens,
                "old_identity_audit_sentinel_lines": sentinel_lines,
                "active_old_identity_lines": active_old_lines,
            })
    return {
        "files_with_version_mentions": rows,
        "old_method_identity_files": [
            row["path"] for row in rows if row["active_old_identity_lines"]
        ],
        "old_method_identity_audit_sentinel": OLD_METHOD_VERSION_AUDIT_SENTINEL,
    }


def runtime_identity() -> dict[str, Any]:
    import numpy as np
    import psutil
    import scipy

    def tool_version(name: str) -> dict[str, Any]:
        executable = shutil.which(name)
        if executable is None:
            return {"executable": None, "version": None}
        try:
            first_line = subprocess.check_output(
                [executable, "--version"], text=True, stderr=subprocess.STDOUT
            ).splitlines()[0]
        except Exception as exc:  # pragma: no cover - environment diagnostic
            first_line = f"{type(exc).__name__}: {exc}"
        return {"executable": str(Path(executable).resolve()), "version": first_line}

    stable = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "psutil": psutil.__version__,
        "pdflatex": tool_version("pdflatex"),
        "bibtex": tool_version("bibtex"),
        "blas_threads": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return {**stable, "sha256": hashlib.sha256(encoded).hexdigest()}


def git_identity(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    if not (root / ".git").exists():
        return {"present": False, "commit": None, "dirty": None}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"present": True, "commit": commit, "dirty": dirty}
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return {"present": True, "commit": None, "dirty": None, "error": f"{type(exc).__name__}: {exc}"}


def current_identity(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    from src.stochastic_screen_sgd import METHOD_VERSION

    grid = root / "FORMAL_GRID.json"
    package = root / "PACKAGE_INFO_V26_1D.json"
    declared_package = (
        strict_json_load(package, require_object=True, reject_symlink=True)
        if package.is_file() else {}
    )
    manifest = producer_source_manifest(root)
    runtime = runtime_identity()
    git = git_identity(root)
    return {
        "method_version": METHOD_VERSION,
        "formal_grid_sha256": sha256_file(grid),
        "producer_source_manifest_sha256": manifest["sha256"],
        "producer_source_manifest": manifest,
        "declared_version_inventory": declared_version_inventory(root),
        "runtime_identity": runtime,
        "git": git,
        "package_declared_git_commit": declared_package.get("git_commit"),
        "package_declared_git_commit_verified": bool(
            git.get("present")
            and git.get("commit") == declared_package.get("git_commit")
        ),
        "formal_seeds_opened": False,
    }


def execution_provenance(
    execution_grid: str | Path,
    *,
    seed_epoch: str,
    formal_seeds_opened: bool,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Return the immutable provenance fields required on every result row."""
    root = Path(root).resolve()
    manifest = producer_source_manifest(root)
    runtime = runtime_identity()
    stable_fields = {
        "producer_source_manifest_sha256": manifest["sha256"],
        "runtime_fingerprint_sha256": runtime["sha256"],
        "source_formal_grid_sha256": sha256_file(root / "FORMAL_GRID.json"),
        "execution_grid_sha256": sha256_file(execution_grid),
        "audit_decision_rules_sha256": PREDECLARED_AUDIT_DECISION_RULES_SHA256,
        "seed_epoch": str(seed_epoch),
    }
    # Whether a seed has already been consumed is audit state, not a namespace key.
    # Excluding it makes a no-draw start check predict the exact output namespace
    # that a later authorized run will use, without falsely claiming a draw occurred.
    return {
        **stable_fields,
        "execution_grid_hash_scope": "caller_supplied_grid_file",
        "formal_seeds_opened": bool(formal_seeds_opened),
        "execution_identity_sha256": canonical_sha256(stable_fields),
    }


def validate_execution_provenance(
    record: dict[str, Any], expected: dict[str, Any], *, label: str = "record"
) -> None:
    """Fail closed unless a cache/result carries the complete current identity."""
    missing = [field for field in EXECUTION_PROVENANCE_FIELDS if field not in record]
    if missing:
        raise ValueError(f"{label} is missing provenance fields: {missing}")
    mismatches = {
        field: {"expected": expected.get(field), "actual": record.get(field)}
        for field in EXECUTION_PROVENANCE_FIELDS
        if not strict_json_equal(record.get(field), expected.get(field))
    }
    if mismatches:
        raise ValueError(f"{label} provenance mismatch: {mismatches}")


def atomic_write_json(path: str | Path, payload: Any, *, overwrite: bool = False) -> None:
    """Durably publish JSON only after a complete, finite serialization exists."""
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.parent.resolve() != path.parent:
        raise ValueError(f"refusing to publish through an aliased parent directory: {path.parent}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite immutable artifact {path}")
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    temporary_name = path.name + ".partial." + uuid.uuid4().hex
    fd = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(encoded)
            fh.flush()
            os.fsync(fh.fileno())
        if overwrite:
            # Mutable, non-authorization diagnostics may explicitly request an
            # atomic replacement.  Immutable evidence takes the no-overwrite
            # branch below.
            os.replace(
                temporary_name, path.name,
                src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
            )
        else:
            # ``exists(); replace()`` has a race in which a concurrent writer
            # can be overwritten between the two operations.  A same-directory
            # hard link is an atomic create-if-absent publication primitive:
            # it fails with FileExistsError if any entry already owns ``path``.
            os.link(
                temporary_name, path.name,
                src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        os.fsync(directory_fd)
        if not overwrite:
            os.unlink(temporary_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
    except Exception:
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(directory_fd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    payload = current_identity(args.root)
    output = Path(args.output)
    atomic_write_json(output, payload)
    print(json.dumps({k: payload[k] for k in (
        "method_version", "formal_grid_sha256", "producer_source_manifest_sha256",
        "formal_seeds_opened",
    )}, indent=2))


if __name__ == "__main__":
    main()
