#!/usr/bin/env python3
"""Immutable formal task plan, start capability, and output-namespace checks."""
from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterable

from audit_provenance import (
    canonical_sha256, sha256_file, strict_json_equal, strict_json_load,
)
import seed_registry as SR
from src.stochastic_screen_sgd import METHOD_VERSION


FORMAL_CHUNK_SIZE = 250
FORMAL_TOTAL_TASKS = 720
START_CAPABILITY_NAME = ".formal_start_capability.json"
START_CAPABILITY_SCHEMA = "formal-start-capability-v1"

# v26.1e disjoint outcome taxonomy and HTP termination contract.  Every route
# record carries exactly one status; full rows (a completed replication) may
# be ok/selection_failure_empty/numerical_failure, error-shaped rows may be
# numerical_failure (an HTP certification failure with machine-readable
# context) or program_or_schema_error (any other exception/schema problem).
ROW_STATUSES = (
    "ok", "selection_failure_empty", "numerical_failure",
    "program_or_schema_error",
)
FULL_ROW_STATUSES = ("ok", "selection_failure_empty", "numerical_failure")
ERROR_ROW_STATUSES = ("numerical_failure", "program_or_schema_error")
TERMINATION_REASONS = ("stable_certified", "cap_certified", "uncertified_failure")
CERTIFIED_TERMINATION_REASONS = ("stable_certified", "cap_certified")


def outcome_taxonomy(
    *, pipeline_numerically_valid: bool, selection_nonempty: bool,
    strict_recovery_and_coverage: bool,
) -> dict[str, Any]:
    """Classify one completed-run route record under the disjoint taxonomy.

    An empty selection with a numerically valid pipeline is a valid total
    statistical failure (strict noncoverage, pinned empty Monte Carlo fields,
    no interval width), never a numerical failure.
    """
    if not pipeline_numerically_valid:
        status = "numerical_failure"
    elif not selection_nonempty:
        status = "selection_failure_empty"
    else:
        status = "ok"
    return {
        "status": status,
        "pipeline_numerically_valid": bool(pipeline_numerically_valid),
        "selection_nonempty": bool(selection_nonempty),
        "interval_reportable": bool(pipeline_numerically_valid and selection_nonempty),
        "strict_recovery_and_coverage": bool(strict_recovery_and_coverage),
        "total_failure": status != "ok",
    }

# This is an execution contract, not a tuning knob.  Gate-3 certifies this
# exact profile; Gate-4 and the no-draw capability bind its canonical hash.
FORMAL_EXECUTION_PROFILE = {
    "schema_version": "formal-execution-profile-v1",
    "array_start": 0,
    "array_end": FORMAL_TOTAL_TASKS - 1,
    "array_concurrency": 50,
    "chunk_size": FORMAL_CHUNK_SIZE,
    "total_tasks": FORMAL_TOTAL_TASKS,
    "replications_per_task": FORMAL_CHUNK_SIZE,
    "task_time_limit_seconds": 2 * 60 * 60,
    "task_memory_limit_gib": 4.0,
    "pretimeout_signal": "USR1",
    "pretimeout_lead_seconds": 5 * 60,
    "requeue_enabled": True,
    "resume_enabled": True,
    "single_rep_wall_ceiling_seconds": 120.0,
    "full_task_wall_safety_factor": 1.25,
    "measured_control_wall_safety_factor": 1.25,
    "full_task_rss_safety_factor": 1.25,
}
FORMAL_EXECUTION_PROFILE_SHA256 = canonical_sha256(FORMAL_EXECUTION_PROFILE)


_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def _open_absolute_directory_no_follow(path: Path) -> int:
    """Pin an absolute directory through an ``openat`` no-follow chain."""
    absolute = path.absolute()
    if not absolute.is_absolute():
        raise ValueError("root-anchored directory must be absolute")
    current_fd = os.open("/", _DIRECTORY_OPEN_FLAGS)
    try:
        for component in absolute.parts[1:]:
            if component in {"", ".", ".."}:
                raise ValueError("unsafe root-anchored path component")
            next_fd = os.open(
                component, _DIRECTORY_OPEN_FLAGS, dir_fd=current_fd,
            )
            os.close(current_fd)
            current_fd = next_fd
        if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
            raise ValueError("root-anchored path is not a directory")
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


class RootAnchoredReader:
    """Read regular descendants from one pinned, no-follow root descriptor.

    Path resolution is used only to reject an aliased initial root.  Every
    descendant is subsequently opened component by component relative to the
    pinned root descriptor, so a rename/symlink swap in an ancestor cannot
    redirect a verified formal read outside the frozen namespace.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).absolute()
        if (
            not self.root.is_dir() or self.root.is_symlink()
            or self.root.resolve() != self.root
        ):
            raise ValueError("root-anchored reader requires an unaliased directory")
        self.fd = _open_absolute_directory_no_follow(self.root)
        root_stat = os.fstat(self.fd)
        self.identity = (root_stat.st_dev, root_stat.st_ino)
        self.assert_path_binding()

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> "RootAnchoredReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def assert_path_binding(self) -> None:
        if self.fd < 0:
            raise RuntimeError("root-anchored reader is closed")
        try:
            current = os.stat(self.root, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise RuntimeError("pinned root path disappeared") from exc
        if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
            raise RuntimeError("pinned root path became aliased/non-directory")
        if (current.st_dev, current.st_ino) != self.identity:
            raise RuntimeError("pinned root path was replaced")

    def _relative_parts(self, path: str | Path) -> tuple[str, ...]:
        absolute = Path(path).absolute()
        try:
            relative = absolute.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"path escapes pinned root: {absolute}") from exc
        parts = relative.parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("unsafe or empty pinned-root relative path")
        return parts

    def open_regular(self, path: str | Path) -> int:
        self.assert_path_binding()
        parts = self._relative_parts(path)
        directory_fd = os.dup(self.fd)
        try:
            for component in parts[:-1]:
                next_fd = os.open(
                    component, _DIRECTORY_OPEN_FLAGS, dir_fd=directory_fd,
                )
                os.close(directory_fd)
                directory_fd = next_fd
            file_fd = os.open(
                parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        finally:
            os.close(directory_fd)
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            os.close(file_fd)
            raise ValueError(f"pinned-root input is not regular: {path}")
        self.assert_path_binding()
        return file_fd

    def read_regular(self, path: str | Path) -> bytes:
        fd = self.open_regular(path)
        try:
            before = os.fstat(fd)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(fd)
            identity_before = (
                before.st_dev, before.st_ino, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns,
            )
            identity_after = (
                after.st_dev, after.st_ino, after.st_size,
                after.st_mtime_ns, after.st_ctime_ns,
            )
            data = b"".join(chunks)
            if identity_before != identity_after or len(data) != before.st_size:
                raise RuntimeError(f"pinned-root input changed while read: {path}")
            return data
        finally:
            os.close(fd)


def require_path_outside_namespace(
    path: str | Path, namespace_root: str | Path, *, label: str,
) -> Path:
    """Return one unaliased absolute path strictly outside a frozen namespace."""
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError(f"{label} must be an explicit absolute path")
    candidate = candidate.absolute()
    resolved_candidate = candidate.resolve(strict=False)
    root = Path(namespace_root)
    if not root.is_absolute():
        raise ValueError("frozen namespace root must be absolute")
    root = root.absolute()
    resolved_root = root.resolve(strict=False)
    if resolved_candidate != candidate:
        raise ValueError(f"{label} or one of its parent components is symlinked/aliased")
    if resolved_root != root:
        raise ValueError("frozen namespace root is symlinked/aliased")
    if candidate == root or root in candidate.parents:
        raise ValueError(f"{label} must be outside the frozen output namespace")
    return candidate


def task_plan(
    grid: dict[str, Any], *, execution_grid_sha256: str,
    chunk_size: int = FORMAL_CHUNK_SIZE,
) -> dict[str, Any]:
    """Build the exact canonical task partition without importing the runner."""
    if type(chunk_size) is not int or chunk_size != FORMAL_CHUNK_SIZE:
        raise ValueError(f"formal chunk size must be exactly {FORMAL_CHUNK_SIZE}")
    if not isinstance(execution_grid_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", execution_grid_sha256,
    ) is None:
        raise ValueError("formal task plan requires the exact execution-grid SHA-256")
    cells = grid.get("cells")
    if not isinstance(cells, list) or len(cells) != SR.FORMAL_CELLS:
        raise ValueError(f"formal task plan requires exactly {SR.FORMAL_CELLS} cells")
    mappings: list[dict[str, Any]] = []
    seen_cell_ids: set[str] = set()
    for position, cell in enumerate(cells):
        if not isinstance(cell, dict) or type(cell.get("cell_index")) is not int:
            raise ValueError(f"cell {position} has an invalid identity")
        if cell["cell_index"] != position:
            raise ValueError(f"cell {position} is not at its canonical index")
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id or cell_id in seen_cell_ids:
            raise ValueError(f"cell {position} has an invalid or duplicate cell ID")
        seen_cell_ids.add(cell_id)
        reps = cell.get("replications")
        if type(reps) is not int or reps != SR.FORMAL_REPLICATIONS:
            raise ValueError(f"cell {position} has an invalid replication count")
        for start in range(0, reps, chunk_size):
            mappings.append({
                "task_id": len(mappings),
                "cell_index_in_grid": position,
                "cell_id": cell_id,
                "rep_start": start,
                "rep_end": min(reps, start + chunk_size),
                "chunk_size": chunk_size,
                "route_records_expected": 2 * (min(reps, start + chunk_size) - start),
            })
    if len(mappings) != FORMAL_TOTAL_TASKS:
        raise ValueError(f"formal task plan must contain exactly {FORMAL_TOTAL_TASKS} tasks")
    stable = {
        "schema_version": "formal-task-plan-v1",
        "method_version": METHOD_VERSION,
        "execution_grid_sha256": execution_grid_sha256,
        "chunk_size": chunk_size,
        "total_tasks": len(mappings),
        "mappings": mappings,
    }
    return {**stable, "sha256": canonical_sha256(stable)}


def normalize_start_capability(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(payload)
    normalized.pop("formal_start_capability_sha256", None)
    return normalized


def start_capability_sha256(payload: dict[str, Any]) -> str:
    return canonical_sha256(normalize_start_capability(payload))


def validate_start_capability_payload(
    payload: Any, *, frozen: dict[str, Any], execution_provenance: dict[str, Any],
    task_plan_payload: dict[str, Any], output_root: Path,
) -> None:
    if not isinstance(payload, dict):
        raise RuntimeError("formal start capability is not an object")
    expected = {
        "schema_version": START_CAPABILITY_SCHEMA,
        "version": METHOD_VERSION,
        "pass": True,
        "freeze_id": frozen["freeze_id"],
        "formal_output_root": str(output_root),
        "formal_seeds_opened": False,
        "no_draw_start_check": True,
        "chunk_size": FORMAL_CHUNK_SIZE,
        "total_tasks": FORMAL_TOTAL_TASKS,
        "formal_task_plan_sha256": task_plan_payload["sha256"],
        "formal_execution_profile_sha256": FORMAL_EXECUTION_PROFILE_SHA256,
        **execution_provenance,
    }
    expected_keys = set(expected) | {"created_utc", "formal_start_capability_sha256"}
    if set(payload) != expected_keys:
        raise RuntimeError(
            "formal start capability has the wrong exact schema keys: "
            f"missing={sorted(expected_keys - set(payload))}, "
            f"unexpected={sorted(set(payload) - expected_keys)}"
        )
    mismatches = {
        field: {"expected": value, "actual": payload.get(field)}
        for field, value in expected.items()
        if field not in payload or not strict_json_equal(payload[field], value)
    }
    if mismatches:
        raise RuntimeError(f"formal start capability identity mismatch: {mismatches}")
    if not isinstance(payload.get("created_utc"), str) or not payload["created_utc"]:
        raise RuntimeError("formal start capability lacks its creation timestamp")
    try:
        created = datetime.fromisoformat(payload["created_utc"])
    except ValueError as exc:
        raise RuntimeError("formal start capability has an invalid creation timestamp") from exc
    if created.tzinfo is None or created.utcoffset() is None:
        raise RuntimeError("formal start capability timestamp is not timezone-aware")
    expected_sha = start_capability_sha256(payload)
    if payload.get("formal_start_capability_sha256") != expected_sha:
        raise RuntimeError("formal start capability self-hash mismatch")


def expected_task_relative_paths(
    task_plan_payload: dict[str, Any], *, freeze_id: str,
    execution_identity_sha256: str,
) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for mapping in task_plan_payload["mappings"]:
        task_id = int(mapping["task_id"])
        name = (
            f"task_{task_id:04d}_reps_{int(mapping['rep_start']):04d}_"
            f"{int(mapping['rep_end']):04d}.jsonl"
        )
        out[task_id] = (
            Path(freeze_id) / execution_identity_sha256
            / str(mapping["cell_id"]) / name
        )
    return out


def validate_formal_namespace(
    output_root: Path, *, task_plan_payload: dict[str, Any], freeze_id: str,
    execution_identity_sha256: str, allow_inflight: bool,
    required_final_task_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Reject every entry outside the frozen task namespace.

    The scan is descriptor-rooted and performs one ``lstat`` per entry.  A
    concurrently removed *allowed* partial/lock is simply absent from this
    snapshot; it is never misclassified as a non-regular foreign entry.
    Unexpected entries that vanish during inspection cause a bounded retry.
    """
    root = output_root.absolute()
    if not root.is_dir() or root.is_symlink() or root.resolve() != root:
        raise RuntimeError("formal output root is not an unaliased directory")
    expected = expected_task_relative_paths(
        task_plan_payload, freeze_id=freeze_id,
        execution_identity_sha256=execution_identity_sha256,
    )
    expected_files = set(expected.values())
    allowed_files = {Path(START_CAPABILITY_NAME), *expected_files}
    if allow_inflight:
        for path in expected_files:
            partial = Path(path.as_posix() + ".partial")
            allowed_files.add(partial)
            allowed_files.add(Path(partial.as_posix() + ".resume.lock"))
    allowed_dirs: set[Path] = set()
    for relative in allowed_files:
        parent = relative.parent
        while parent != Path("."):
            allowed_dirs.add(parent)
            parent = parent.parent

    def scan(reader: RootAnchoredReader) -> tuple[set[Path], list[str], bool]:
        actual_files: set[Path] = set()
        unexpected: list[str] = []
        retry = False

        def visit(directory_fd: int, relative_dir: Path) -> None:
            nonlocal retry
            with os.scandir(directory_fd) as entries:
                snapshot = list(entries)
            for entry in snapshot:
                relative = relative_dir / entry.name
                try:
                    info = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    # Partial/lock churn is normal under a live array.  A
                    # disappearing unknown entry is rescanned before deciding.
                    if relative not in allowed_files and relative not in allowed_dirs:
                        retry = True
                    continue
                if stat.S_ISLNK(info.st_mode):
                    unexpected.append(relative.as_posix() + " [symlink/alias]")
                elif stat.S_ISDIR(info.st_mode):
                    if relative not in allowed_dirs:
                        unexpected.append(relative.as_posix() + " [directory]")
                        continue
                    try:
                        child_fd = os.open(
                            entry.name, _DIRECTORY_OPEN_FLAGS, dir_fd=directory_fd,
                        )
                    except FileNotFoundError:
                        continue
                    try:
                        pinned = os.fstat(child_fd)
                        if (pinned.st_dev, pinned.st_ino) != (info.st_dev, info.st_ino):
                            retry = True
                            continue
                        visit(child_fd, relative)
                    finally:
                        os.close(child_fd)
                elif stat.S_ISREG(info.st_mode):
                    actual_files.add(relative)
                    if relative not in allowed_files:
                        unexpected.append(relative.as_posix())
                else:
                    unexpected.append(relative.as_posix() + " [non-regular]")
        visit(reader.fd, Path("."))
        reader.assert_path_binding()
        return actual_files, unexpected, retry

    with RootAnchoredReader(root) as reader:
        for _attempt in range(8):
            actual_files, unexpected, retry = scan(reader)
            if not retry:
                break
        else:
            raise RuntimeError("formal output namespace did not stabilize during audit")
    if unexpected:
        raise RuntimeError(f"formal output namespace contains foreign entries: {unexpected[:50]}")
    if Path(START_CAPABILITY_NAME) not in actual_files:
        raise RuntimeError("formal output namespace lacks the start capability")
    if required_final_task_ids is not None:
        required_ids = list(required_final_task_ids)
        if any(type(task_id) is not int or task_id not in expected for task_id in required_ids):
            raise RuntimeError("required formal task IDs are invalid")
        missing = [task_id for task_id in required_ids if expected[task_id] not in actual_files]
        if missing:
            raise RuntimeError(f"formal output namespace is missing task outputs: {missing[:50]}")
    return {
        "files": sorted(path.as_posix() for path in actual_files),
        "expected_task_paths": {task_id: path.as_posix() for task_id, path in expected.items()},
    }


def load_start_capability(path: Path) -> dict[str, Any]:
    return strict_json_load(path, require_object=True, reject_symlink=True)


def same_capability_files(source_path: Path, output_root: Path) -> dict[str, Any]:
    token_path = output_root / START_CAPABILITY_NAME
    source_before = sha256_file(source_path)
    token_before = sha256_file(token_path)
    source = load_start_capability(source_path)
    token = load_start_capability(token_path)
    source_after = sha256_file(source_path)
    token_after = sha256_file(token_path)
    parsed_payload_sha = hashlib.sha256(
        (json.dumps(source, indent=2, allow_nan=False) + "\n").encode("utf-8")
    ).hexdigest()
    if (
        source_before != source_after
        or token_before != token_after
        or source_before != token_before
        or source_after != parsed_payload_sha
        or not strict_json_equal(source, token)
    ):
        raise RuntimeError("source and output-root formal start capabilities differ")
    return source
