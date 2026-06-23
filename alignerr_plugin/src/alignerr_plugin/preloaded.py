"""Content-addressed deploy-time mount substrate.

Large datasets and model weights are mounted read-only at deploy time instead of
baked into per-task images. The flow:

1. The author declares ``[[preloaded_files]]`` in ``task.toml`` (a local
   ``source`` tree or an ``hf_repo``).
2. ``scripts/sync_mount.sh`` packs each source into a content-addressed squashfs
   (``<task>/<name>-<sha256:16>.squashfs``), uploads it once to a shared cache
   prefix (reused across tasks -- no duplication), and stamps the concrete
   ``remote_path`` entries into ``.alignerr/preloaded_files.json``.
3. The exporter reads that manifest and emits ``preloaded_files`` on the Boreal
   problem entry; the platform mounts each at ``local_path`` read-only.

This module holds the pure, testable pieces: content addressing, the manifest
shape, and the agent-facing mount notice. The shell scripts call into the same
naming so producer and consumer agree.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PRELOADED_MANIFEST_PATH = Path(".alignerr") / "preloaded_files.json"


def content_address(path: Path) -> str:
    """Deterministic 16-char content hash of a file or directory tree.

    For a directory, hashes the sorted ``(relpath, bytes)`` of every regular
    file so identical trees produce identical addresses (cross-task dedup).
    """
    digest = hashlib.sha256()
    path = Path(path)
    if path.is_dir():
        files = sorted(
            (p for p in path.rglob("*") if p.is_file()),
            key=lambda p: p.relative_to(path).as_posix(),
        )
        for file in files:
            digest.update(file.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(file.read_bytes())
            digest.update(b"\0")
    else:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def remote_squashfs_name(task_id: str, entry_name: str, address: str) -> str:
    """Content-addressed remote object name (shared cache prefix-relative)."""
    return f"{task_id}/{entry_name}-{address}.squashfs"


# Conventional dataset dirs auto-mounted (read-only) by trusted CI for known
# task types, so the raw dataset is mounted at deploy time instead of baked into
# the per-task image. Each entry is (source-relative-to-problem-dir, mount_path).
# Mirrors the ML_Envs convention (data/public -> /data, data/private ->
# /mcp_server/data) using this template's layout.
AUTO_MOUNT_SPECS: dict[str, list[tuple[str, str]]] = {
    "ml": [
        ("data", "/data"),
        ("scorer/data", "/mcp_server/data"),
    ],
}


def _tree_has_content(path: Path) -> bool:
    """True when a directory holds at least one real file (ignoring ``.gitkeep``)."""
    path = Path(path)
    if not path.is_dir():
        return False
    return any(p.is_file() and p.name != ".gitkeep" for p in path.rglob("*"))


def auto_mount_entries(problem_dir: Path, task_type: str) -> list[tuple[str, str]]:
    """Implicit conventional mounts for a task type (e.g. ``ml``).

    Returns ``[(source_rel, mount_path), ...]`` for each conventional dataset dir
    that actually has content, so trusted CI can pack/upload/mount it read-only
    instead of baking it into the image. Empty/missing trees are skipped (they
    keep the baked-empty fallback). Unknown task types return ``[]``.
    """
    problem_dir = Path(problem_dir)
    specs = AUTO_MOUNT_SPECS.get((task_type or "").strip().lower(), [])
    return [(source_rel, mount_path) for source_rel, mount_path in specs if _tree_has_content(problem_dir / source_rel)]


def manifest_entry(remote_path: str, local_path: str, *, read_only: bool = True) -> dict[str, Any]:
    """One Boreal/Taiga ``preloaded_files`` entry."""
    return {
        "remote_path": remote_path,
        "local_path": local_path,
        "is_read_only": bool(read_only),
    }


def load_preloaded_manifest(problem_dir: Path) -> list[dict[str, Any]]:
    """Return the stamped manifest (``[]`` when none has been produced)."""
    manifest_path = Path(problem_dir) / PRELOADED_MANIFEST_PATH
    if not manifest_path.exists():
        return []
    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, dict):
        data = data.get("preloaded_files", [])
    return [entry for entry in data if isinstance(entry, dict)]


def mount_notice(manifest: list[dict[str, Any]]) -> str:
    """A short agent-facing note describing the read-only mounts (or '')."""
    mounts = [str(entry.get("local_path")) for entry in manifest if entry.get("local_path")]
    if not mounts:
        return ""
    listed = ", ".join(sorted(set(mounts)))
    return (
        "\n\nThe following paths are mounted read-only and provided for you "
        f"(do not attempt to download or modify them): {listed}."
    )
