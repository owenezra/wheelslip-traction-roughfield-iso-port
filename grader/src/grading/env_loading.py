"""Load TRUSTED task-author Python modules from ``/mcp_server/data/`` via importlib.

For loading agent-submitted code (e.g. ``/tmp/output/policy.py``), use
:func:`grading.helpers.load_submitted_policy` instead -- that path runs in a
sandboxed worker that drops privileges to the unprivileged ``agent`` account and
isolates stdout/stderr. Calling raw importlib on agent code (or this helper)
gives the agent in-process execution as root with live stdout, which lets them
read ``/mcp_server/data/`` and forge a grade.

This helper is for the OTHER case: loading code that ships with the task at
build time, lives under ``/mcp_server/data/`` (root-only, agent can't read it),
and is part of the trusted grader -- typically the hidden environment module
(``env.py`` defining ``make_env``) used by ``[environment].hidden_env`` tasks
and by the ``env_server`` package. No sandboxing is needed; this just
deduplicates the ``importlib.util.spec_from_file_location`` incantation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# Filesystem roots a ``load_env_module`` caller may load from. The helper is for
# trusted task-author code, so the allowlist matches the canonical task image's
# root-only data path. ``/tmp/output`` is deliberately excluded: that is the
# agent's writable scratch, and loading code from there bypasses the no-sandbox
# premise (use ``load_submitted_policy`` for agent code instead).
_DEFAULT_TRUSTED_ROOTS: tuple[Path, ...] = (Path("/mcp_server"),)


def load_env_module(
    path: str | Path,
    *,
    factory_name: str | None = "make_env",
    module_name: str | None = None,
    trusted_roots: tuple[Path, ...] | None = None,
) -> ModuleType:
    """Import a trusted Python module from a filesystem path and return it.

    Args:
        path: Path to a single ``.py`` file (e.g. ``/mcp_server/data/env.py``)
            or a package ``__init__.py`` (e.g.
            ``/mcp_server/data/envs/__init__.py``). The package case sets
            ``submodule_search_locations`` to the parent dir so one level of
            relative imports (``from .grid import GridEnv``) resolves.
        factory_name: Name of a callable the module must define. Defaults to
            ``"make_env"`` (the env-task convention). Pass ``None`` to skip the
            check (helpers / metric modules).
        module_name: ``sys.modules`` key. When ``None``, derived from the path
            stem plus a hash of the parent dir so concurrent loads of two
            ``env.py`` files under different task dirs do not alias.
        trusted_roots: Allowed parent directories for ``path``; paths outside
            them raise ``ValueError`` (prevents loading agent code from
            ``/tmp/output/``). Defaults to ``/mcp_server``.

    Returns:
        The loaded module; pull attributes directly (``module.make_env(seed=0)``).

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: ``path`` is outside ``trusted_roots``, or ``factory_name``
            was given and the module does not define a callable of that name.
        RuntimeError: importlib could not build a spec for the path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"env module not found at {path}")

    # resolve() collapses ../symlinks so /mcp_server/../tmp/output/policy.py
    # cannot masquerade as living under /mcp_server.
    resolved = path.resolve()
    roots = trusted_roots if trusted_roots is not None else _DEFAULT_TRUSTED_ROOTS
    if not any(resolved.is_relative_to(root.resolve()) for root in roots):
        root_list = ", ".join(str(r) for r in roots)
        raise ValueError(
            f"load_env_module refuses to load {resolved} -- path is outside the "
            f"trusted-roots allowlist [{root_list}]. This helper is for "
            "task-author code under /mcp_server/data/, not agent code under "
            "/tmp/output/. For agent code use "
            "grading.helpers.load_submitted_policy (sandboxed worker). To extend "
            "the allowlist for a legitimate non-/mcp_server load, pass "
            "trusted_roots=(...)."
        )

    if module_name is None:
        import hashlib

        parent_hash = hashlib.sha1(
            str(resolved.parent).encode("utf-8")
        ).hexdigest()[:8]
        module_name = f"task_env__{resolved.stem}__{parent_hash}"

    submodule_locations = [str(path.parent)] if path.name == "__init__.py" else None
    spec = importlib.util.spec_from_file_location(
        module_name, path, submodule_search_locations=submodule_locations
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {path}")

    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the module's own internal references (decorators
    # inspecting __module__, package-relative imports) resolve. Pop on failure
    # so a broken module does not leave a half-initialized stub behind.
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise

    if factory_name is not None:
        factory = getattr(mod, factory_name, None)
        if not callable(factory):
            kind = type(factory).__name__ if factory is not None else "missing"
            raise ValueError(
                f"{path} must define a callable named {factory_name!r}; got {kind}"
            )

    return mod
