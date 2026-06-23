"""Deterministic predicate helpers for `RubricBuilder.criterion` bodies.

Each helper is a plain Python function that returns a `bool` or `float` in
``[0, 1]``. Authors call them inside their own criteria — there is no
"grading_type" indirection. If the helper you need doesn't exist, just write
plain Python; these are convenience, not contract.

Example::

    from grading import RubricBuilder, helpers

    @rb.criterion(id="answer_present", weight=1.0)
    def _():
        return helpers.file_exists(workspace / "answer.txt", non_empty=True)

    @rb.criterion(id="contains_signoff", weight=0.5)
    def _():
        return helpers.file_contains(workspace / "answer.txt", "Signed-off-by:")

    @rb.criterion(id="rmse_close", weight=2.0)
    def _():
        return helpers.abs_error(actual=measured, expected=0.0, tolerance=0.05)
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from grading.faults import AgentFault

_DEFAULT_MAX_SUBMISSION_BYTES = 64 * 1024 * 1024
_DEFAULT_MAX_EXECUTABLE_OUTPUT_BYTES = 16 * 1024 * 1024
_DEFAULT_MAX_HDF5_BYTES = 2 * 1024 * 1024 * 1024
# Execution timeout for the privilege-dropped HDF5/h5ad reader worker. Bounds a
# crafted file that tries to hang or OOM the grader during the libhdf5 parse;
# the read itself is one RPC call, so this is the per-parse budget.
_H5_READ_TIMEOUT_S = 120.0
_RUBRIC_SCORE_MARKER = "RUBRIC_SCORE="


def file_exists(path: str | Path, *, non_empty: bool = False) -> bool:
    """True iff ``path`` exists. Pass ``non_empty=True`` to also require content."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return False
    if non_empty:
        try:
            return bool(p.read_bytes().strip())
        except OSError:
            return False
    return True


def file_contains(
    path: str | Path,
    needle: str,
    *,
    case_sensitive: bool = True,
    encoding: str = "utf-8",
) -> bool:
    """True iff the file at ``path`` contains ``needle``."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return False
    try:
        text = p.read_text(encoding=encoding, errors="replace")
    except OSError:
        return False
    if case_sensitive:
        return needle in text
    return needle.lower() in text.lower()


def regex_search(
    text: str,
    pattern: str,
    *,
    flags: int = 0,
) -> bool:
    """True iff ``pattern`` matches anywhere in ``text``. Returns False on bad regex."""
    try:
        return re.search(pattern, text, flags) is not None
    except re.error:
        return False


def exact_match(
    actual: Any,
    expected: Any,
    *,
    value_type: str = "string",
) -> float:
    """Return 1.0 iff ``actual`` matches ``expected`` under the given type.

    ``value_type``: ``"string"`` (default), ``"number"``, ``"boolean"``,
    ``"list"`` (ordered), ``"set"`` (unordered).
    """
    if value_type == "number":
        try:
            return 1.0 if float(actual) == float(expected) else 0.0
        except (TypeError, ValueError):
            return 0.0
    if value_type == "boolean":
        return (
            1.0 if str(actual).strip().lower() == str(expected).strip().lower() else 0.0
        )
    if value_type in ("list", "set"):
        actual_items = (
            list(actual)
            if isinstance(actual, (list, tuple))
            else [x.strip() for x in str(actual).split(",") if x.strip()]
        )
        expected_items = (
            list(expected)
            if isinstance(expected, (list, tuple))
            else [x.strip() for x in str(expected).split(",") if x.strip()]
        )
        if value_type == "set":
            return 1.0 if set(actual_items) == set(expected_items) else 0.0
        return 1.0 if actual_items == expected_items else 0.0
    return 1.0 if str(actual).strip() == str(expected).strip() else 0.0


def abs_error(
    actual: float,
    expected: float,
    *,
    tolerance: float,
) -> float:
    """Tolerance-anchored score: ``max(0, 1 - |actual-expected|/tolerance)``.

    Returns ``1.0`` when ``actual == expected``. Returns ``0.0`` when the
    error meets or exceeds ``tolerance``. Useful for "regression target"
    style criteria.
    """
    if tolerance == 0:
        return 1.0 if actual == expected else 0.0
    return max(0.0, 1.0 - abs(float(actual) - float(expected)) / abs(float(tolerance)))


def kendall_tau(
    actual: list[Any],
    expected: list[Any],
) -> float:
    """Kendall's tau ranking similarity, mapped to ``[0, 1]``.

    ``1.0`` = identical ordering, ``0.5`` = random, ``0.0`` = perfectly reversed.
    """
    if not expected or not actual:
        return 0.0
    n = min(len(expected), len(actual))
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if i >= len(actual) or j >= len(actual):
                continue
            try:
                ei = expected.index(actual[i])
                ej = expected.index(actual[j])
            except ValueError:
                discordant += 1
                continue
            sign = (i - j) * (ei - ej)
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    tau = (concordant - discordant) / total
    return max(0.0, (tau + 1) / 2)


def jaccard(actual: Iterable[Any], expected: Iterable[Any]) -> float:
    """Jaccard set similarity: ``|A & E| / |A | E|``. Empty/empty = 1.0."""
    a = set(actual)
    e = set(expected)
    if not a and not e:
        return 1.0
    union = a | e
    if not union:
        return 0.0
    return len(a & e) / len(union)


def json_path(data: Any, path: str) -> Any:
    """JSONPath-like extraction. Returns ``None`` on miss.

    Supported syntax:
      * ``$.key.0.nested`` (dot-numeric for list indices)
      * ``$.items[0].name``
      * ``$['keys.with.dots']["display name"]``
      * ``$.items[?symbol=TRX].amount`` (filter by field equality)
    """
    if not path:
        return data
    parts = _parse_json_path(path)
    if parts is None:
        return None
    current: Any = data
    for part in parts:
        if isinstance(part, tuple):
            _, field, expected = part
            if not isinstance(current, (list, tuple)):
                return None
            match = None
            for item in current:
                value = (
                    item.get(field)
                    if isinstance(item, dict)
                    else getattr(item, field, None)
                )
                if value is not None and str(value) == expected:
                    match = item
                    break
            current = match
            if current is None:
                return None
            continue

        if not part:
            continue
        try:
            idx = int(part)
            if isinstance(current, (list, tuple)):
                try:
                    current = current[idx]
                except IndexError:
                    return None
                continue
        except ValueError:
            pass
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current


def transcript_contains(
    transcript: str | None,
    needle: str,
    *,
    case_sensitive: bool = False,
) -> bool:
    """True iff ``needle`` appears in the transcript text."""
    if not transcript:
        return False
    if case_sensitive:
        return needle in transcript
    return needle.lower() in transcript.lower()


def load_json(path: str | Path) -> Any:
    """Load a JSON file, returning ``None`` on missing/invalid."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_submission_or_fault(
    path: os.PathLike | str,
    *,
    required_columns: Iterable[str] | None = None,
    n_rows: int | None = None,
    numeric_columns: Iterable[str] | None = None,
    unique_key_column: str | None = None,
    max_bytes: int = _DEFAULT_MAX_SUBMISSION_BYTES,
    read_csv_kwargs: dict[str, Any] | None = None,
    allow_extra_columns: bool = False,
):
    """Read an agent CSV submission, raising ``AgentFault`` for bad outputs.

    This closes common free-veto paths: symlinks to hidden data, FIFOs that
    hang grading, oversized files that OOM the verifier, malformed CSV, missing
    columns, non-finite values, duplicate merge keys, and unexpected columns.
    """
    import pandas as pd

    p = Path(path)
    try:
        st = os.lstat(p)
    except FileNotFoundError as exc:
        raise AgentFault(f"submission not found at {p}") from exc
    except OSError as exc:
        raise AgentFault(f"submission at {p} could not be stat'd: {exc}") from exc

    if not stat.S_ISREG(st.st_mode):
        raise AgentFault(
            f"submission at {p} is not a regular file "
            f"(mode={stat.filemode(st.st_mode)}); expected a plain CSV file"
        )
    if st.st_size == 0:
        raise AgentFault(f"submission at {p} is empty")
    if st.st_size > max_bytes:
        raise AgentFault(
            f"submission at {p} is {st.st_size} bytes, over the "
            f"{max_bytes}-byte limit"
        )

    try:
        df = pd.read_csv(p, **(read_csv_kwargs or {}))
    except Exception as exc:
        raise AgentFault(
            f"submission at {p} is not a readable CSV: {type(exc).__name__}: {exc}"
        ) from exc

    required = list(required_columns) if required_columns is not None else None
    numeric = list(numeric_columns) if numeric_columns is not None else None

    if required is not None:
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise AgentFault(f"submission missing required columns: {missing}")

    if n_rows is not None and len(df) != n_rows:
        raise AgentFault(f"submission has {len(df)} rows, expected {n_rows}")

    if numeric is not None:
        absent = [column for column in numeric if column not in df.columns]
        if absent:
            raise AgentFault(f"submission missing numeric columns: {absent}")
        try:
            values = df[numeric].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise AgentFault(f"non-numeric values in columns {numeric}: {exc}") from exc
        if not np.isfinite(values).all():
            raise AgentFault(f"non-finite (NaN/Inf) values in columns {numeric}")

    if unique_key_column is not None:
        if unique_key_column not in df.columns:
            raise AgentFault(f"submission missing key column {unique_key_column!r}")
        duplicate_count = int(df[unique_key_column].duplicated().sum())
        if duplicate_count:
            raise AgentFault(
                f"submission has {duplicate_count} duplicate value(s) in key "
                f"column {unique_key_column!r}; expected one row per key"
            )

    if not allow_extra_columns and (required is not None or numeric is not None):
        expected: set[str] = set()
        if required is not None:
            expected.update(required)
        if numeric is not None:
            expected.update(numeric)
        if unique_key_column is not None:
            expected.add(unique_key_column)
        extra = [column for column in df.columns if column not in expected]
        if extra:
            raise AgentFault(
                f"submission has unexpected column(s) {extra}; expected "
                f"exactly {sorted(expected)}. Pass allow_extra_columns=True "
                "if extra columns are intended."
            )

    return df


_POLICY_FILENAMES = ("policy.py", "submission.py", "agent.py")


def _resolve_policy_path(policy: str | Path) -> Path:
    """Resolve ``policy`` (a file or a dir/workspace) to a concrete policy file.

    Accepts the submitted policy file directly, or a workspace/output directory
    in which a conventional policy file (``policy.py`` etc.) is located.
    """
    p = Path(policy)
    if p.is_file():
        return p
    if p.is_dir():
        for name in _POLICY_FILENAMES:
            candidate = p / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"no submitted policy found at {policy}")


def run_policy(
    policy: str | Path,
    *,
    timeout_s: float = 5.0,
    first_call_timeout_s: float | None = None,
    cwd: str | Path | None = None,
):
    """Return a hardened worker for a submitted policy — the safe way to run agent code.

    This is the single approved entry point for executing submitted Python in a
    grader. It wraps :class:`grading.policy_runner.PolicyWorker` so every task
    gets the same sandbox contract for free:

    * the policy runs in a **non-root** (``agent``) subprocess when the grader
      is root, so it cannot read hidden ``private`` data or overwrite the grader;
    * its return value travels over a dedicated proto file descriptor — the
      score is **never** parsed from the policy's stdout (defeats stdout forging);
    * the agent-writable workspace is kept off the grader's import path;
    * the first call gets a generous timeout to absorb worker/MuJoCo startup.

    Use it as a context manager and read the score from the **return value**::

        from grading import helpers

        with helpers.run_policy(workspace) as policy:
            action = policy.act(public_obs)        # or policy.call("plan", ...)
        score = evaluate(action)   # never trust policy stdout for the score

    ``policy`` may be the policy file itself or a directory to search.
    """
    from .policy_runner import PolicyWorker

    return PolicyWorker(
        _resolve_policy_path(policy),
        timeout_s=timeout_s,
        first_call_timeout_s=first_call_timeout_s,
        cwd=Path(cwd) if cwd is not None else None,
    )


def run_model_module(
    policy: str | Path,
    method: str,
    *args: Any,
    timeout_s: float = 60.0,
    first_call_timeout_s: float | None = None,
    cwd: str | Path | None = None,
    **kwargs: Any,
) -> Any:
    """Call one function of a submitted module in the sandbox, and return it.

    Convenience over :func:`run_policy` for the common structural/numerical
    pattern where the model submits a module exposing a single callable — e.g.
    ``evaluate_section(case_path)`` or ``compute_rayleigh_damping(omegas, ...)``
    — and the grader calls it once and scores the returned values. This is the
    **only approved way** for a grader to execute that submitted function:
    never ``import``/``exec_module`` the model's file in the grading process
    (which runs as root). The call runs in a non-root subprocess; the result
    travels over a dedicated proto file descriptor, never the policy's stdout.

    Inputs and the return value cross a JSON boundary, so pass JSON-serializable
    arguments (e.g. a path string, lists of numbers) and return JSON-serializable
    results (dicts/lists/scalars). Example::

        from grading import helpers

        result = helpers.run_model_module(
            "/tmp/output/fixed_fiber_section_processor.py",
            "evaluate_section",
            str(public_case_path),
        )
        score = compare_to_reference(result)   # never trust the module's stdout
    """
    with run_policy(
        policy,
        timeout_s=timeout_s,
        first_call_timeout_s=first_call_timeout_s,
        cwd=cwd,
    ) as worker:
        return worker.call(method, *args, **kwargs)


def load_submitted_policy(*args: Any, **kwargs: Any) -> Any:
    """Load a submitted policy in a sandboxed worker, returning proxy handle(s).

    ML_Envs-style loader (named ``load_policy`` factory by default, source mode,
    and multi-policy dict / list factories). Thin re-export of
    :func:`grading.policy_runner.load_submitted_policy`; see that docstring for
    the full contract. Raises :class:`grading.faults.AgentFault` for agent-caused
    load failures (missing / non-regular / oversized policy, or a load-time
    raise).
    """
    from .policy_runner import load_submitted_policy as _impl

    return _impl(*args, **kwargs)


def world_integrity(
    model: Any,
    *,
    expect_gravity: tuple[float, float, float] | None = (0.0, 0.0, -9.81),
    gravity_tol: float = 0.10,
    forbid_gravcomp: bool = True,
    forbid_equality: bool = True,
    require_contacts: bool = True,
) -> tuple[bool, list[str]]:
    """Validate a submitted MJCF preserves the intended physics. ``(ok, violations)``.

    Agents frequently "win" a MuJoCo task by quietly rigging the world rather
    than controlling it. This catches the recurring MJCF reward-hacks so a
    scorer can zero (or gate) a submission whose physics was tampered with:

    * **gravity** disabled or tilted away from ``expect_gravity`` (pass
      ``expect_gravity=None`` to skip, eg. space tasks);
    * **gravcomp** > 0 on any body (per-body gravity compensation = free
      floating);
    * **<equality>** constraints (weld/connect/joint) that can slave a
      "passive" body to an actuated one;
    * **contacts** globally disabled, or every geom opted out of collision.

    Each check is opt-out via its keyword so legitimately exotic tasks can relax
    it. ``model`` is a ``mujoco.MjModel``. Returns ``(True, [])`` when clean.

    Example::

        ok, why = helpers.world_integrity(model, expect_gravity=(0, 0, -9.81))
        if not ok:
            rb.metadata["world_violations"] = why
            return 0.0  # rigged world ⇒ no credit
    """
    import mujoco  # lazy: only graders that call this need the dep
    import numpy as np

    violations: list[str] = []

    if expect_gravity is not None:
        gravity = np.asarray(model.opt.gravity, dtype=float)
        target = np.asarray(expect_gravity, dtype=float)
        if float(np.linalg.norm(gravity - target)) > gravity_tol:
            violations.append(
                f"gravity {gravity.tolist()} deviates from expected "
                f"{target.tolist()} (tol {gravity_tol})"
            )

    if forbid_gravcomp:
        gravcomp = getattr(model, "body_gravcomp", None)
        if gravcomp is not None and float(np.max(np.abs(np.asarray(gravcomp)))) > 1e-9:
            violations.append("body gravcomp is non-zero (gravity compensation)")

    if forbid_equality and int(getattr(model, "neq", 0)) > 0:
        active = model.eq_active0 if hasattr(model, "eq_active0") else None
        n_active = (
            int(np.count_nonzero(np.asarray(active)))
            if active is not None
            else int(model.neq)
        )
        if n_active > 0:
            violations.append(
                f"{n_active} <equality> constraint(s) present (can slave passive bodies)"
            )

    if require_contacts:
        disabled = int(model.opt.disableflags) & int(
            mujoco.mjtDisableBit.mjDSBL_CONTACT
        )
        if disabled:
            violations.append("contacts globally disabled via opt.disableflags")
        elif model.ngeom > 0:
            contype = np.asarray(model.geom_contype)
            conaffinity = np.asarray(model.geom_conaffinity)
            if not contype.any() and not conaffinity.any():
                violations.append(
                    "every geom opts out of collision (contype/conaffinity all 0)"
                )

    return (not violations), violations


# ── Submission loaders: executable + HDF5 (per-type attack closures) ──────


@dataclass(frozen=True)
class ExecutableResult:
    """Result of running a submitted executable in the sandbox."""

    returncode: int
    stdout: str


def _kill_quietly(proc: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(OSError):
        proc.kill()


def run_submitted_executable(
    path: str | Path,
    *,
    args: Iterable[str] = (),
    input_text: str | None = None,
    timeout_s: float = 30.0,
    cwd: str | Path | None = None,
    max_bytes: int = _DEFAULT_MAX_SUBMISSION_BYTES,
    max_output_bytes: int = _DEFAULT_MAX_EXECUTABLE_OUTPUT_BYTES,
) -> ExecutableResult:
    """Run a submitted executable in a non-root sandbox; return its scrubbed
    stdout + return code.

    Closes the executable-submission attack surface:

    * runs as the unprivileged ``agent`` account when the grader is root, so the
      binary cannot read the held-out truth under ``/mcp_server``;
    * strips every line containing ``RUBRIC_SCORE=`` from the returned stdout, so
      the executable cannot forge a grade across the stdout boundary;
    * discards stderr to ``/dev/null`` (the result only exposes stdout) so the
      binary cannot exhaust grader memory by flooding an uncapped stderr pipe;
    * streams stdout through a capped reader that kills the child the moment its
      output passes ``max_output_bytes``, so the full stdout is never buffered in
      the grader (a true memory bound, not a post-capture check);
    * ``lstat`` + regular-file + size + executable-bit guards and a timeout, all
      raising :class:`AgentFault` for agent-controlled failures.

    Do NOT parse the score from the returned stdout for trust-sensitive logic;
    compute the score yourself from the executable's declared output artifacts.
    """
    from .policy_runner import _agent_drop_kwargs  # lazy: avoid import cycle

    p = Path(path)
    try:
        st = os.lstat(p)
    except OSError as exc:
        raise AgentFault(f"submitted executable at {p} is missing: {exc}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise AgentFault(
            f"submitted executable at {p} is not a regular file "
            f"(mode={stat.filemode(st.st_mode)})"
        )
    if st.st_size > max_bytes:
        raise AgentFault(
            f"submitted executable at {p} is {st.st_size} bytes, over the "
            f"{max_bytes}-byte limit"
        )
    if not os.access(p, os.X_OK):
        raise AgentFault(
            f"submitted executable at {p} is not marked executable (chmod +x)"
        )

    drop_kwargs = _agent_drop_kwargs()
    try:
        proc = subprocess.Popen(
            [str(p), *[str(arg) for arg in args]],
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            # stderr is never returned; discard it at the kernel so a submission
            # cannot flood an uncapped stderr pipe and exhaust grader memory.
            stderr=subprocess.DEVNULL,
            cwd=str(cwd) if cwd is not None else str(p.parent),
            **drop_kwargs,
        )
    except OSError as exc:
        raise AgentFault(
            f"submitted executable {p.name} failed to start: {exc}"
        ) from exc

    # Stream stdout with a hard cap: the reader stops and kills the child once
    # output crosses the cap, so at most ~one chunk past the limit is ever held
    # in memory (a true bound, not a post-capture check). stderr -> /dev/null.
    captured = bytearray()
    cap_limit = max_output_bytes + 1
    over_cap = threading.Event()

    def _drain_stdout() -> None:
        assert proc.stdout is not None
        try:
            while True:
                chunk = proc.stdout.read1(65536)
                if not chunk:
                    break
                if len(captured) < cap_limit:
                    captured.extend(chunk)
                if len(captured) > max_output_bytes:
                    over_cap.set()
                    _kill_quietly(proc)
                    break
        except (OSError, ValueError):
            pass

    def _feed_stdin() -> None:
        if input_text is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(input_text.encode("utf-8"))
        except (OSError, ValueError):
            pass
        finally:
            with contextlib.suppress(OSError):
                proc.stdin.close()

    reader = threading.Thread(target=_drain_stdout, daemon=True)
    reader.start()
    writer = threading.Thread(target=_feed_stdin, daemon=True)
    writer.start()

    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        _kill_quietly(proc)
        raise AgentFault(
            f"submitted executable {p.name} timed out after {timeout_s:.1f}s"
        ) from exc
    finally:
        reader.join(timeout=5.0)
        writer.join(timeout=5.0)
        with contextlib.suppress(OSError):
            if proc.stdout is not None:
                proc.stdout.close()

    if over_cap.is_set():
        raise AgentFault(
            f"submitted executable {p.name} produced more than "
            f"{max_output_bytes} bytes of stdout"
        )

    stdout = bytes(captured).decode("utf-8", "replace")
    scrubbed = "\n".join(
        line for line in stdout.splitlines() if _RUBRIC_SCORE_MARKER not in line
    )
    return ExecutableResult(returncode=proc.returncode, stdout=scrubbed)


# LITERAL source wrapper executed INSIDE the privilege-dropped policy worker
# (grading.policy_runner.load_submitted_policy, source mode). The libhdf5 parse
# of the agent-controlled file therefore runs as the unprivileged agent uid --
# never in the (root) grader process -- so a crafted ``.h5`` that trips a
# libhdf5 memory-safety bug cannot execute with root + held-out-truth read
# access. Requested dataset names + the output dir arrive as msgpack CALL
# ARGUMENTS to ``read`` (never interpolated into this source, per the
# policy-loader security note). Each dataset is written to ``out_dir`` as a
# non-object ``.npy`` (the parent loads it with ``allow_pickle=False``, so no
# pickle ever executes on the cross-process transfer); only the small file
# manifest returns over the msgpack wire, so a multi-hundred-MiB read is not
# bounded by the policy protocol's 1 GiB per-frame limit. Every logical guard
# runs here, before any dataset is materialized.
_H5_READER_WRAPPER = '''
def load_reader():
    import os
    import h5py
    import numpy as np

    src_path = __file__

    class _Reader:
        def read(self, names, out_dir):
            try:
                written = []
                with h5py.File(src_path, "r", locking=False) as f:
                    if names is None:
                        names = list(f.keys())
                    for idx, name in enumerate(names):
                        if "/" in name.strip("/"):
                            return {"ok": False, "reason": "dataset %r must be top-level (no nested groups)" % (name,)}
                        link = f.get(name, getlink=True)
                        if link is None:
                            return {"ok": False, "reason": "dataset %r not found" % (name,)}
                        if not isinstance(link, h5py.HardLink):
                            return {"ok": False, "reason": "dataset %r must be a direct hard link, got %s (external/soft links are rejected)" % (name, type(link).__name__)}
                        obj = f[name]
                        if not isinstance(obj, h5py.Dataset):
                            return {"ok": False, "reason": "%r is not a dataset" % (name,)}
                        if getattr(obj, "is_virtual", False):
                            return {"ok": False, "reason": "dataset %r is a virtual dataset (rejected)" % (name,)}
                        if obj.file.filename != f.filename:
                            return {"ok": False, "reason": "dataset %r resolves outside the submission file" % (name,)}
                        if obj.dtype.hasobject:
                            return {"ok": False, "reason": "dataset %r has object/vlen dtype; use the .h5ad loader" % (name,)}
                        fname = "%d.npy" % idx
                        # allow_pickle=False refuses any object array (already
                        # rejected above); the parent also loads with
                        # allow_pickle=False, so the transfer never executes a
                        # pickle.
                        np.save(os.path.join(out_dir, fname), np.asarray(obj[()]), allow_pickle=False)
                        written.append({"name": name, "file": fname})
                return {"ok": True, "written": written}
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return {"ok": False, "reason": "%s: %s" % (type(exc).__name__, exc)}

    return _Reader()
'''


def load_submission_h5_or_fault(
    path: str | Path,
    *,
    datasets: Iterable[str] | None = None,
    max_bytes: int = _DEFAULT_MAX_HDF5_BYTES,
) -> dict[str, np.ndarray]:
    """Safely read top-level datasets from a submitted HDF5 file.

    Closes the HDF5-submission attack surface (a crafted ``.h5`` that maps a
    grader-private file into a "dataset", or that trips a libhdf5 memory-safety
    bug while parsing):

    * the libhdf5 parse runs in a **privilege-dropped worker** (the same
      uid-1000 sandbox as :func:`run_submitted_executable` /
      :func:`load_submitted_policy`); ``h5py.File`` is never opened in this
      (root) grader process, so a parser crash cannot execute with root +
      ``/mcp_server`` truth-read access;
    * rejects external links, soft links, virtual datasets (VDS), object/vlen
      dtypes, and datasets that resolve to a different file -- any of which
      could pull ``/mcp_server`` truth into the agent's submission;
    * only top-level datasets are read, so traversal never dereferences a link;
    * ``lstat`` + regular-file + size guards, plus a worker timeout that bounds
      a file crafted to hang or OOM the grader.

    The worker writes each dataset to a temp ``.npy`` and the parent loads it
    with ``allow_pickle=False``; only the file manifest crosses the RPC wire,
    so a read up to ``max_bytes`` is not capped by the policy protocol's 1 GiB
    per-frame limit and no pickle executes on the transfer.

    Agent-controlled failures (missing / non-regular / over-sized / malformed /
    unsafe-link / crashing file) raise :class:`AgentFault` (kept 0.0); a genuine
    infra fault (missing drop identity, un-spawnable worker, h5py absent)
    propagates (discarded). Returns ``{name: ndarray}`` for the requested
    top-level datasets (or every top-level dataset when ``datasets`` is
    ``None``). The public signature is unchanged from the in-process reader.
    """
    from .policy_runner import PolicyWorkerError, load_submitted_policy

    # h5py is imported (NOT used to parse) here only to fail closed with an
    # infra error when the image lacks it; the actual libhdf5 parse happens in
    # the dropped worker below, never in this (possibly root) process.
    try:
        import h5py  # noqa: F401
    except Exception as exc:  # pragma: no cover - the task image ships h5py
        raise RuntimeError(
            "h5py is required to read HDF5 submissions but is not installed"
        ) from exc

    p = Path(path)
    try:
        st = os.lstat(p)
    except OSError as exc:
        raise AgentFault(f"submitted HDF5 at {p} is missing: {exc}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise AgentFault(f"submitted HDF5 at {p} is not a regular file")
    if st.st_size > max_bytes:
        raise AgentFault(
            f"submitted HDF5 at {p} is {st.st_size} bytes, over the "
            f"{max_bytes}-byte limit"
        )

    names = list(datasets) if datasets is not None else None

    # The worker writes each dataset as a ``.npy`` here and returns only the
    # file manifest, so an arbitrarily large read is not bounded by the policy
    # protocol's 1 GiB per-frame wire limit. The dir must be writable by the
    # dropped (uid-1000) worker when the grader is root.
    out_dir = tempfile.mkdtemp(prefix="h5-submission-read-")
    try:
        if os.geteuid() == 0:
            os.chmod(out_dir, 0o777)

        # A missing drop identity / un-spawnable worker raises PolicyWorkerError
        # from load_submitted_policy (infra, discarded). The read RPC is wrapped
        # separately so a crash/timeout while parsing the agent file is
        # attributed to the agent.
        reader = load_submitted_policy(
            source=_H5_READER_WRAPPER,
            factory_name="load_reader",
            path=p,
            sys_path_dirs=[],
            timeout_s=_H5_READ_TIMEOUT_S,
        )
        try:
            result = reader.read(names, out_dir)
        except TimeoutError as exc:
            raise AgentFault(
                f"submitted HDF5 at {p} timed out after {_H5_READ_TIMEOUT_S:.1f}s"
            ) from exc
        except PolicyWorkerError as exc:
            # The worker died parsing the agent file (e.g. a libhdf5 crash); its
            # only job is to parse the submission, so attribute it to the agent
            # (kept 0.0) rather than discarding.
            raise AgentFault(
                f"submitted HDF5 at {p} crashed the reader: {exc}"
            ) from exc
        finally:
            reader.close()

        if not isinstance(result, dict) or not result.get("ok"):
            reason = result.get("reason") if isinstance(result, dict) else result
            raise AgentFault(f"submitted HDF5 at {p} could not be read: {reason}")

        out: dict[str, np.ndarray] = {}
        for entry in result.get("written") or []:
            # allow_pickle=False: a compromised worker cannot smuggle a pickled
            # object array back across the boundary as code to execute.
            npy = os.path.join(out_dir, os.path.basename(str(entry["file"])))
            out[str(entry["name"])] = np.load(npy, allow_pickle=False)
        return out
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


# LITERAL source wrapper executed INSIDE the privilege-dropped policy worker
# (source mode). Reading whole-object / categorical / sparse AnnData needs the
# anndata library's structure handling rather than a dense-dataset read; doing
# it in the dropped worker keeps the parse off the root grader process. The
# in-worker ``read_h5ad`` + ``write_h5ad`` round-trip yields a self-contained
# copy (no external links / virtual datasets), and the uid-1000 worker cannot
# reach ``/mcp_server``, so a truth-pointing submission fails the read instead
# of baking truth into the clean copy.
_H5AD_SANITIZE_WRAPPER = '''
def load_reader():
    src_path = __file__

    class _Sanitizer:
        def sanitize(self, out_path):
            try:
                import anndata
            except ImportError as exc:
                return {"ok": False, "infra": True, "reason": "anndata is not installed: %s" % (exc,)}
            try:
                anndata.read_h5ad(src_path).write_h5ad(out_path)
                return {"ok": True}
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return {"ok": False, "reason": "%s: %s" % (type(exc).__name__, exc)}

    return _Sanitizer()
'''


def load_submission_h5ad_or_fault(
    path: str | Path,
    *,
    out_path: str | Path,
    max_bytes: int = _DEFAULT_MAX_HDF5_BYTES,
) -> Path:
    """Round-trip a submitted ``.h5ad`` (AnnData) through anndata IN the
    privilege-dropped worker into a self-contained clean copy at ``out_path``
    that the grader can then read normally.

    Use this (not :func:`load_submission_h5_or_fault`) for categorical / sparse
    / whole-object AnnData that the dense-dataset reader cannot represent. The
    in-worker ``read_h5ad`` + ``write_h5ad`` drops external links and virtual
    datasets, and the unprivileged worker cannot reach ``/mcp_server``, so a
    truth-pointing submission fails the read rather than baking truth in.

    Agent-controlled failures (missing / non-regular / over-sized / malformed /
    over-structured submission) raise :class:`AgentFault` (kept 0.0); a missing
    ``anndata`` or a dead/un-spawnable worker propagates as infra (discarded).
    ``out_path`` must live in a directory the (possibly dropped) worker can
    write. Returns ``out_path``.
    """
    from .policy_runner import PolicyWorkerError, load_submitted_policy

    p = Path(path)
    out = Path(out_path)
    try:
        st = os.lstat(p)
    except OSError as exc:
        raise AgentFault(f"submitted .h5ad at {p} is missing: {exc}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise AgentFault(f"submitted .h5ad at {p} is not a regular file")
    if st.st_size > max_bytes:
        raise AgentFault(
            f"submitted .h5ad at {p} is {st.st_size} bytes, over the "
            f"{max_bytes}-byte limit"
        )

    sanitizer = load_submitted_policy(
        source=_H5AD_SANITIZE_WRAPPER,
        factory_name="load_reader",
        path=p,
        sys_path_dirs=[],
        timeout_s=_H5_READ_TIMEOUT_S,
    )
    try:
        result = sanitizer.sanitize(str(out))
    except TimeoutError as exc:
        raise AgentFault(
            f"submitted .h5ad at {p} timed out after {_H5_READ_TIMEOUT_S:.1f}s"
        ) from exc
    except PolicyWorkerError as exc:
        raise AgentFault(
            f"submitted .h5ad at {p} crashed the sanitizer: {exc}"
        ) from exc
    finally:
        sanitizer.close()

    if not isinstance(result, dict) or not result.get("ok"):
        reason = result.get("reason") if isinstance(result, dict) else result
        if isinstance(result, dict) and result.get("infra"):
            raise RuntimeError(f"could not sanitize submitted .h5ad: {reason}")
        raise AgentFault(f"submitted .h5ad at {p} could not be sanitized: {reason}")
    return out


# ── Internal: JSON path parser ────────────────────────────────────


def _parse_filter_expr(expr: str) -> tuple[str, str, str] | None:
    op = "==" if "==" in expr else "="
    if op not in expr:
        return None
    field, value = expr.split(op, 1)
    field = field.strip()
    value = value.strip()
    if not field or not value:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return ("filter_eq", field, value)


def _parse_json_path(path: str):
    path = path.strip()
    if not path:
        return []

    parts: list = []
    i = 1 if path.startswith("$") else 0
    n = len(path)

    while i < n:
        char = path[i]
        if char == ".":
            i += 1
            start = i
            while i < n and path[i] not in ".[":
                i += 1
            if i > start:
                parts.append(path[start:i])
            continue

        if char == "[":
            i += 1
            while i < n and path[i].isspace():
                i += 1
            if i >= n:
                return None

            if path[i] == "?":
                i += 1
                start = i
                while i < n and path[i] != "]":
                    i += 1
                if i >= n:
                    return None
                filter_part = _parse_filter_expr(path[start:i])
                if filter_part is None:
                    return None
                i += 1
                parts.append(filter_part)
                continue

            if path[i] in ("'", '"'):
                quote = path[i]
                i += 1
                buf: list[str] = []
                while i < n:
                    ch = path[i]
                    if ch == "\\":
                        i += 1
                        if i >= n:
                            return None
                        buf.append(path[i])
                        i += 1
                        continue
                    if ch == quote:
                        i += 1
                        break
                    buf.append(ch)
                    i += 1
                else:
                    return None
                while i < n and path[i].isspace():
                    i += 1
                if i >= n or path[i] != "]":
                    return None
                i += 1
                parts.append("".join(buf))
                continue

            start = i
            while i < n and path[i] != "]":
                i += 1
            if i >= n:
                return None
            part = path[start:i].strip()
            i += 1
            if part:
                parts.append(part)
            continue

        start = i
        while i < n and path[i] not in ".[":
            i += 1
        if i > start:
            parts.append(path[start:i])

    return parts


__all__ = [
    "ExecutableResult",
    "abs_error",
    "exact_match",
    "file_contains",
    "file_exists",
    "jaccard",
    "json_path",
    "kendall_tau",
    "load_json",
    "load_submission_h5_or_fault",
    "load_submission_h5ad_or_fault",
    "load_submission_or_fault",
    "regex_search",
    "load_submitted_policy",
    "run_model_module",
    "run_policy",
    "run_submitted_executable",
    "transcript_contains",
    "world_integrity",
]
