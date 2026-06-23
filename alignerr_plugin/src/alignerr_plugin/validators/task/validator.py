"""Universal presence-driven task validator."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

from alignerr_plugin.ground_truth import (
    VIDEO_SUFFIXES,
    artifact_matches_proof,
    expected_ground_truth_score,
    failed_criteria,
    metadata_validation_issues,
    render_expected,
    validate_video_file,
)
from alignerr_plugin.local_runtime import ensure_local_base_image
from alignerr_plugin.proof import PROOF_PATH, verify_build_proof, write_build_proof
from alignerr_plugin.schemas import StageResult, ValidationResult
from alignerr_plugin.utils import load_metadata, load_task_toml, read_json, task_id

KNOWN_TOOLS = {"browser", "terminal", "terminal_persistent", "desktop"}

# Minimum length for a usable agent-facing prompt (ported from ML_Envs
# validate_task.validate_problem_quality). Anything shorter is almost never a
# real task description.
_MIN_PROMPT_LENGTH = 200

# Reliable tells of AI-spam / non-task content in human-authored prompt text.
# Ported from ML_Envs scripts/validate_task.py (AI_ARTIFACT_PATTERNS). The
# "certainly!" pattern drops the trailing \b that ML_Envs carries: `!` is a
# non-word char, so `\bcertainly!\b` can never match (no word boundary follows
# `!`). The trailing boundary is removed here so the check actually fires.
AI_ARTIFACT_PATTERNS = [
    re.compile(r"\bas an ai\b", re.IGNORECASE),
    re.compile(r"\bcertainly!", re.IGNORECASE),
]

# Unicode ranges covering the main emoji blocks. Ported from ML_Envs
# scripts/validate_task.py (EMOJI_PATTERN).
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa70-\U0001faff"
    "]+",
    flags=re.UNICODE,
)

# Human-authored files that must stay ASCII-only. These are the load-bearing
# files an agent reads/executes verbatim: the prompt and the grader. README* is
# scanned too when present (resolved at runtime). Generated artifacts, data
# files, and __pycache__ are deliberately excluded.
_ASCII_SCAN_FILES = ("instruction.md", "scorer/compute_score.py")

_PRIVATE_ROOTS = ("/mcp_server/data", "/mcp_server/grader")
_PUBLIC_ROOTS = ("/data", "/workdir", "/tmp/output", "/app", "/workspace")
# On-disk location of the private held-out truth in the task source tree. It is
# committed but baked root-only into the container as ``/mcp_server/data`` (see
# `_PRIVATE_ROOTS`). A reference solution must never read it under either name.
_PRIVATE_DISK_ROOT = "scorer/data"
_PRIVATE_DOCKER_SOURCES = (
    "scorer",
    "/mcp_server/data",
    "/mcp_server/grader",
    "/root/task-private",
)
_COPY_FLAGS_WITH_VALUE = {"chown", "chmod", "from", "exclude"}
_SENSITIVE_PRIVATE_NAME_PARTS = (
    "hidden",
    "secret",
    "secrets",
    "private",
    "truth",
    "truths",
    "expected",
    "answer",
    "answers",
    "oracle",
    "oracles",
    "anchor",
    "anchors",
)
_PRIVATE_LAYOUT_PROBE = r"""
import os
import pwd
import stat
import sys

PRIVATE_ROOTS = ("/mcp_server/data", "/mcp_server/grader")
PUBLIC_ROOTS = ("/data", "/workdir", "/tmp/output", "/app", "/workspace")


def fail(message):
    print(message, file=sys.stderr)
    raise SystemExit(1)


try:
    agent = pwd.getpwnam(os.environ.get("RUBRIC_AGENT_USER") or "agent")
except KeyError as exc:
    fail(f"missing unprivileged agent account: {exc}")

for root in PRIVATE_ROOTS:
    if not os.path.exists(root):
        continue
    root_stat = os.stat(root)
    if root_stat.st_uid != 0:
        fail(f"private root is not root-owned: {root}")
    for dirpath, dirnames, filenames in os.walk(root):
        for name in [".", *dirnames, *filenames]:
            path = dirpath if name == "." else os.path.join(dirpath, name)
            st = os.stat(path, follow_symlinks=False)
            mode = stat.S_IMODE(st.st_mode)
            if st.st_uid != 0:
                fail(f"private path is not root-owned: {path}")
            if stat.S_ISLNK(st.st_mode):
                continue
            if mode & 0o077:
                fail(f"private path has group/world permissions {mode:o}: {path}")

    pid = os.fork()
    if pid == 0:
        try:
            os.setgroups([])
            os.setgid(agent.pw_gid)
            os.setuid(agent.pw_uid)
            if os.access(root, os.R_OK | os.X_OK):
                os._exit(42)
            os._exit(0)
        except BaseException:
            os._exit(43)
    _pid, status = os.waitpid(pid, 0)
    if os.WEXITSTATUS(status) == 42:
        fail(f"agent can read/traverse private root: {root}")
    if os.WEXITSTATUS(status) != 0:
        fail(f"agent readability probe failed for {root}")

for public_root in PUBLIC_ROOTS:
    if not os.path.isdir(public_root):
        continue
    for dirpath, dirnames, filenames in os.walk(public_root, followlinks=False):
        for name in [*dirnames, *filenames]:
            path = os.path.join(dirpath, name)
            resolved = os.path.realpath(path)
            if any(
                resolved == private or resolved.startswith(private + os.sep)
                for private in PRIVATE_ROOTS
            ):
                fail(f"public path links to private root: {path} -> {resolved}")
"""

_AGENT_PYTHON_PROBE = (
    'command -v python && python -c "import sys; print(sys.executable)"'
)

# Attribute / function names that load and execute Python from an arbitrary file
# path. A grader must NEVER use these on the model's deliverables: the grading
# process runs as root, so importing or exec'ing agent-authored code lets a
# submission monkeypatch the grader, read hidden fixtures, or forge its score.
# The single approved way to run submitted Python is grading.helpers.run_policy,
# which executes it in a non-root sandbox subprocess.
# Distinctive enough to flag whether referenced as a bare name or an attribute.
_DYNAMIC_CODE_EXEC_MARKERS = (
    "spec_from_file_location",
    "exec_module",
    "SourceFileLoader",
)
# Generic words that are only suspicious as attribute access (e.g. runpy.run_path,
# imp.load_source), so we do not flag same-named local variables.
_DYNAMIC_CODE_EXEC_ATTR_ONLY = (
    "run_path",
    "load_source",
)
_AGENT_WRITABLE_HINTS = (
    "/tmp/output",
    "/workdir",
    "/workspace",
    "/app",
)
_SANDBOX_HELPER_HINT = (
    "Run submitted Python through grading.helpers.run_policy(...) "
    "(a non-root sandbox subprocess) and read the score from its return value; "
    "never import, exec, or eval the model's deliverable in the grader process."
)

# --- AgentFault keep-vs-discard lint -------------------------------------------
# The grader runs as root and reads the held-out truth. Two failure modes poison
# RL training data:
#   * over-keep: a broad `except` that returns a score swallows an author/infra
#     bug (missing truth, broken import) into a kept 0.0, training the agent on
#     garbage and teaching it that crashing the grader yields a defined score;
#   * over-discard: an UNguarded read of an agent-writable path lets a planted
#     directory/FIFO raise OSError out of compute_score, so the runtime flags
#     env_internal_failure and DISCARDS the 0.0 the agent actually earned.
# The fix in both cases is the same discipline: catch the specific
# agent-controlled exception and `raise grading.faults.AgentFault`, and let
# author/infra faults propagate.
_AGENT_FAULT_NAME = "AgentFault"
_BROAD_EXC_NAMES = {"Exception", "BaseException"}
# Handlers that absorb the directory/permission (OSError) discard veto.
_DIR_VETO_GUARD_EXC = {
    "OSError",
    "IOError",
    "EnvironmentError",
    "FileNotFoundError",
    "IsADirectoryError",
    "PermissionError",
    "ValueError",
    "Exception",
    "BaseException",
}
# Modules whose ``.load`` / ``.loads`` execute pickle (arbitrary code) at load.
_PICKLE_MODULE_NAMES = {"pickle", "cloudpickle", "dill", "joblib", "torch"}
# Bare-name readers a planted directory/FIFO/oversize file can weaponize.
_GENERIC_AGENT_READER_NAMES = {"open"}
# Attribute-form readers where the path is the first argument
# (df.read_csv(path), np.loadtxt(path), ...).
_GENERIC_AGENT_READER_ATTRS = {
    "read_csv",
    "read_parquet",
    "read_json",
    "read_table",
    "read_excel",
    "read_hdf",
    "read_feather",
    "loadtxt",
    "genfromtxt",
}
# Path methods where the *receiver* is the path being read
# (``(workspace / "f").read_text()``, ``p.read_bytes()``). A planted
# directory/FIFO at that path raises OSError just like the arg-form readers.
_AGENT_PATH_METHOD_READERS = {"read_text", "read_bytes"}
# Tokens that mark a path as living in the agent-writable submission area.
_AGENT_PATH_TOKENS = ("workspace", "/tmp/output", "/workdir", "/workspace", "/app")
# Names of helpers that return a zero/failure grade (so `return _failure(...)`
# inside a broad except is the over-keep pattern, not an intermediate sentinel).
_FAILURE_FUNC_RE = re.compile(r"fail|zero|no_credit|nocredit", re.IGNORECASE)
_SUBMISSION_PATH_KWARGS = {"path", "filepath_or_buffer", "filepath", "file", "fname"}
_AGENT_FAULT_HELPER_HINT = (
    "Read agent submissions via grading.helpers.load_submission_or_fault(...) "
    "(CSV) or the sandboxed run_model_module / run_policy / run_submitted_executable "
    "helpers, which raise AgentFault for you; or guard the read with "
    "`except OSError: raise grading.faults.AgentFault(...)`."
)

# Heuristic reward-hacking lint markers (advisory only). These flag patterns
# that frequently make criteria gameable; they are warnings, not hard failures.
_SENTIMENT_POSITIVE_WORDS = (
    "viable",
    "safe",
    "acceptable",
    "proceed",
    "usable",
    "approved",
    "passes",
)
_SENTIMENT_NEGATIVE_WORDS = (
    "unsafe",
    "unusable",
    "unacceptable",
    "not acceptable",
    "reject",
    "cannot be accepted",
    "fails",
)


class TaskValidator:
    """Validate generic Boreal/Harbor task submissions without task classification."""

    def load_submission(self, problem_dir: Path) -> dict:
        """Load task metadata for the Alignerr CLI."""
        return load_metadata(problem_dir).model_dump()

    def get_problem_id(self, problem_data: dict) -> str:
        """Extract the problem id from metadata."""
        instance_id = problem_data.get("instance_id") or problem_data.get("id")
        if not isinstance(instance_id, str):
            raise ValueError("problem_data must include instance_id")
        return instance_id

    def validate(
        self, problem_dir: Path, results_dir: Path, _project_root: Path
    ) -> ValidationResult:
        """Run mandatory and presence-driven validation stages."""
        problem_id = task_id(problem_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        return_stage, return_meta = self._compute_score_return(problem_dir)

        stages = {
            "schema": self._schema(problem_dir),
            "prompt_runtime_references": self._prompt_runtime_references(problem_dir),
            "prompt_quality": self._prompt_quality(problem_dir),
            "grader_import": self._grader_import(problem_dir),
            "grader_sandbox": self._grader_sandbox(problem_dir),
            "agent_fault": self._agent_fault(problem_dir),
            "scorer_determinism": self._scorer_determinism(problem_dir),
            "outputs": self._outputs(problem_dir),
            "ground_truth": self._ground_truth(problem_dir),
            "private_data_layout": self._private_data_layout(problem_dir),
            "solution_answer_key_leak": self._solution_answer_key_leak(problem_dir),
            "hidden_env": self._hidden_env(problem_dir),
            "compute_score_return": return_stage,
            "local_build_proof": self._local_build_proof(problem_dir),
            "conditional": self._conditional(problem_dir),
        }

        status = (
            "valid" if all(stage.passed for stage in stages.values()) else "invalid"
        )
        result = ValidationResult(
            problem_id=problem_id,
            benchmark="taiga_task",
            status=status,
            stages=stages,
            metadata={
                "instance_id": problem_id,
                "return_shape": return_meta.get("return_shape", "unknown"),
                "uses_llm_judge": return_meta.get("uses_llm_judge", False),
                "sample_score": return_meta.get("sample_score"),
                "ground_truth_score": return_meta.get("ground_truth_score"),
                "ground_truth_passed": return_meta.get("ground_truth_passed", False),
                "review_artifacts": return_meta.get("review_artifacts", []),
            },
        )
        output_path = results_dir / problem_id / "validation_result.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.model_dump_json(indent=2) + "\n")
        return result

    def _schema(self, problem_dir: Path) -> StageResult:
        issues: list[str] = []
        for rel in [
            "metadata.json",
            "task.toml",
            "instruction.md",
            "scorer/compute_score.py",
        ]:
            if not (problem_dir / rel).exists():
                issues.append(f"missing required file: {rel}")
        try:
            metadata = load_metadata(problem_dir)
            if metadata.benchmark != "taiga_task":
                issues.append("metadata.json benchmark must be taiga_task")
        except Exception as exc:
            issues.append(f"metadata.json invalid: {exc}")
        try:
            task_toml = load_task_toml(problem_dir)
            task_type = (task_toml.difficulty.task_type or "").strip().lower()
            if task_type == "ml" and task_toml.environment.allow_internet:
                issues.append(
                    "[environment].allow_internet must be false for ml tasks: the "
                    "Taiga sandbox runs offline and attempters must not control "
                    "network access. Set allow_internet = false and fetch any "
                    "dependency/dataset at image-build time in the environment "
                    "Dockerfile instead."
                )
        except Exception as exc:
            issues.append(f"task.toml invalid: {exc}")
        return StageResult(passed=not issues, issues=issues, duration_ms=0)

    def _grader_import(self, problem_dir: Path) -> StageResult:
        issues: list[str] = []
        grader_path = problem_dir / "scorer" / "compute_score.py"
        repo_root = problem_dir.parent.parent
        try:
            spec = importlib.util.spec_from_file_location(
                "task_grader_compute_score", grader_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot import {grader_path}")
            module = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str(problem_dir))
            sys.path.insert(0, str(grader_path.parent))
            shared_grader_src = repo_root / "grader" / "src"
            if shared_grader_src.exists():
                sys.path.insert(0, str(shared_grader_src))
            spec.loader.exec_module(module)
            compute_score = getattr(module, "compute_score", None)
            if not callable(compute_score):
                issues.append("scorer.compute_score must define callable compute_score")
            else:
                params = list(inspect.signature(compute_score).parameters)
                if params[:3] != ["workspace", "trajectory", "private"]:
                    issues.append(
                        "compute_score signature must start with workspace, trajectory, private"
                    )
        except Exception as exc:
            issues.append(f"grader import failed: {exc}")
        finally:
            if str(problem_dir) in sys.path:
                sys.path.remove(str(problem_dir))
            if str(grader_path.parent) in sys.path:
                sys.path.remove(str(grader_path.parent))
            shared_grader_src = repo_root / "grader" / "src"
            if str(shared_grader_src) in sys.path:
                sys.path.remove(str(shared_grader_src))
        return StageResult(passed=not issues, issues=issues, duration_ms=0)

    def _grader_sandbox(self, problem_dir: Path) -> StageResult:
        """Forbid the grader from importing or exec'ing the model's deliverables.

        The grading process runs as root so it can read hidden fixtures, so a
        grader that ``exec_module``/``import``/``exec``s agent-authored Python
        gives the submission root-level code execution: it can monkeypatch the
        grader, read the private answer key, or forge its own score (the most
        severe failure mode behind several Failed-QA problems). Submitted Python
        must instead be run via ``grading.helpers.run_policy`` in a non-root
        sandbox subprocess. This stage statically scans every ``scorer/*.py``.
        """
        issues: list[str] = []
        scorer_dir = problem_dir / "scorer"
        if not scorer_dir.is_dir():
            return StageResult(passed=True, issues=[], duration_ms=0)
        for source in sorted(scorer_dir.rglob("*.py")):
            if "__pycache__" in source.parts:
                continue
            try:
                text = source.read_text()
            except OSError:
                continue
            issues.extend(
                _grader_sandbox_issues(source.relative_to(problem_dir).as_posix(), text)
            )
        return StageResult(passed=not issues, issues=issues, duration_ms=0)

    def _agent_fault(self, problem_dir: Path) -> StageResult:
        """Enforce the AgentFault keep-vs-discard discipline across scorer files.

        Statically scans every ``scorer/*.py`` for the over-keep (broad except
        returning a score), RCE (pickle deserialization of an agent artifact),
        and over-discard (unguarded / unsignalled agent-path read) patterns.
        """
        issues: list[str] = []
        scorer_dir = problem_dir / "scorer"
        if not scorer_dir.is_dir():
            return StageResult(passed=True, issues=[], duration_ms=0)
        for source in sorted(scorer_dir.rglob("*.py")):
            if "__pycache__" in source.parts:
                continue
            try:
                text = source.read_text()
            except OSError:
                continue
            issues.extend(
                _agent_fault_issues(source.relative_to(problem_dir).as_posix(), text)
            )
        return StageResult(passed=not issues, issues=issues, duration_ms=0)

    def _prompt_runtime_references(self, problem_dir: Path) -> StageResult:
        """Reject prompt text that points agents at build/source internals."""
        instruction = problem_dir / "instruction.md"
        if not instruction.exists():
            return StageResult(passed=True, issues=[], duration_ms=0)
        try:
            text = instruction.read_text()
        except OSError as exc:
            return StageResult(
                passed=False,
                issues=[f"instruction.md could not be read: {exc}"],
                duration_ms=0,
            )
        issues = _prompt_internal_env_issues(text)
        return StageResult(passed=not issues, issues=issues, duration_ms=0)

    def _prompt_quality(self, problem_dir: Path) -> StageResult:
        """Mechanical prompt-quality gate (ported from ML_Envs).

        Enforces a minimum prompt length, that the prompt mentions the runtime
        data mount (``/data/``) and output path (``/tmp/output``), and rejects
        emoji / AI-artifact phrases. Also runs an ASCII-only scan over the
        human-authored files (the prompt and the grader, plus README* when
        present) so non-ASCII tells like em-dashes and smart quotes are caught
        with a located message.
        """
        start = time.perf_counter()
        issues: list[str] = []
        instruction = problem_dir / "instruction.md"
        if not instruction.exists():
            return StageResult(
                passed=False,
                issues=["instruction.md is required but missing"],
                duration_ms=int((time.perf_counter() - start) * 1000),
            )
        try:
            prompt_text = instruction.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return StageResult(
                passed=False,
                issues=[f"instruction.md could not be read: {exc}"],
                duration_ms=int((time.perf_counter() - start) * 1000),
            )
        require_data_token = (problem_dir / "data").is_dir()
        issues.extend(
            _prompt_quality_issues(
                "instruction.md",
                prompt_text,
                require_data_token=require_data_token,
            )
        )

        ascii_targets = [problem_dir / rel for rel in _ASCII_SCAN_FILES]
        ascii_targets.extend(sorted(problem_dir.glob("README*")))
        for source in ascii_targets:
            if not source.is_file():
                continue
            rel_path = source.relative_to(problem_dir).as_posix()
            try:
                text = source.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                issues.append(f"{rel_path} could not be read as UTF-8: {exc}")
                continue
            issues.extend(_non_ascii_issues(rel_path, text))

        return StageResult(
            passed=not issues,
            issues=issues,
            duration_ms=int((time.perf_counter() - start) * 1000),
        )

    def _scorer_determinism(self, problem_dir: Path) -> StageResult:
        """Statically reject non-deterministic scorers (deterministic CI gate).

        Scores must be reproducible. This flags wall-clock sources and unseeded
        RNG in the grading code that actually runs during grading -- the
        mechanical half of the otherwise advisory, LLM-judged "scorer
        determinism" check.
        """
        issues: list[str] = []
        scorer_dir = problem_dir / "scorer"
        if not scorer_dir.is_dir():
            return StageResult(passed=True, issues=[], duration_ms=0)
        for source in sorted(scorer_dir.rglob("*.py")):
            if "__pycache__" in source.parts:
                continue
            try:
                text = source.read_text()
            except OSError:
                continue
            issues.extend(
                _scorer_determinism_issues(
                    source.relative_to(problem_dir).as_posix(), text
                )
            )
        return StageResult(passed=not issues, issues=issues, duration_ms=0)

    def _outputs(self, problem_dir: Path) -> StageResult:
        issues: list[str] = []
        try:
            task_toml = load_task_toml(problem_dir)
            if not task_toml.outputs:
                issues.append("task.toml should declare at least one [[outputs]] entry")
            issues.extend(
                metadata_validation_issues(
                    task_type=task_toml.difficulty.task_type,
                    domain=task_toml.difficulty.domain,
                    reward_type=task_toml.difficulty.reward_type,
                    license_id=task_toml.difficulty.license,
                    license_source=task_toml.difficulty.license_source,
                )
            )
        except Exception as exc:
            issues.append(f"cannot validate outputs: {exc}")
        return StageResult(passed=not issues, issues=issues, duration_ms=0)

    def _hidden_env(self, problem_dir: Path) -> StageResult:
        """Validate ``[environment].hidden_env`` (env / hybrid) tasks.

        When a task opts into the hidden-environment RPC server, require the
        held-out env module, a public client, and that the env source is not
        leaked to the agent. A no-op for static tasks (hidden_env unset).
        """
        issues: list[str] = []
        try:
            task_toml = load_task_toml(problem_dir)
        except Exception:
            return StageResult(passed=True, issues=[], duration_ms=0)
        mode = task_toml.environment.hidden_env
        if not mode:
            return StageResult(passed=True, issues=[], duration_ms=0)

        scorer_data = problem_dir / "scorer" / "data"
        public_data = problem_dir / "data"

        # Resolve the env module from env_config.json (if present) or the default
        # scorer/data/env.py. The module is baked to /mcp_server/data/ (root-only).
        module_rel = "env.py"
        factory = "make_env"
        config_path = scorer_data / "env_config.json"
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text())
                if not isinstance(cfg, dict):
                    raise ValueError("env_config.json must be a JSON object")
                module_rel = cfg.get("module", module_rel)
                factory = cfg.get("factory", factory)
                if not isinstance(module_rel, str) or not isinstance(factory, str):
                    issues.append(
                        "scorer/data/env_config.json: 'module' and 'factory' must "
                        "be strings"
                    )
            except Exception as exc:
                issues.append(f"scorer/data/env_config.json is invalid: {exc}")

        env_module = scorer_data / module_rel
        if not env_module.exists():
            issues.append(
                f"[environment].hidden_env = {mode!r} requires the held-out env "
                f"module at scorer/data/{module_rel} (baked root-only to "
                f"/mcp_server/data/), but it is missing."
            )
        else:
            try:
                if f"def {factory}" not in env_module.read_text():
                    issues.append(
                        f"scorer/data/{module_rel} must define a {factory}(**kwargs) "
                        f"factory returning the env instance."
                    )
            except OSError as exc:
                issues.append(f"cannot read scorer/data/{module_rel}: {exc}")

        # The agent needs a client to reach /tmp/env.sock; ship it publicly.
        if not (public_data / "env_client.py").exists():
            issues.append(
                f"[environment].hidden_env = {mode!r} requires a public "
                "data/env_client.py (agent-facing client; baked to "
                "/data/env_client.py). Copy env_server/env_client.py and adapt it."
            )

        # The env source must stay hidden: never ship it under the public data/.
        if (public_data / module_rel).exists():
            issues.append(
                f"the hidden env module data/{module_rel} is agent-visible "
                f"(baked to /data/); the env source must live ONLY under "
                f"scorer/data/ (root-only /mcp_server/data/) so the agent cannot "
                f"read the hidden dynamics. Remove data/{module_rel}."
            )

        return StageResult(passed=not issues, issues=issues, duration_ms=0)

    def _ground_truth(self, problem_dir: Path) -> StageResult:
        issues: list[str] = []
        try:
            task_toml = load_task_toml(problem_dir)
            ground_truth = task_toml.ground_truth
            render_required = render_expected(
                task_toml.difficulty.task_type, ground_truth.render_outputs
            )
            if render_required and not (problem_dir / "solution" / "solve.sh").exists():
                issues.append("ground truth requires solution/solve.sh")
            if render_required and not ground_truth.render_command.strip():
                issues.append("task.toml [ground_truth].render_command is required")
            if render_required and not ground_truth.render_outputs:
                issues.append(
                    "task.toml [ground_truth].render_outputs must declare at least one video"
                )
            for output in ground_truth.render_outputs:
                if not output.required:
                    issues.append(
                        f"ground truth render output must be required: {output.path}"
                    )
                rel = _output_relative_path(output.path)
                if rel is None:
                    issues.append(
                        f"ground truth render output must be under /tmp/output: {output.path}"
                    )
                elif rel.suffix.lower() not in VIDEO_SUFFIXES:
                    issues.append(
                        "ground truth render output must be a video file "
                        f"({', '.join(sorted(VIDEO_SUFFIXES))}): {output.path}"
                    )
                elif rel.name != "rendering.mp4":
                    issues.append(
                        "ground truth render output must be named /tmp/output/rendering.mp4"
                    )
        except Exception as exc:
            issues.append(f"cannot validate ground truth config: {exc}")
        return StageResult(passed=not issues, issues=issues, duration_ms=0)

    def _private_data_layout(self, problem_dir: Path) -> StageResult:
        issues: list[str] = []
        dockerfile = problem_dir / "environment" / "Dockerfile"
        if dockerfile.exists():
            try:
                issues.extend(_dockerfile_private_layout_issues(dockerfile))
            except OSError as exc:
                issues.append(f"could not inspect Dockerfile private layout: {exc}")
        issues.extend(_public_private_duplicate_issues(problem_dir))
        return StageResult(passed=not issues, issues=issues, duration_ms=0)

    def _solution_answer_key_leak(self, problem_dir: Path) -> StageResult:
        """Hard-fail when the reference solution reads the private answer key."""
        start = time.monotonic()
        issues = _solution_private_read_issues(problem_dir)
        return StageResult(
            passed=not issues,
            issues=issues,
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    def _local_build_proof(self, problem_dir: Path) -> StageResult:
        passed, errors, _proof = verify_build_proof(problem_dir)
        if passed:
            return StageResult(passed=True, issues=[], duration_ms=0)

        start = time.monotonic()
        try:
            repo_root = problem_dir.parent.parent
            base = ensure_local_base_image(repo_root, problem_dir)
            problem_rel = problem_dir.relative_to(repo_root)
            image_tag = f"local/{task_id(problem_dir)}:build-proof"
            iidfile = problem_dir / ".alignerr" / "image.iid"
            iidfile.parent.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(
                [
                    "docker",
                    "buildx",
                    "build",
                    "--load",
                    "--platform",
                    "linux/amd64",
                    "--file",
                    str(problem_dir / "environment" / "Dockerfile"),
                    "--build-arg",
                    f"BASE_IMAGE={base.image}",
                    "--build-arg",
                    f"BASE_TAG={base.tag}",
                    "--build-arg",
                    f"PROBLEM_DIR={problem_rel.as_posix()}",
                    "--tag",
                    image_tag,
                    "--iidfile",
                    str(iidfile),
                    str(repo_root),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                return StageResult(
                    passed=False,
                    issues=[
                        *errors,
                        f"docker build failed: {completed.stderr.strip()}",
                    ],
                    duration_ms=int((time.monotonic() - start) * 1000),
                )
            probe_error = _run_private_layout_image_probe(image_tag)
            if probe_error is not None:
                return StageResult(
                    passed=False,
                    issues=[*errors, probe_error],
                    duration_ms=int((time.monotonic() - start) * 1000),
                )
            agent_python_error = _run_agent_python_image_probe(image_tag)
            if agent_python_error is not None:
                return StageResult(
                    passed=False,
                    issues=[*errors, agent_python_error],
                    duration_ms=int((time.monotonic() - start) * 1000),
                )
            image_digest = (
                iidfile.read_text().strip() if iidfile.exists() else image_tag
            )
            write_build_proof(
                problem_dir,
                image_digest=image_digest,
                base_image_ref=base.ref,
                platform="linux/amd64",
                alignerr_cli_version="0.1.0",
                duration_seconds=time.monotonic() - start,
            )
        except Exception as exc:
            return StageResult(
                passed=False,
                issues=[*errors, f"could not create build proof: {exc}"],
                duration_ms=int((time.monotonic() - start) * 1000),
            )

        return StageResult(
            passed=True, issues=[], duration_ms=int((time.monotonic() - start) * 1000)
        )

    def _conditional(self, problem_dir: Path) -> StageResult:
        issues: list[str] = []

        compose_file = problem_dir / "environment" / "docker-compose.yaml"
        if compose_file.exists():
            completed = subprocess.run(
                ["docker", "compose", "-f", str(compose_file), "config"],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                issues.append(
                    f"docker compose config failed: {completed.stderr.strip()}"
                )

        for task_file in (
            (problem_dir / "tasks").glob("*.json")
            if (problem_dir / "tasks").exists()
            else []
        ):
            task_data = json.loads(task_file.read_text())
            unknown_tools = set(task_data.get("tools", [])) - KNOWN_TOOLS
            if unknown_tools:
                issues.append(
                    f"{task_file.name} uses unknown tools: {sorted(unknown_tools)}"
                )

        if (problem_dir / "scorer" / "data" / "envs" / "__init__.py").exists():
            # A full rollout smoke belongs in the implementation pass; this scaffold catches obvious omissions.
            envs_text = (
                problem_dir / "scorer" / "data" / "envs" / "__init__.py"
            ).read_text()
            if "make_env" not in envs_text:
                issues.append(
                    "scorer/data/envs/__init__.py exists but does not define make_env"
                )

        issues.extend(_preloaded_files_issues(problem_dir))

        return StageResult(passed=not issues, issues=issues, duration_ms=0)

    def _compute_score_return(
        self, problem_dir: Path
    ) -> tuple[StageResult, dict[str, Any]]:
        """Sample-run the grader against an empty workspace + the reference
        solution (when available) and verify the return shape normalizes.

        Records:
          * ``return_shape``  one of ``bare_float`` / ``score_dict`` / ``rubric_grade`` / ``error``
          * ``sample_score``  float in [0, 1] from the reference run (or 0.0 when
                              there is no reference / the grader errors)
          * ``uses_llm_judge``  static check of the grader source; LLM judges are disallowed

        Asserts the normalized headline lands in ``[0, 1]`` (catches the most
        common ML_Envs bug — forgotten clip on a custom anchor mapping).
        """
        import tempfile

        issues: list[str] = []
        meta: dict[str, Any] = {
            "return_shape": "unknown",
            "uses_llm_judge": False,
            "sample_score": None,
            "ground_truth_score": None,
            "ground_truth_passed": False,
            "review_artifacts": [],
        }
        start = time.monotonic()

        grader_source = problem_dir / "scorer" / "compute_score.py"
        if not grader_source.exists():
            issues.append(
                "scorer/compute_score.py is missing; tasks must define a scorer."
            )
            return (
                StageResult(
                    passed=False,
                    issues=issues,
                    duration_ms=int((time.monotonic() - start) * 1000),
                ),
                meta,
            )

        try:
            text = grader_source.read_text()
            meta["uses_llm_judge"] = "llm_criterion" in text or "LLMJudge" in text
            if meta["uses_llm_judge"]:
                issues.append(
                    "scorer/compute_score.py uses an LLM judge. Rubrics must be deterministic; "
                    "replace rb.llm_criterion/LLMJudge usage with code-checkable criteria."
                )
        except OSError:
            pass

        # In-container ground-truth tasks (e.g. OpenFOAM/SU2/Meep/OpenROAD) cannot
        # be probed on the host -- their grader invokes engines that live only in
        # the task image. The oracle score + reviewer artifacts are produced and
        # committed by the in-container ground-truth harness run instead.
        try:
            _task_toml_for_gt = load_task_toml(problem_dir)
            _gt_in_container = _task_toml_for_gt.ground_truth.in_container
        except Exception:
            _task_toml_for_gt = None
            _gt_in_container = False
        if _gt_in_container:
            meta["ground_truth_in_container"] = True
            issues.extend(
                _validate_committed_ground_truth_result(
                    problem_dir,
                    task_toml=_task_toml_for_gt,
                    meta=meta,
                )
            )
            return (
                StageResult(
                    passed=not issues,
                    issues=issues,
                    duration_ms=int((time.monotonic() - start) * 1000),
                ),
                meta,
            )

        # Add the shared grading library and the task-local scorer dir to sys.path
        # so `from grading import ...` and scorer-local imports resolve.
        # File at: <repo>/alignerr_plugin/src/alignerr_plugin/validators/task/validator.py
        # parents[5] is repo root.
        repo_root = Path(__file__).resolve().parents[5]
        grading_src = repo_root / "grader" / "src"
        added: list[str] = []
        for p in (str(grading_src), str(grader_source.parent)):
            if p not in sys.path:
                sys.path.insert(0, p)
                added.append(p)

        module = None
        try:
            spec = importlib.util.spec_from_file_location(
                "task_grader_compute_score_probe", grader_source
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot import {grader_source}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:
            issues.append(f"grader import failed: {exc}")
            for p in added:
                if p in sys.path:
                    sys.path.remove(p)
            return (
                StageResult(
                    passed=False,
                    issues=issues,
                    duration_ms=int((time.monotonic() - start) * 1000),
                ),
                meta,
            )

        compute_score = getattr(module, "compute_score", None)
        if not callable(compute_score):
            issues.append("compute_score must be a callable in compute_score.py")
            for p in added:
                if p in sys.path:
                    sys.path.remove(p)
            return (
                StageResult(
                    passed=False,
                    issues=issues,
                    duration_ms=int((time.monotonic() - start) * 1000),
                ),
                meta,
            )

        # Probe with a temp workspace. If the task ships a reference
        # `solution/solve.sh`, run it inside the temp workspace first so
        # the grader sees a "passing" submission.
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "workspace"
            workspace.mkdir()
            _gt = load_task_toml(problem_dir)
            render_required = render_expected(
                _gt.difficulty.task_type, _gt.ground_truth.render_outputs
            )
            solution = problem_dir / "solution" / "solve.sh"
            has_solution = solution.exists()
            if has_solution:
                # Rewrite container paths so the script can run on the host:
                #   /tmp/output -> our temp workspace
                #   /data       -> the task's `data/` directory
                # The reference solution is the oracle path for every task type;
                # failure is fatal regardless of whether a reviewer video is
                # declared.
                src = solution.read_text()
                src = src.replace("/tmp/output", str(workspace))

                # Substituted paths get evaluated by bash from `cwd=workspace`
                # (a temp dir), not from this Python process's cwd, so they
                # must be absolute:
                host_data = (problem_dir / "data").resolve()
                if host_data.exists():
                    src = src.replace("/data/", str(host_data) + "/")
                completed = subprocess.run(
                    ["bash", "-c", src],
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                )
                if completed.returncode != 0:
                    meta["reference_solution_exit"] = completed.returncode
                    meta["reference_solution_stderr_tail"] = completed.stderr.strip()[
                        :200
                    ]
                    issues.append(
                        f"ground truth solution exited with status {completed.returncode}: "
                        f"{completed.stderr.strip()[:200]}"
                    )
                    for p in added:
                        if p in sys.path:
                            sys.path.remove(p)
                    return (
                        StageResult(
                            passed=False,
                            issues=issues,
                            duration_ms=int((time.monotonic() - start) * 1000),
                        ),
                        meta,
                    )
            elif render_required:
                issues.append("ground truth solution/solve.sh is missing")
                for p in added:
                    if p in sys.path:
                        sys.path.remove(p)
                return (
                    StageResult(
                        passed=False,
                        issues=issues,
                        duration_ms=int((time.monotonic() - start) * 1000),
                    ),
                    meta,
                )

            try:
                from grading import normalize_compute_score_return  # type: ignore
                from grading.rubric_builder import RubricBuilder  # type: ignore
            except Exception as exc:
                issues.append(f"could not import shared grading library: {exc}")
                for p in added:
                    if p in sys.path:
                        sys.path.remove(p)
                return (
                    StageResult(
                        passed=False,
                        issues=issues,
                        duration_ms=int((time.monotonic() - start) * 1000),
                    ),
                    meta,
                )

            try:
                params = list(inspect.signature(compute_score).parameters)
                args: list[Any] = []
                kwargs: dict[str, Any] = {}
                if params[:3] == ["workspace", "trajectory", "private"]:
                    args = [workspace, None, problem_dir / "scorer" / "data"]
                else:
                    issues.append(
                        "compute_score signature must start with "
                        "(workspace, trajectory, private)"
                    )
                raw = compute_score(*args, **kwargs)
                if isinstance(raw, RubricBuilder):
                    raw = raw.grade().to_dict()
            except Exception as exc:
                issues.append(f"compute_score raised during probe: {exc}")
                for p in added:
                    if p in sys.path:
                        sys.path.remove(p)
                return (
                    StageResult(
                        passed=not issues,
                        issues=issues,
                        duration_ms=int((time.monotonic() - start) * 1000),
                    ),
                    meta,
                )

            try:
                grade = normalize_compute_score_return(raw)
            except Exception as exc:
                issues.append(
                    f"compute_score return value could not be normalized: {exc}. "
                    "Return a float, a dict with at least a 'score' key, or "
                    "a Grade (e.g. RubricBuilder.grade())."
                )
                for p in added:
                    if p in sys.path:
                        sys.path.remove(p)
                return (
                    StageResult(
                        passed=False,
                        issues=issues,
                        duration_ms=int((time.monotonic() - start) * 1000),
                    ),
                    meta,
                )

            sample_score = grade.score()
            meta["sample_score"] = sample_score
            meta["ground_truth_score"] = sample_score
            if isinstance(raw, float | int):
                meta["return_shape"] = "bare_float"
            elif hasattr(raw, "to_dict") and not isinstance(raw, dict):
                meta["return_shape"] = "rubric_grade"
            elif isinstance(raw, dict):
                # Look for the structured_subscores marker that to_dict
                # would have emitted -- otherwise a "score_dict" return.
                meta["return_shape"] = (
                    "rubric_grade" if "structured_subscores" in raw else "score_dict"
                )

            if not (0.0 <= sample_score <= 1.0):
                issues.append(
                    f"compute_score returned a sample score {sample_score} outside [0, 1]. "
                    "Clamp the headline before returning (this is the most common "
                    "ML_Envs migration bug)."
                )
            _gt2 = load_task_toml(problem_dir)
            expectation = expected_ground_truth_score(
                _gt2.difficulty.reward_type,
                deterministic_epsilon=_gt2.ground_truth.score_epsilon,
                continuous_epsilon=_gt2.ground_truth.continuous_score_epsilon,
            )
            render_required = render_expected(
                _gt2.difficulty.task_type, _gt2.ground_truth.render_outputs
            )
            if has_solution and not expectation.passed(sample_score):
                failed = failed_criteria(grade.to_dict())
                detail = (
                    f" Failed criteria: {'; '.join(failed[:10])}." if failed else ""
                )
                issues.append(
                    "ground truth solution for reward_type "
                    f"{expectation.reward_type!r} must score {expectation.description}, "
                    f"got {sample_score:.6f}.{detail}"
                )
            elif has_solution:
                meta["ground_truth_passed"] = True

            # Anti-reward-hacking gate: a no-op submission (and, when cleanly
            # extractable, the prompt's own example) must NOT out-score genuine
            # work. This catches graders where "submit nothing" or "copy the
            # example" beats real attempts -- a dominant Failed-QA failure mode.
            try:
                max_trivial = _gt2.ground_truth.max_trivial_score
            except Exception:
                max_trivial = 0.5
            for label, trivial_score in _trivial_submission_scores(
                problem_dir,
                compute_score,
                params,
                normalize_compute_score_return,
                RubricBuilder,
                meta,
            ):
                if trivial_score is not None and trivial_score > max_trivial + 1e-9:
                    issues.append(
                        f"{label} scores {trivial_score:.3f} through the real "
                        f"grader, above the max_trivial_score ceiling "
                        f"({max_trivial:.3f}). A trivial submission must not "
                        f"out-score genuine work: recalibrate so doing nothing "
                        f"(or copying the example) scores ~0, gate guardrail "
                        f"subscores on non-trivial progress, and raise/lower "
                        f"[ground_truth].max_trivial_score only with justification."
                    )

            # Calibration learnability gate: for continuous tasks, committed
            # naive baselines must score clearly below the reference (0.5).
            if expectation.reward_type == "continuous_scoring_function":
                issues.extend(
                    _baseline_calibration_issues(
                        problem_dir,
                        compute_score,
                        params,
                        normalize_compute_score_return,
                        RubricBuilder,
                        target=expectation.target,
                    )
                )

            try:
                if render_required:
                    proof_artifacts = _valid_committed_render_artifacts(problem_dir)
                    if proof_artifacts is not None:
                        meta["review_artifacts"] = proof_artifacts
                    else:
                        render_artifacts = _run_ground_truth_render_probe(
                            problem_dir, workspace
                        )
                        meta["review_artifacts"] = render_artifacts
            except Exception as exc:
                issues.append(f"ground truth render probe failed: {exc}")

        for p in added:
            if p in sys.path:
                sys.path.remove(p)

        return (
            StageResult(
                passed=not issues,
                issues=issues,
                duration_ms=int((time.monotonic() - start) * 1000),
            ),
            meta,
        )


def _output_relative_path(path: str) -> Path | None:
    prefix = "/tmp/output/"
    if path == "/tmp/output":
        return Path(".")
    if not path.startswith(prefix):
        return None
    return Path(path.removeprefix(prefix))


_PROMPT_PACKAGE_WORDS = (
    "package",
    "packages",
    "dependency",
    "dependencies",
    "library",
    "libraries",
    "installed",
    "available",
    "ships",
    "includes",
)
_PROMPT_PACKAGE_WORD_RE = "|".join(re.escape(word) for word in _PROMPT_PACKAGE_WORDS)


def _prompt_internal_env_issues(text: str) -> list[str]:
    """Reject prompt guidance that exposes build/source metadata as an oracle.

    Agents should be told which tools and libraries are installed directly, not
    pointed at base-image/source metadata or task metadata as a dependency list.
    Dataset metadata files under /data are allowed.
    """
    issues: list[str] = []
    normalized = re.sub(r"\s+", " ", text.strip())
    lower = normalized.lower()
    segments = re.split(r"(?<=[.!?])\s+", lower)
    base_image_dependency_reference = any(
        re.search(
            rf"\b(base image|base-image)\b.{{0,120}}\b({_PROMPT_PACKAGE_WORD_RE})\b",
            segment,
        )
        or re.search(
            rf"\b({_PROMPT_PACKAGE_WORD_RE})\b.{{0,120}}\b(base image|base-image)\b",
            segment,
        )
        for segment in segments
    )
    if base_image_dependency_reference:
        issues.append(
            "instruction.md must not direct the agent to infer available "
            "packages from the Docker/base image. Name installed tools and "
            "libraries directly instead."
        )
    prompt_without_data_metadata = re.sub(r"/data/[^\s,;:)]*metadata\.json", "", lower)
    if re.search(r"\bmetadata\.json", prompt_without_data_metadata) and re.search(
        r"\b(dependencies|dependency|packages|libraries|installed|declared)\b",
        prompt_without_data_metadata,
    ):
        issues.append(
            "instruction.md must not point the agent at task metadata.json or "
            "declared task dependencies as a package source. Name installed "
            "tools and libraries directly instead."
        )
    if re.search(r"\bthis task'?s declared dependencies\b", lower):
        issues.append(
            "instruction.md must not refer to this task's declared dependencies "
            "as runtime guidance; name installed tools and libraries directly."
        )
    return issues


def _prompt_quality_issues(
    rel_path: str, text: str, *, require_data_token: bool = True
) -> list[str]:
    """Return prompt-quality findings (length, runtime tokens, AI-spam tells).

    Ported from ML_Envs ``validate_problem_quality``. Each message is prefixed
    with ``rel_path`` to match the style of ``_scorer_determinism_issues``.

    ``require_data_token`` gates the ``/data/`` mention: it is a hard error only
    for tasks that actually ship a public ``data/`` mount. Pure simulator tasks
    (mujoco, hidden-env) have no ``/data/`` mount -- their data is the
    environment -- so requiring it there would be a false positive.
    """
    issues: list[str] = []
    if len(text) < _MIN_PROMPT_LENGTH:
        issues.append(
            f"{rel_path}: prompt must be at least {_MIN_PROMPT_LENGTH} characters "
            f"(got {len(text)})"
        )
    if require_data_token and "/data/" not in text:
        issues.append(
            f"{rel_path}: prompt must describe the data available in /data/"
        )
    if "/tmp/output" not in text:
        issues.append(
            f"{rel_path}: prompt must describe the output path in /tmp/output/"
        )
    for pattern in AI_ARTIFACT_PATTERNS:
        if pattern.search(text):
            issues.append(
                f"{rel_path}: prompt contains AI-artifact phrase matching "
                f"/{pattern.pattern}/"
            )
    if EMOJI_PATTERN.search(text):
        issues.append(f"{rel_path}: prompt appears to contain emoji characters")
    return issues


def _non_ascii_issues(rel_path: str, text: str) -> list[str]:
    """Flag any non-ASCII characters, located by line, in a human-authored file.

    Catches em-dashes, smart quotes, and other AI-spam tells that break agents
    interpolating the text into code/JSON. Reports the line number and the
    offending character(s) (with code points) so authors can find them.
    """
    issues: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        offending = sorted({ch for ch in line if ord(ch) > 0x7F})
        if not offending:
            continue
        rendered = ", ".join(f"{ch!r} (U+{ord(ch):04X})" for ch in offending)
        issues.append(
            f"{rel_path}:{lineno}: non-ASCII character(s) not allowed: {rendered}"
        )
    return issues


def _grader_sandbox_issues(rel_path: str, text: str) -> list[str]:
    """Return findings for dynamic execution of model code in a grader file."""
    issues: list[str] = []
    seen: set[tuple[str, int]] = set()

    def record(name: str, lineno: int) -> None:
        key = (name, lineno)
        if key in seen:
            return
        seen.add(key)
        issues.append(
            f"{rel_path}:{lineno}: grader uses `{name}`, which loads/executes "
            f"code from a file path. {_SANDBOX_HELPER_HINT}"
        )

    try:
        tree = ast.parse(text)
    except SyntaxError:
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.split("#", 1)[0]
            for marker in (*_DYNAMIC_CODE_EXEC_MARKERS, *_DYNAMIC_CODE_EXEC_ATTR_ONLY):
                if marker in stripped:
                    record(marker, lineno)
            if re.search(r"\b(exec|eval)\s*\(", stripped):
                record("exec/eval", lineno)
        return issues

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and (
            node.attr in _DYNAMIC_CODE_EXEC_MARKERS
            or node.attr in _DYNAMIC_CODE_EXEC_ATTR_ONLY
        ):
            record(node.attr, node.lineno)
        elif isinstance(node, ast.Name) and node.id in _DYNAMIC_CODE_EXEC_MARKERS:
            record(node.id, node.lineno)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in {"exec", "eval"}:
                record(func.id, node.lineno)
    return issues


# Wall-clock calls whose return value makes a score depend on when grading ran.
_WALLCLOCK_CALLS = (
    "time.time",
    "time.time_ns",
    "datetime.now",
    "datetime.utcnow",
    "datetime.today",
)
def _scorer_determinism_issues(rel_path: str, text: str) -> list[str]:
    """Flag wall-clock / unseeded-RNG non-determinism in live grading code.

    Only analyzes code that actually runs during grading: module-level statements
    (executed on import) plus functions reachable from ``compute_score``. Helper
    modules that do not define ``compute_score`` are checked at module level only;
    without cross-file call graph information, treating every helper function as
    live creates false positives for dead utilities. Seeded RNG is allowed only
    when the seed call is itself on the live path.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []  # the schema / import stages report parse errors

    func_defs = {
        fn.name
        for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    reachable = _reachable_from_compute_score(tree)
    enclosing = _enclosing_func_name(tree)
    imports = _import_aliases(tree)
    parents = _parent_map(tree)
    import_call_lines = _module_level_call_lines(tree)
    import_reachable = set(import_call_lines)
    function_call_lines = _function_call_lines(tree)
    has_compute_score = "compute_score" in func_defs

    def _live(node: ast.AST) -> bool:
        name = enclosing.get(id(node))
        if name is None:
            return True  # module-level code runs when the grader imports the scorer
        if name in import_reachable:
            return True  # called by module-level initialization code
        if has_compute_score:
            return name in reachable
        return False  # helper-only module: function bodies are live only if called

    def _scope(node: ast.AST) -> str | None:
        return enclosing.get(id(node))

    def _execution_line(node: ast.AST) -> int:
        scope = _scope(node)
        if scope in import_call_lines:
            return import_call_lines[scope]
        return node.lineno

    issues: list[str] = []
    rng_flagged = False
    live_calls: list[tuple[ast.Call, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _live(node):
            continue
        try:
            func_src = _canonical_call_name(node.func, imports)
        except Exception:
            continue
        live_calls.append((node, func_src))
    live_global_seeds: list[tuple[str | None, int, set[str]]] = [
        (
            _scope(node),
            node.lineno,
            kinds,
        )
        for node, func_src in live_calls
        if _unconditional_in_scope(node, parents)
        if (kinds := _rng_global_seed_kinds(func_src, node))
    ]
    live_global_seeds = _propagate_global_seeds(
        live_global_seeds,
        import_call_lines=import_call_lines,
        function_call_lines=function_call_lines,
    )

    def _has_prior_seed(node: ast.Call, kind: str) -> bool:
        scope = _scope(node)
        execution_line = _execution_line(node)
        return any(
            (
                (
                    seed_scope is None
                    and scope is not None
                    and (scope not in import_reachable or seed_lineno < execution_line)
                )
                or (
                    seed_scope == scope
                    and seed_lineno < execution_line
                )
                or (
                    seed_scope in function_call_lines
                    and scope in function_call_lines[seed_scope]
                    and seed_lineno < function_call_lines[seed_scope][scope]
                )
            )
            and (kind in seed_kinds or "all" in seed_kinds)
            for seed_scope, seed_lineno, seed_kinds in live_global_seeds
        )

    for node, func_src in live_calls:
        if any(func_src == c or func_src.endswith("." + c) for c in _WALLCLOCK_CALLS):
            issues.append(
                f"{rel_path}:{node.lineno}: scorer calls `{func_src}()`, a "
                "wall-clock source that makes the score depend on when grading "
                "ran. Scores must be reproducible; remove the time/date dependence."
            )
            continue
        if _rng_call_needs_seed(func_src, node, _has_prior_seed) and not rng_flagged:
            rng_flagged = True
            issues.append(
                f"{rel_path}:{node.lineno}: scorer uses `{func_src}()` but no RNG "
                "is seeded in the file, so the score is non-deterministic. Seed it "
                "(e.g. rng = np.random.default_rng(<fixed>)) so grading is "
                "reproducible."
            )
    return issues


def _rng_global_seed_kinds(func_src: str, node: ast.Call) -> set[str]:
    """Module-global RNG families seeded by this live call.

    Seed scope matters: ``np.random.default_rng(0)`` creates a seeded local
    generator, but it does NOT seed the module-global ``np.random`` functions.
    Likewise ``random.Random(0)`` does not seed ``random.random``. Only calls that
    seed the global family suppress findings for that family, and the caller only
    counts unconditional, same-scope, prior seed calls.
    """
    if not _seed_call_has_fixed_literal(node):
        return set()
    if func_src in {"np.random.seed", "numpy.random.seed"}:
        return {"numpy_global"}
    if func_src == "random.seed":
        return {"python_global"}
    return set()


def _seed_call_has_fixed_literal(node: ast.Call) -> bool:
    if node.args:
        return _fixed_seed_literal(node.args[0])
    for keyword in node.keywords:
        if keyword.arg in {"seed", "x"}:
            return _fixed_seed_literal(keyword.value)
    return False


def _rng_call_needs_seed(func_src: str, node: ast.Call, has_prior_seed) -> bool:
    """True when this RNG call is an entropy source not covered by a live seed."""
    if func_src.endswith(".default_rng") or func_src.endswith(".RandomState"):
        # Seeded constructors are deterministic; unseeded constructors draw
        # entropy. They do not seed module-global np.random either way.
        return not _rng_constructor_has_fixed_seed(node, keyword_names={"seed"})
    if func_src == "random.Random":
        return not _rng_constructor_has_fixed_seed(node, keyword_names={"x"})
    if "np.random." in func_src or "numpy.random." in func_src:
        if func_src in {"np.random.seed", "numpy.random.seed"}:
            return False
        return not has_prior_seed(node, "numpy_global")
    if func_src.startswith("random."):
        if func_src == "random.seed":
            return False
        return not has_prior_seed(node, "python_global")
    return False


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".", 1)[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == "*":
                    continue
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _canonical_call_name(func: ast.AST, imports: dict[str, str]) -> str:
    raw = ast.unparse(func)
    parts = raw.split(".")
    if not parts:
        return raw
    head = imports.get(parts[0])
    if head is None:
        return raw
    return ".".join([head, *parts[1:]])


def _rng_constructor_has_fixed_seed(
    node: ast.Call, *, keyword_names: set[str]
) -> bool:
    """True when a RNG constructor receives an obvious fixed seed literal."""
    if node.args:
        return _fixed_seed_literal(node.args[0])
    for keyword in node.keywords:
        if keyword.arg in keyword_names:
            return _fixed_seed_literal(keyword.value)
    return False


def _fixed_seed_literal(node: ast.AST) -> bool:
    """Conservative literal-seed check: fixed constants are OK; None/vars are not."""
    return isinstance(node, ast.Constant) and node.value is not None


def _parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    return {
        id(child): node
        for node in ast.walk(tree)
        for child in ast.iter_child_nodes(node)
    }


def _unconditional_in_scope(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    """True when ``node`` is not guarded by branch/loop/exception control flow."""
    guarded = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.Try,
        ast.With,
        ast.AsyncWith,
        ast.Match,
        ast.IfExp,
        ast.BoolOp,
    )
    cur = node
    while id(cur) in parents:
        cur = parents[id(cur)]
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return True
        if isinstance(cur, guarded):
            return False
    return True


def _reachable_from_compute_score(tree: ast.AST) -> set[str]:
    """Names of functions transitively called from ``compute_score`` (inclusive).

    The runner only ever calls ``compute_score``; a guard or read inside a
    leftover ``test_*`` / unused helper the runner never invokes is dead during
    grading, so the lint only analyzes code on this live set (plus module-level,
    handled by the caller). Empty when ``compute_score`` is absent.
    """
    defs: dict[str, ast.AST] = {
        fn.name: fn
        for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "compute_score" not in defs:
        return set()
    reachable: set[str] = set()
    stack = ["compute_score"]
    while stack:
        name = stack.pop()
        if name in reachable or name not in defs:
            continue
        reachable.add(name)
        for node in ast.walk(defs[name]):
            if isinstance(node, ast.Call):
                func = node.func
                callee = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr
                    if isinstance(func, ast.Attribute)
                    else None
                )
                if callee in defs and callee not in reachable:
                    stack.append(callee)
    return reachable


def _module_level_call_lines(tree: ast.AST) -> dict[str, int]:
    """Functions called by module-level import-time code and their call order."""
    parents = _parent_map(tree)
    defs: dict[str, ast.AST] = {
        fn.name: fn
        for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    root_calls: dict[str, int] = {}
    for stmt in getattr(tree, "body", []):
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Call)
                and _unconditional_in_scope(node, parents)
            ):
                func = node.func
                callee = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr
                    if isinstance(func, ast.Attribute)
                    else None
                )
                if callee in defs:
                    root_calls[callee] = min(root_calls.get(callee, node.lineno), node.lineno)

    reachable: dict[str, int] = {}
    stack = list(root_calls.items())
    while stack:
        name, root_lineno = stack.pop()
        if name in reachable or name not in defs:
            continue
        reachable[name] = root_lineno
        for node in ast.walk(defs[name]):
            if isinstance(node, ast.Call):
                func = node.func
                callee = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr
                    if isinstance(func, ast.Attribute)
                    else None
                )
                if callee in defs and callee not in reachable:
                    stack.append((callee, root_lineno))
    return reachable


def _function_call_lines(tree: ast.AST) -> dict[str, dict[str, int]]:
    """Unconditional direct function-call line numbers within each function."""
    parents = _parent_map(tree)
    defs = {
        fn.name
        for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    calls: dict[str, dict[str, int]] = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        by_callee: dict[str, int] = {}
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call) or not _unconditional_in_scope(node, parents):
                continue
            func = node.func
            callee = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if callee in defs:
                by_callee[callee] = min(by_callee.get(callee, node.lineno), node.lineno)
        calls[fn.name] = by_callee
    return calls


def _propagate_global_seeds(
    seeds: list[tuple[str | None, int, set[str]]],
    *,
    import_call_lines: dict[str, int],
    function_call_lines: dict[str, dict[str, int]],
) -> list[tuple[str | None, int, set[str]]]:
    """Lift fixed global seeds through unconditional helper calls to callers.

    A seed inside ``_seed`` can cover a later ``_score`` only when the caller
    invokes ``_seed`` unconditionally before ``_score``. For import-time helpers,
    the seed becomes module-level at the module call site, preserving import
    execution order.
    """
    out: list[tuple[str | None, int, set[str]]] = list(seeds)
    seen = {(scope, line, tuple(sorted(kinds))) for scope, line, kinds in out}
    changed = True
    while changed:
        changed = False
        for seed_scope, _seed_lineno, kinds in list(out):
            if seed_scope is None:
                continue
            propagated: list[tuple[str | None, int, set[str]]] = []
            if seed_scope in import_call_lines:
                propagated.append((None, import_call_lines[seed_scope], kinds))
            for caller, callees in function_call_lines.items():
                call_lineno = callees.get(seed_scope)
                if call_lineno is not None:
                    propagated.append((caller, call_lineno, kinds))
            for item in propagated:
                key = (item[0], item[1], tuple(sorted(item[2])))
                if key not in seen:
                    seen.add(key)
                    out.append(item)
                    changed = True
    return out


def _enclosing_func_name(tree: ast.AST) -> dict[int, str | None]:
    """Map each node id to the name of the function lexically enclosing it
    (``None`` at module level)."""
    parent: dict[int, str | None] = {}

    def visit(node: ast.AST, fname: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = fname
            next_fname = (
                child.name
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                else fname
            )
            visit(child, next_fname)

    visit(tree, None)
    return parent


def _on_live_path(node_id: int, enclosing: dict[int, str | None], reachable: set[str]) -> bool:
    """True if the node runs at grade time: module-level (exec'd when the runner
    imports the scorer) or inside a function reachable from compute_score."""
    fname = enclosing.get(node_id)
    return fname is None or fname in reachable


def _raises_agent_fault(tree: ast.AST, enclosing: dict[int, str | None], reachable: set[str]) -> bool:
    """True if ``AgentFault`` is raised anywhere on the live grading path."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        if not _on_live_path(id(node), enclosing, reachable):
            continue
        exc = node.exc
        if isinstance(exc, ast.Call):
            exc = exc.func
        if isinstance(exc, ast.Name) and exc.id == _AGENT_FAULT_NAME:
            return True
        if isinstance(exc, ast.Attribute) and exc.attr == _AGENT_FAULT_NAME:
            return True
    return False


def _handler_exc_names(handler: ast.ExceptHandler) -> set[str]:
    if handler.type is None:
        return set()
    nodes = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    names: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _handler_is_broad(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True  # bare except
    return bool(_handler_exc_names(handler) & _BROAD_EXC_NAMES)


def _handler_guards_dir_veto(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True  # bare except catches OSError too
    return bool(_handler_exc_names(handler) & _DIR_VETO_GUARD_EXC)


def _return_is_score_like(value: ast.AST) -> bool:
    """True if a returned value looks like a final grade (so swallowing an
    exception into it is the over-keep anti-pattern), vs an intermediate sentinel
    like ``(np.zeros(nu), False)`` or ``{"valid": False}`` that a helper returns
    to the scoring logic.

    Score-like: a numeric literal (``return 0.0``), a dict literal with a
    ``"score"`` key, or a call to a ``*fail*/*zero*`` helper (``return
    _failure(...)``).
    """
    if (
        isinstance(value, ast.Constant)
        and isinstance(value.value, (int, float))
        and not isinstance(value.value, bool)
    ):
        return True
    if isinstance(value, ast.Dict):
        for key in value.keys:
            if isinstance(key, ast.Constant) and key.value == "score":
                return True
    if isinstance(value, ast.Call):
        func = value.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else ""
        )
        if _FAILURE_FUNC_RE.search(name):
            return True
    return False


def _handler_swallows_to_score(handler: ast.ExceptHandler) -> bool:
    """True if the handler converts the caught exception into a final score
    without raising. A handler that re-raises, or returns an intermediate
    sentinel (handled by ``_return_is_score_like``), is not flagged."""
    has_score_return = False
    has_raise = False
    for stmt in handler.body:
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Return)
                and node.value is not None
                and _return_is_score_like(node.value)
            ):
                has_score_return = True
            elif isinstance(node, ast.Raise):
                has_raise = True
    return has_score_return and not has_raise


def _guarded_node_ids(tree: ast.AST) -> set[int]:
    """ids of nodes lexically inside a ``try`` body whose handler set absorbs the
    directory/permission veto (so a planted dir/FIFO won't escape as OSError)."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(
            _handler_guards_dir_veto(handler) for handler in node.handlers
        ):
            for stmt in node.body:
                for descendant in ast.walk(stmt):
                    ids.add(id(descendant))
    return ids


def _pickle_exec_reader(call: ast.Call) -> str | None:
    """Return a label if the call deserializes pickle (RCE on load), else None."""
    func = call.func
    if isinstance(func, ast.Attribute):
        if func.attr in {"load", "loads"} and isinstance(func.value, ast.Name):
            module = func.value.id
            if module in _PICKLE_MODULE_NAMES:
                return f"{module}.{func.attr}"
        if func.attr == "read_pickle":
            return "read_pickle"
        if (
            func.attr == "load"
            and isinstance(func.value, ast.Name)
            and func.value.id in {"np", "numpy"}
        ):
            for kw in call.keywords:
                if (
                    kw.arg == "allow_pickle"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    return "numpy.load(allow_pickle=True)"
    return None


def _generic_reader_name(call: ast.Call) -> str | None:
    """Return a reader name if the call reads a file path (open / pandas readers /
    np.load / json.load / h5py.File), else None."""
    func = call.func
    if isinstance(func, ast.Name) and func.id in _GENERIC_AGENT_READER_NAMES:
        return func.id
    if isinstance(func, ast.Attribute):
        if func.attr in _GENERIC_AGENT_READER_ATTRS:
            return func.attr
        if func.attr in _AGENT_PATH_METHOD_READERS:
            return func.attr
        if isinstance(func.value, ast.Name):
            module = func.value.id
            if func.attr == "load" and module in {"np", "numpy", "json"}:
                return f"{module}.load"
            if func.attr == "File" and module in {"h5py"}:
                return "h5py.File"
    return None


def _reader_path_arg(call: ast.Call) -> ast.AST | None:
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr in _AGENT_PATH_METHOD_READERS:
        # ``path.read_text()`` / ``path.read_bytes()``: the path is the receiver.
        return func.value
    if call.args and not isinstance(call.args[0], ast.Starred):
        return call.args[0]
    for kw in call.keywords:
        if kw.arg in _SUBMISSION_PATH_KWARGS:
            return kw.value
    return None


def _is_write_open(call: ast.Call) -> bool:
    mode = None
    if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
        mode = call.args[1].value
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            mode = kw.value.value
    return isinstance(mode, str) and any(ch in mode for ch in "wax+")


def _path_segment_is_agent_writable(segment: str) -> bool:
    return any(token in segment for token in _AGENT_PATH_TOKENS)


def _agent_writable_names(tree: ast.AST, text: str) -> set[str]:
    """Variable names (transitively) assigned from an agent-writable path.

    Closes the alias gap: ``path = workspace / "submission.csv"`` followed by
    ``pd.read_csv(path)`` should still be recognized as an agent-path read.
    Module-wide fixpoint over assignments (precision over recall is acceptable;
    scorer files are small)."""
    names: set[str] = set()
    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    changed = True
    while changed:
        changed = False
        for assign in assigns:
            value = assign.value
            if value is None:
                continue
            targets = (
                assign.targets if isinstance(assign, ast.Assign) else [assign.target]
            )
            target_names = [t.id for t in targets if isinstance(t, ast.Name)]
            if not target_names:
                continue
            segment = ast.get_source_segment(text, value) or ""
            referenced = {n.id for n in ast.walk(value) if isinstance(n, ast.Name)}
            if _path_segment_is_agent_writable(segment) or (referenced & names):
                for name in target_names:
                    if name not in names:
                        names.add(name)
                        changed = True
    return names


def _agent_fault_issues(rel_path: str, text: str) -> list[str]:
    """Findings for the AgentFault keep-vs-discard discipline in one scorer file.

    Three classes: a broad except that returns a score (over-keep), a
    pickle-executing read of an agent artifact (RCE-as-root), and an
    agent-writable-path read that is unguarded or not signalled via AgentFault
    (over-discard).
    """
    issues: list[str] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return issues  # syntax errors surface in grader_import / grader_sandbox
    reachable = _reachable_from_compute_score(tree)
    if not reachable:
        return issues
    enclosing = _enclosing_func_name(tree)
    module_raises_af = _raises_agent_fault(tree, enclosing, reachable)
    guarded = _guarded_node_ids(tree)
    agent_names = _agent_writable_names(tree, text)

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Try)
            and _on_live_path(id(node), enclosing, reachable)
        ):
            for handler in node.handlers:
                if _handler_is_broad(handler) and _handler_swallows_to_score(handler):
                    issues.append(
                        f"{rel_path}:{handler.lineno}: a broad `except` returns a "
                        f"score instead of raising. This converts BOTH agent faults "
                        f"and author/infra bugs (missing truth, broken import) into a "
                        f"kept 0.0, which poisons RL training. Catch the specific "
                        f"agent-controlled exception and `raise "
                        f"grading.faults.AgentFault(...)`; let author/infra faults "
                        f"propagate so the runtime records env_internal_failure and "
                        f"discards the rollout."
                    )

        if not isinstance(node, ast.Call) or not _on_live_path(
            id(node), enclosing, reachable
        ):
            continue

        pickle_reader = _pickle_exec_reader(node)
        if pickle_reader is not None:
            issues.append(
                f"{rel_path}:{node.lineno}: `{pickle_reader}` deserializes pickle in "
                f"the root grader (arbitrary code execution via __reduce__ on a "
                f"submission the agent controls). Load agent artifacts through the "
                f"sandboxed grading.helpers.run_model_module(...) / run_policy(...), "
                f"or ship a non-pickle format."
            )
            continue

        reader_name = _generic_reader_name(node)
        if reader_name is None:
            continue
        if reader_name == "open" and _is_write_open(node):
            continue
        path_arg = _reader_path_arg(node)
        if path_arg is None:
            continue
        segment = ast.get_source_segment(text, path_arg) or ""
        is_agent_path = _path_segment_is_agent_writable(segment)
        if not is_agent_path and isinstance(path_arg, ast.Name):
            is_agent_path = path_arg.id in agent_names
        if not is_agent_path:
            continue
        if id(node) not in guarded:
            issues.append(
                f"{rel_path}:{node.lineno}: `{reader_name}` reads an agent-writable "
                f"path with no guard. A planted directory/FIFO at that path raises "
                f"OSError out of compute_score, so the runtime flags "
                f"env_internal_failure and DISCARDS the 0.0 the agent earned. "
                f"{_AGENT_FAULT_HELPER_HINT}"
            )
        elif not module_raises_af:
            issues.append(
                f"{rel_path}:{node.lineno}: `{reader_name}` reads an agent-writable "
                f"path behind a guard, but the scorer never raises "
                f"grading.faults.AgentFault on the grading path. Signal "
                f"agent-caused failures via the typed AgentFault so they are kept as "
                f"a real 0.0 for training, not a homebrew `return 0.0`. "
                f"{_AGENT_FAULT_HELPER_HINT}"
            )
    return issues


def _preloaded_files_issues(problem_dir: Path) -> list[str]:
    """Statically validate declared ``[[preloaded_files]]`` mounts.

    Trusted CI packs, uploads, and stamps mounts at submit time
    (``scripts/sync_mount.sh`` in the grade workflow), and ``ml`` tasks get their
    conventional dataset dirs auto-mounted, so authors no longer pre-commit
    ``.alignerr/preloaded_files.json``. Here we only check that each declaration
    is well-formed and that a local ``source`` actually exists, so a typo'd mount
    is caught in the sandbox instead of silently deploying without its dataset.
    """
    try:
        task_toml = load_task_toml(problem_dir)
    except Exception:
        return []  # schema stage reports parse errors
    declared = list(getattr(task_toml, "preloaded_files", []) or [])
    if not declared:
        return []

    issues: list[str] = []
    for preload in declared:
        source = (getattr(preload, "source", "") or "").strip()
        hf_repo = (getattr(preload, "hf_repo", "") or "").strip()
        if not source and not hf_repo:
            issues.append(
                f"[[preloaded_files]] mount {preload.mount_path!r} must set either "
                "'source' (a path under the problem dir) or 'hf_repo'."
            )
            continue
        if source and not (problem_dir / source).exists():
            issues.append(
                f"[[preloaded_files]] source {source!r} (mount "
                f"{preload.mount_path!r}) does not exist under the problem dir."
            )
    return issues


def _grade_workspace_score(
    workspace: Path,
    problem_dir: Path,
    compute_score: Any,
    params: list[str],
    normalize_fn: Any,
    rubric_builder_cls: Any,
    *,
    agent_fault_as_zero: bool = False,
) -> float | None:
    """Grade an arbitrary workspace through the real scorer; None on error.

    When ``agent_fault_as_zero`` is set, a submission the scorer rejects via
    ``AgentFault`` counts as a real ``0.0`` (a kept, in-bounds score) rather than
    being skipped -- so a baseline that correctly triggers AgentFault is still
    measured by the learnability gate instead of silently evading it.
    """
    if params[:3] != ["workspace", "trajectory", "private"]:
        return None
    try:
        raw = compute_score(workspace, None, problem_dir / "scorer" / "data")
        if isinstance(raw, rubric_builder_cls):
            raw = raw.grade().to_dict()
        grade = normalize_fn(raw)
        return grade.score()
    except Exception as exc:
        # Match AgentFault by name across the MRO so subclasses (and AgentFault
        # imported under any path) all count, not just the exact type.
        if agent_fault_as_zero and any(
            base.__name__ == "AgentFault" for base in type(exc).__mro__
        ):
            return 0.0
        return None


def _baseline_calibration_issues(
    problem_dir: Path,
    compute_score: Any,
    params: list[str],
    normalize_fn: Any,
    rubric_builder_cls: Any,
    *,
    target: float,
) -> list[str]:
    """Learnability gate for continuous-scoring tasks (FLOOR/REF/PERFECT).

    Each committed ``baselines/<name>/`` submission is graded through the real
    scorer; a naive baseline must land strictly below the reference ``target``
    (0.5). A baseline that reaches the reference means the task is not learnable
    (a constant/naive guess matches the expert) or the ``FLOOR`` anchor is set
    too low. A baseline the scorer rejects via AgentFault counts as ``0.0`` (a
    valid near-zero anchor). Baselines without a gradeable committed artifact are
    skipped (the harness / human review covers those); absence of baselines is a
    documented convention, not a hard failure.
    """
    import shutil
    import tempfile

    baselines_dir = problem_dir / "baselines"
    baseline_dirs = (
        [child for child in sorted(baselines_dir.iterdir()) if child.is_dir()]
        if baselines_dir.is_dir()
        else []
    )
    if not baseline_dirs:
        # Absence is a documented convention, not a hard failure: the no-op gate
        # (max_trivial_score) already anchors a trivial floor for every task.
        return []
    try:
        task_toml = load_task_toml(problem_dir)
    except Exception:
        return []
    output_names = [PurePosixPath(out.path).name for out in task_toml.outputs]
    if not output_names:
        return []

    issues: list[str] = []
    for bdir in baseline_dirs:
        staged: dict[str, Path] = {}
        for name in output_names:
            matches = [p for p in sorted(bdir.rglob(name)) if p.is_file()]
            if matches:
                staged[name] = matches[0]
        if not staged:
            continue
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td) / "ws"
            workspace.mkdir()
            for name, src in staged.items():
                shutil.copy(src, workspace / name)
            score = _grade_workspace_score(
                workspace,
                problem_dir,
                compute_score,
                params,
                normalize_fn,
                rubric_builder_cls,
                agent_fault_as_zero=True,
            )
        if score is None:
            continue
        if score >= target - 1e-9:
            issues.append(
                f"baseline {bdir.name!r} scores {score:.3f} through the real "
                f"grader, at or above the reference target {target:.2f}. A naive "
                f"baseline must score clearly below the reference: the task must be "
                f"learnable (a constant/naive guess cannot match the expert). "
                f"FLOOR must be the worst plausible raw metric (NOT this baseline's "
                f"measured value) -- set it below naive performance so the baseline "
                f"maps near 0 on its own; if it still reaches the reference, the task "
                f"is not learnable and needs a harder reference or a different metric."
            )
    return issues


def _extract_prompt_example_submission(
    problem_dir: Path,
) -> tuple[str, str] | None:
    """Return (output_filename, json_text) for the prompt's example, if unambiguous.

    Only fires when the task declares exactly one ``.json`` output and exactly
    one fenced code block in instruction.md parses as JSON, so we never guess.
    """
    try:
        task_toml = load_task_toml(problem_dir)
    except Exception:
        return None
    json_outputs = [
        out.path for out in task_toml.outputs if out.path.lower().endswith(".json")
    ]
    if len(json_outputs) != 1:
        return None
    filename = PurePosixPath(json_outputs[0]).name

    instruction = problem_dir / "instruction.md"
    if not instruction.exists():
        return None
    try:
        text = instruction.read_text()
    except OSError:
        return None
    blocks = re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    parsed: list[str] = []
    for block in blocks:
        candidate = block.strip()
        try:
            json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        parsed.append(candidate)
    if len(parsed) != 1:
        return None
    return filename, parsed[0]


def _trivial_submission_scores(
    problem_dir: Path,
    compute_score: Any,
    params: list[str],
    normalize_fn: Any,
    rubric_builder_cls: Any,
    meta: dict[str, Any],
) -> list[tuple[str, float | None]]:
    """Grade no-op and (when extractable) prompt-example submissions."""
    import tempfile

    results: list[tuple[str, float | None]] = []

    with tempfile.TemporaryDirectory() as td:
        noop_ws = Path(td) / "noop"
        noop_ws.mkdir()
        noop_score = _grade_workspace_score(
            noop_ws,
            problem_dir,
            compute_score,
            params,
            normalize_fn,
            rubric_builder_cls,
        )
        meta["noop_score"] = noop_score
        results.append(("a no-op (empty) submission", noop_score))

    example = _extract_prompt_example_submission(problem_dir)
    if example is not None:
        filename, json_text = example
        with tempfile.TemporaryDirectory() as td:
            example_ws = Path(td) / "example"
            example_ws.mkdir()
            (example_ws / filename).write_text(json_text)
            example_score = _grade_workspace_score(
                example_ws,
                problem_dir,
                compute_score,
                params,
                normalize_fn,
                rubric_builder_cls,
            )
        meta["example_score"] = example_score
        results.append(("the prompt's example submission", example_score))

    return results


_ORACLE_NAME_RE = re.compile(
    r"oracle|expected|reference|canonical|answer|ground_truth|hidden", re.IGNORECASE
)


def reward_hack_lint(problem_dir: Path) -> list[str]:
    """Advisory heuristics for gameable graders. Returns warnings (never fails).

    Flags three recurring reward-hacking smells: (1) an exact-match shortcut that
    short-circuits to full credit on equality with an oracle/reference value;
    (2) sentiment-gated text criteria that reward positive words or penalize an
    honest negative verdict; (3) keyword/substring-only text criteria that a
    model can satisfy by stuffing keywords without doing the analysis.
    """
    warnings: list[str] = []
    scorer_dir = problem_dir / "scorer"
    if not scorer_dir.is_dir():
        return warnings
    for source in sorted(scorer_dir.rglob("*.py")):
        if "__pycache__" in source.parts:
            continue
        try:
            text = source.read_text()
        except OSError:
            continue
        rel = source.relative_to(problem_dir).as_posix()
        warnings.extend(_reward_hack_warnings(rel, text))
    return warnings


def _reward_hack_warnings(rel_path: str, text: str) -> list[str]:
    warnings: list[str] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return warnings

    # (1) exact-match shortcut to full credit.
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.get_source_segment(text, node.test) or ""
        has_eq = any(
            isinstance(op, (ast.Eq, ast.Is))
            for cmp in ast.walk(node.test)
            if isinstance(cmp, ast.Compare)
            for op in cmp.ops
        )
        if (
            has_eq
            and _ORACLE_NAME_RE.search(test_src)
            and _branch_returns_full_credit(node)
        ):
            warnings.append(
                f"{rel_path}:{node.lineno}: looks like an exact-match shortcut that "
                f"awards full credit when the submission equals an oracle/reference "
                f"value. The oracle must reach 1.0 through the real numeric path; an "
                f"equality fast-path masks miscalibration and is unreachable by honest "
                f"solutions. Remove it or verify the oracle scores ~1.0 without it."
            )

    # (2) sentiment-gated criteria.
    string_consts = [
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    blob = " ".join(string_consts)
    positive_hits = {w for w in _SENTIMENT_POSITIVE_WORDS if w in blob}
    negative_hits = {w for w in _SENTIMENT_NEGATIVE_WORDS if w in blob}
    if negative_hits or len(positive_hits) >= 3:
        warnings.append(
            f"{rel_path}: grader text contains verdict-sentiment words "
            f"(positive={sorted(positive_hits)}, negative={sorted(negative_hits)}). "
            f"If a criterion rewards positive words or rejects 'unsafe/unacceptable', "
            f"it penalizes honest negative engineering judgment and is keyword-gameable. "
            f"Score conclusions on consistency with the numeric result, not sentiment."
        )

    # (3) keyword/substring-only membership checks.
    keyword_literals: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and any(
            isinstance(op, ast.In) for op in node.ops
        ):
            if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                literal = node.left.value.strip()
                if 0 < len(literal) <= 40:
                    keyword_literals.add(literal)
    if len(keyword_literals) >= 3:
        warnings.append(
            f"{rel_path}: {len(keyword_literals)} substring/keyword membership checks "
            f"against text. Keyword-only criteria are satisfiable by keyword stuffing; "
            f"tie credit to numeric results (e.g. require corrected values to be "
            f"near-correct before awarding defect/diagnosis credit)."
        )

    # (4) deprecated continuous-scoring curve. PiecewiseLinearCurve is the only
    # sanctioned curve (ReviewerInstructions 1.6.2); the exponential curve
    # compresses the sub-reference region and inflates near-perfect work.
    if "ExponentialCurve" in text or "solve_exponential_constants" in text or "exponential_score" in text:
        warnings.append(
            f"{rel_path}: uses the deprecated exponential calibration curve. "
            f"PiecewiseLinearCurve is the only sanctioned continuous-scoring curve; "
            f"the exponential curve compresses the sub-reference (stumped-agent) "
            f"region and inflates near-perfect scores. Switch to "
            f"grading.calibration.PiecewiseLinearCurve.from_reference(x_ref)."
        )
    return warnings


def _branch_returns_full_credit(node: ast.If) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float)):
            if float(sub.value) == 1.0:
                return True
    return False


def _validate_committed_ground_truth_result(
    problem_dir: Path,
    *,
    task_toml: Any,
    meta: dict[str, Any],
) -> list[str]:
    """Validate committed oracle proof for in-container ground-truth tasks.

    These tasks cannot be probed on the host because their grader invokes
    engines that exist only in the task image (OpenFOAM/SU2/Meep/OpenROAD/etc.).
    Instead, require the committed build proof produced by the local harness to
    contain a 1.0 `ground_truth_result` and, when a render is expected, matching
    committed reviewer artifacts.
    """
    issues: list[str] = []
    if task_toml is None:
        return ["could not load task.toml for in-container ground-truth validation"]

    proof_path = problem_dir / PROOF_PATH
    if not proof_path.exists():
        return [
            "in-container ground truth requires a committed .alignerr/build_proof.json "
            "with ground_truth_result; run `uv run lbx-rl-harness run --runtime "
            "ground-truth --problem-dir <task>`"
        ]

    try:
        proof = read_json(proof_path)
    except Exception as exc:  # noqa: BLE001
        return [f"could not read build proof for in-container ground truth: {exc}"]

    result = proof.get("ground_truth_result")
    if not isinstance(result, dict):
        return [
            "in-container ground truth proof is missing ground_truth_result; run "
            "`uv run lbx-rl-harness run --runtime ground-truth --problem-dir <task>`"
        ]

    try:
        score = float(result.get("score"))
    except (TypeError, ValueError):
        return ["ground_truth_result.score is missing or not numeric"]

    meta["ground_truth_score"] = score
    expectation = expected_ground_truth_score(
        task_toml.difficulty.reward_type,
        deterministic_epsilon=task_toml.ground_truth.score_epsilon,
        continuous_epsilon=task_toml.ground_truth.continuous_score_epsilon,
    )
    if not expectation.passed(score):
        issues.append(
            "ground truth solution for reward_type "
            f"{expectation.reward_type!r} must score {expectation.description}, "
            f"got {score:.6f}"
        )
    else:
        meta["ground_truth_passed"] = True

    # In-container tasks cannot be no-op-probed on the host. If the harness
    # recorded a trivial-baseline score in the proof, enforce the same
    # anti-reward-hacking ceiling here (no-op must not out-score real work).
    trivial_score = result.get("trivial_baseline_score")
    if isinstance(trivial_score, (int, float)):
        meta["noop_score"] = float(trivial_score)
        max_trivial = task_toml.ground_truth.max_trivial_score
        if float(trivial_score) > max_trivial + 1e-9:
            issues.append(
                f"a trivial/no-op submission scores {float(trivial_score):.3f}, "
                f"above the max_trivial_score ceiling ({max_trivial:.3f}). "
                f"Recalibrate so doing nothing scores ~0."
            )

    if render_expected(
        task_toml.difficulty.task_type, task_toml.ground_truth.render_outputs
    ):
        artifacts = _valid_committed_render_artifacts(problem_dir)
        if artifacts is None:
            issues.append(
                "in-container ground truth render artifacts are missing, stale, or do "
                "not match build_proof.json"
            )
        else:
            meta["review_artifacts"] = artifacts

    return issues


def _dockerfile_instructions(dockerfile: Path) -> list[tuple[int, str]]:
    instructions: list[tuple[int, str]] = []
    current = ""
    start_line = 0
    for line_number, raw_line in enumerate(dockerfile.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not current:
            start_line = line_number
        if line.endswith("\\"):
            current += line[:-1].strip() + " "
            continue
        current += line
        instructions.append((start_line, current.strip()))
        current = ""
    if current:
        instructions.append((start_line, current.strip()))
    return instructions


def _is_public_container_path(path: str) -> bool:
    normalized = path.rstrip("/") or "/"
    return any(
        normalized == root or normalized.startswith(root + "/")
        for root in _PUBLIC_ROOTS
    )


def _is_private_container_path(path: str) -> bool:
    normalized = path.rstrip("/") or "/"
    return any(
        normalized == root or normalized.startswith(root + "/")
        for root in _PRIVATE_ROOTS
    )


# Tokens whose presence anywhere in a reference-solution file means the
# reference reaches for the held-out answer key. The container forms reuse
# `_PRIVATE_ROOTS`; `_PRIVATE_DISK_ROOT` is the on-disk source-tree form. The
# public `/data/` mount is deliberately absent: it is the agent's legitimate
# input and is not a substring of any token here, so it never false-positives.
_PRIVATE_READ_TOKENS = (*_PRIVATE_ROOTS, _PRIVATE_DISK_ROOT)

_PRIVATE_READ_HINT = (
    "reference solution must not read the private answer key "
    "(scorer/data / /mcp_server/data / /mcp_server/grader); it must solve the "
    "task, not read held-out truth"
)


def _private_token_in(text: str) -> str | None:
    """Return the first private-root token found in ``text``, else ``None``."""
    for token in _PRIVATE_READ_TOKENS:
        if token in text:
            return token
    return None


def _solution_shell_private_read_issues(rel_path: str, text: str) -> list[str]:
    """Flag any line of a reference shell script that names a private root."""
    issues: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        token = _private_token_in(line)
        if token is not None:
            issues.append(f"{rel_path}:{lineno}: {_PRIVATE_READ_HINT} (matched {token!r})")
    return issues


def _solution_python_private_read_issues(rel_path: str, text: str) -> list[str]:
    """Flag string literals in a reference *.py that name a private root.

    Mirrors the AST-with-line-scan-fallback style used elsewhere in this module
    (e.g. `_grader_sandbox_issues`): on a syntax error we degrade to a raw line
    scan so an unparseable reference cannot smuggle the answer-key path through.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        issues: list[str] = []
        for lineno, line in enumerate(text.splitlines(), 1):
            token = _private_token_in(line)
            if token is not None:
                issues.append(
                    f"{rel_path}:{lineno}: {_PRIVATE_READ_HINT} (matched {token!r})"
                )
        return issues

    issues = []
    seen: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        token = _private_token_in(node.value)
        if token is None:
            continue
        lineno = getattr(node, "lineno", 0)
        if lineno in seen:
            continue
        seen.add(lineno)
        issues.append(f"{rel_path}:{lineno}: {_PRIVATE_READ_HINT} (matched {token!r})")
    return issues


def _solution_private_read_issues(problem_dir: Path) -> list[str]:
    """Static check: the reference solution must not read the private answer key.

    Scans ``solution/*.sh`` (line/substring) and ``solution/*.py`` (string
    literals via AST) for references to the private held-out truth, under either
    its on-disk name (``scorer/data``) or its baked container roots
    (``/mcp_server/data``, ``/mcp_server/grader``). A reference that reads the
    answer key would score perfectly and miscalibrate the reference/0.5 anchor.

    No-ops (returns ``[]``) when ``solution/`` is absent, e.g. a task type that
    ships no reference.
    """
    solution_dir = problem_dir / "solution"
    if not solution_dir.is_dir():
        return []
    issues: list[str] = []
    for path in sorted(solution_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(problem_dir).as_posix()
        if path.suffix == ".sh":
            scanner = _solution_shell_private_read_issues
        elif path.suffix == ".py":
            scanner = _solution_python_private_read_issues
        else:
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        issues.extend(scanner(rel, text))
    return issues


def _has_private_source(source: str) -> bool:
    lower = source.lower()
    normalized = lower.replace("\\", "/").strip("/")
    if any(
        normalized == marker.strip("/")
        or normalized.startswith(marker.strip("/") + "/")
        for marker in _PRIVATE_DOCKER_SOURCES
        if marker.startswith("/")
    ):
        return True
    return "scorer" in {part for part in normalized.split("/") if part}


def _dockerfile_private_layout_issues(dockerfile: Path) -> list[str]:
    issues: list[str] = []
    # Private roots that received a COPY/ADD which is not self-hardened via a
    # safe `--chmod`, mapped to the line for diagnostics. Each must be locked
    # down by a later RUN (root-owned, no group/world bits).
    needs_hardening: dict[str, int] = {}
    hardened_roots: set[str] = set()
    for line_number, instruction in _dockerfile_instructions(dockerfile):
        lower = instruction.lower()
        if lower.startswith(("copy ", "add ")):
            try:
                tokens = shlex.split(instruction)
            except ValueError:
                tokens = instruction.split()
            flags, positional = _dockerfile_copy_parts(tokens)
            if len(positional) >= 2:
                sources = positional[:-1]
                destination = positional[-1]
                if _is_public_container_path(destination) and any(
                    _has_private_source(source) for source in sources
                ):
                    issues.append(
                        f"{dockerfile}:{line_number}: private scorer fixtures must "
                        f"not be copied to public/model-writable path {destination!r}"
                    )
                if _is_private_container_path(destination):
                    issues.extend(
                        _copy_private_layout_flag_issues(dockerfile, line_number, flags)
                    )
                    root = _matching_private_root(destination)
                    chmod_flag = flags.get("chmod")
                    self_hardened = (
                        chmod_flag is not None
                        and not _mode_grants_group_world(chmod_flag)
                    )
                    if root is not None and not self_hardened:
                        needs_hardening.setdefault(root, line_number)
        if lower.startswith("run "):
            run_body = lower.removeprefix("run ").strip()
            if "chmod" in lower:
                issues.extend(
                    _chmod_private_layout_issues(dockerfile, line_number, run_body)
                )
            if "chown" in lower:
                issues.extend(
                    _chown_private_layout_issues(dockerfile, line_number, run_body)
                )
            hardened_roots |= _private_roots_hardened_by_run(run_body)
    for root in sorted(set(needs_hardening) - hardened_roots):
        issues.append(
            f"{dockerfile}:{needs_hardening[root]}: scorer fixtures copied into "
            f"{root} are left group/world-readable. Lock them down so the agent "
            f"cannot read the answer key: add a RUN that makes them root-owned "
            f"0700/0600 (e.g. `chown -R root:root {root} && find {root} -type d "
            f"-exec chmod 0700 {{}} + && find {root} -type f -exec chmod 0600 {{}} +`) "
            f"or copy with `--chmod=0600`."
        )
    return issues


def _matching_private_root(destination: str) -> str | None:
    normalized = destination.rstrip("/") or "/"
    for root in _PRIVATE_ROOTS:
        if normalized == root or normalized.startswith(root + "/"):
            return root
    return None


def _private_roots_hardened_by_run(instruction_lower: str) -> set[str]:
    """Return private roots locked down by a RUN with a restrictive chmod.

    A RUN hardens a private root when it both references that root and applies a
    chmod whose mode grants no group/world bits (covers `chmod 0700/0600`,
    `chmod -R go-rwx`, and the canonical `find <root> -exec chmod 0600 {} +`).
    """
    has_restrictive_chmod = False
    for segment in instruction_lower.replace("&&", ";").split(";"):
        parts = segment.strip().split()
        for index, token in enumerate(parts):
            if token != "chmod":
                continue
            rest = parts[index + 1 :]
            while rest and rest[0].startswith("-"):
                rest = rest[1:]
            if rest and not _mode_grants_group_world(rest[0]):
                has_restrictive_chmod = True
    if not has_restrictive_chmod:
        return set()
    return {root for root in _PRIVATE_ROOTS if root in instruction_lower}


def _dockerfile_copy_parts(tokens: list[str]) -> tuple[dict[str, str], list[str]]:
    flags: dict[str, str] = {}
    positional: list[str] = []
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--"):
            raw_flag = token[2:]
            if "=" in raw_flag:
                name, value = raw_flag.split("=", 1)
            elif raw_flag.lower() in _COPY_FLAGS_WITH_VALUE and index + 1 < len(tokens):
                name, value = raw_flag, tokens[index + 1]
                index += 1
            else:
                name, value = raw_flag, "true"
            flags[name.lower()] = value
        else:
            positional.append(token)
        index += 1
    if (
        len(positional) >= 2
        and positional[0].startswith("[")
        and positional[-1].endswith("]")
    ):
        positional = _dockerfile_json_array_parts(positional)
    return flags, positional


def _dockerfile_json_array_parts(tokens: list[str]) -> list[str]:
    raw = " ".join(tokens).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        inner = raw.removeprefix("[").removesuffix("]")
        return [part.strip().strip("\"'") for part in inner.split(",") if part.strip()]
    if not isinstance(parsed, list):
        return tokens
    return [item for item in parsed if isinstance(item, str)]


def _copy_private_layout_flag_issues(
    dockerfile: Path, line_number: int, flags: dict[str, str]
) -> list[str]:
    issues: list[str] = []
    owner = flags.get("chown")
    if owner is not None and _owner_assigns_nonroot_user(owner):
        issues.append(
            f"{dockerfile}:{line_number}: private grader/data COPY --chown must "
            "keep root as the owning user"
        )
    mode_token = flags.get("chmod")
    if mode_token is not None and _mode_grants_group_world(mode_token):
        issues.append(
            f"{dockerfile}:{line_number}: private grader/data COPY --chmod "
            f"{mode_token!r} grants group/world access"
        )
    return issues


def _chmod_private_layout_issues(
    dockerfile: Path, line_number: int, instruction: str
) -> list[str]:
    issues: list[str] = []
    segments = instruction.replace("&&", ";").split(";")
    for segment in segments:
        parts = segment.strip().split()
        if not parts or parts[0] != "chmod":
            continue
        rest = parts[1:]
        while rest and rest[0].startswith("-"):
            rest = rest[1:]
        if len(rest) < 2:
            continue
        mode_token = rest[0]
        targets = rest[1:]
        if _targets_include_private_root(targets):
            if _mode_grants_group_world(mode_token):
                issues.append(
                    f"{dockerfile}:{line_number}: private grader/data chmod "
                    f"{mode_token!r} grants group/world access"
                )
    return issues


def _chown_private_layout_issues(
    dockerfile: Path, line_number: int, instruction: str
) -> list[str]:
    issues: list[str] = []
    segments = instruction.replace("&&", ";").split(";")
    for segment in segments:
        parts = segment.strip().split()
        if not parts or parts[0] != "chown":
            continue
        rest = parts[1:]
        while rest and rest[0].startswith("-"):
            rest = rest[1:]
        if len(rest) < 2:
            continue
        owner = rest[0]
        targets = rest[1:]
        if not _owner_assigns_nonroot_user(owner):
            continue
        if _targets_include_private_root(targets):
            issues.append(
                f"{dockerfile}:{line_number}: private grader/data paths must "
                "remain root-owned"
            )
    return issues


def _targets_include_private_root(targets: list[str]) -> bool:
    return any(
        _is_private_container_path(_clean_dockerfile_path(target)) for target in targets
    )


def _clean_dockerfile_path(path: str) -> str:
    return path.strip().strip("\"'").rstrip(",")


def _mode_grants_group_world(mode_token: str) -> bool:
    try:
        return bool(int(mode_token, 8) & 0o077)
    except ValueError:
        pass
    for clause in mode_token.lower().split(","):
        operator_index = next(
            (index for index, char in enumerate(clause) if char in "+="), None
        )
        if operator_index is None:
            continue
        who = clause[:operator_index] or "a"
        perms = clause[operator_index + 1 :]
        if perms and any(entity in who for entity in ("a", "g", "o")):
            return True
    return False


def _owner_assigns_nonroot_user(owner: str) -> bool:
    user = owner.split(":", 1)[0].split(".", 1)[0].lower()
    return bool(user) and user not in {"0", "root"}


def _public_private_duplicate_issues(problem_dir: Path) -> list[str]:
    public_dir = problem_dir / "data"
    private_dir = problem_dir / "scorer" / "data"
    if not public_dir.exists() or not private_dir.exists():
        return []

    public_by_digest: dict[bytes, Path] = {}
    for public_file in public_dir.rglob("*"):
        if public_file.is_file() and public_file.stat().st_size:
            public_by_digest[_file_digest(public_file)] = public_file

    issues: list[str] = []
    for private_file in private_dir.rglob("*"):
        if not private_file.is_file() or not private_file.stat().st_size:
            continue
        if not _is_sensitive_private_name(private_file):
            continue
        public_match = public_by_digest.get(_file_digest(private_file))
        if public_match is not None:
            issues.append(
                "sensitive private fixture duplicates public data byte-for-byte: "
                f"{private_file.relative_to(problem_dir)} == "
                f"{public_match.relative_to(problem_dir)}"
            )
    return issues


def _file_digest(path: Path) -> bytes:
    import hashlib

    return hashlib.sha256(path.read_bytes()).digest()


def _is_sensitive_private_name(path: Path) -> bool:
    return any(
        token in _SENSITIVE_PRIVATE_NAME_PARTS for token in _name_tokens(path.name)
    )


def _name_tokens(name: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for char in name.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _run_private_layout_image_probe(image_tag: str) -> str | None:
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "root",
            "--entrypoint",
            "python",
            image_tag,
            "-c",
            _PRIVATE_LAYOUT_PROBE,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode == 0:
        return None
    detail = (completed.stderr or completed.stdout).strip()
    if detail:
        return f"private data layout image probe failed: {detail}"
    return f"private data layout image probe failed with exit {completed.returncode}"


def _run_agent_python_image_probe(image_tag: str) -> str | None:
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "1000:1000",
            "--entrypoint",
            "/bin/sh",
            image_tag,
            "-lc",
            _AGENT_PYTHON_PROBE,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode == 0:
        return None
    detail = (completed.stderr or completed.stdout).strip()
    if detail:
        return f"agent python image probe failed: {detail}"
    return f"agent python image probe failed with exit {completed.returncode}"


def _valid_committed_render_artifacts(problem_dir: Path) -> list[str] | None:
    proof_path = problem_dir / PROOF_PATH
    if not proof_path.exists():
        return None
    try:
        proof = read_json(proof_path)
    except Exception:
        return None
    result = proof.get("ground_truth_result")
    if not isinstance(result, dict):
        return None
    artifacts = result.get("review_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return None
    logical_paths: list[str] = []
    for artifact in artifacts:
        if not artifact_matches_proof(problem_dir, artifact):
            return None
        if isinstance(artifact, dict) and isinstance(artifact.get("logical_path"), str):
            logical_paths.append(artifact["logical_path"])
    return logical_paths


def _run_ground_truth_render_probe(problem_dir: Path, workspace: Path) -> list[str]:
    task_toml = load_task_toml(problem_dir)
    ground_truth = task_toml.ground_truth
    if not ground_truth.render_command.strip() or not ground_truth.render_outputs:
        raise RuntimeError("ground truth render command and outputs are required")

    env = os.environ.copy()
    env["LBT_OUTPUT_DIR"] = str(workspace)
    completed = subprocess.run(
        ["bash", "-lc", ground_truth.render_command],
        cwd=problem_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=task_toml.agent.timeout_sec or 3600,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "ground truth render command exited with status "
            f"{completed.returncode}: {completed.stderr.strip()[:200]}"
        )

    artifacts: list[str] = []
    for output in ground_truth.render_outputs:
        rel = _output_relative_path(output.path)
        if rel is None:
            raise RuntimeError(
                f"ground truth render output must be under /tmp/output: {output.path}"
            )
        validate_video_file(workspace / rel, output.path)
        artifacts.append(output.path)
    return artifacts
