from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from alignerr_plugin.local_runtime import ensure_local_base_image
from alignerr_plugin.proof import write_build_proof

from lbx_rl_tasks_harness.models import HarnessProblem

TAIGA_PLATFORM = "linux/amd64"


@dataclass(frozen=True)
class StartedContainer:
    image_tag: str
    container_id: str


def _docker(
    args: list[str], *, timeout: int = 1200
) -> subprocess.CompletedProcess[str]:
    if shutil.which("docker") is None:
        raise RuntimeError("docker is required for agent harness runtimes")
    return subprocess.run(
        ["docker", *args],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=True,
    )


def _repo_root(problem_dir: Path) -> Path:
    for parent in [problem_dir, *problem_dir.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "grader").exists():
            return parent
    raise RuntimeError(f"could not locate repo root for {problem_dir}")


def build_task_image(problem: HarnessProblem, *, write_proof: bool = True) -> str:
    if problem.source_problem_dir is None:
        raise RuntimeError(
            "agent harness runtimes require --source-problem-dir for exported formats"
        )
    problem_dir = problem.source_problem_dir.resolve()
    repo_root = _repo_root(problem_dir)
    dockerfile = problem_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        raise FileNotFoundError(f"task Dockerfile not found: {dockerfile}")

    base = ensure_local_base_image(repo_root, problem_dir)
    rel_problem_dir = problem_dir.relative_to(repo_root)
    safe_id = problem.id.replace("/", "-").replace("_", "-")
    image_tag = f"lbx-rl-harness-{safe_id}:{int(time.time())}"
    iidfile = problem_dir / ".alignerr" / "image.iid"
    iidfile.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    _docker(
        [
            "build",
            "--platform",
            TAIGA_PLATFORM,
            "--file",
            str(dockerfile),
            "--build-arg",
            f"BASE_IMAGE={base.image}",
            "--build-arg",
            f"BASE_TAG={base.tag}",
            "--build-arg",
            f"PROBLEM_DIR={rel_problem_dir}",
            "--tag",
            image_tag,
            "--iidfile",
            str(iidfile),
            ".",
        ],
        timeout=3600,
    )
    image_digest = iidfile.read_text().strip() if iidfile.exists() else image_tag
    if write_proof:
        write_build_proof(
            problem_dir,
            image_digest=image_digest,
            base_image_ref=base.ref,
            platform=TAIGA_PLATFORM,
            alignerr_cli_version="0.1.0",
            duration_seconds=time.monotonic() - start,
        )
    return image_tag


def start_task_container(image_tag: str) -> StartedContainer:
    args = [
        "run",
        "--platform",
        TAIGA_PLATFORM,
        "-d",
        "--rm",
        "-e",
        "ANTHROPIC_API_KEY",
        "-e",
        "IS_SANDBOX=yes",
        image_tag,
        "sleep",
        "infinity",
    ]
    proc = _docker(args, timeout=120)
    cid = proc.stdout.strip()
    if not cid:
        raise RuntimeError(f"docker run returned no container id for {image_tag}")
    patch_rubric_stdout_noise(cid)
    return StartedContainer(image_tag=image_tag, container_id=cid)


def patch_rubric_stdout_noise(cid: str) -> None:
    """Keep rubric grading prints off MCP stdout.

    The current image-baked rubric mirrors scorer subprocess stdout to its own
    stdout. That corrupts MCP stdio because lines like `RUBRIC_RESULT_JSON=...`
    are not JSON-RPC messages. Local harness runs patch the server to mirror
    those scorer lines to stderr instead.
    """
    script = r"""
from pathlib import Path

path = Path("/mcp_server/src/rubric/server.py")
if not path.exists():
    raise SystemExit(0)
text = path.read_text()
text = text.replace("sys.stdout.write(proc.stdout)", "sys.stderr.write(proc.stdout)")
path.write_text(text)
"""
    subprocess.run(
        ["docker", "exec", cid, "python", "-c", script],
        text=True,
        capture_output=True,
        check=False,
    )


def stop_task_container(cid: str) -> None:
    if shutil.which("docker") is None:
        return
    subprocess.run(
        ["docker", "rm", "-f", cid], text=True, capture_output=True, check=False
    )


def copy_output_from_container(cid: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["docker", "cp", f"{cid}:/tmp/output/.", str(destination)],
        text=True,
        capture_output=True,
        check=False,
    )
