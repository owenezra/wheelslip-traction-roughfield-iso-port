"""Local Docker runtime image helpers for template self-contained builds."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from alignerr_plugin.utils import load_task_toml

LOCAL_CPU_BASE_IMAGE = "lbx-tasks-base"
LOCAL_GPU_BASE_IMAGE = "lbx-tasks-base-gpu"
LOCAL_GPU_OPENROAD_BASE_IMAGE = "lbx-tasks-base-gpu-openroad"
LOCAL_GPU_BLACKWELL_BASE_IMAGE = "lbx-tasks-base-gpu-blackwell"
LOCAL_CUDA_GRAPHICS_BASE_IMAGE = "lbx-tasks-base-cuda-graphics"
LOCAL_TPU_BASE_IMAGE = "lbx-tasks-base-tpu"
LOCAL_BASE_TAG = "runtime-ml-core-py313-local"
LOCAL_PLATFORM = "linux/amd64"

# Per-flavor (local image name, Dockerfile path).
_LOCAL_BASE_BY_FLAVOR: dict[str, tuple[str, str]] = {
    "cpu": (LOCAL_CPU_BASE_IMAGE, "base/cpu/Dockerfile"),
    "gpu": (LOCAL_GPU_BASE_IMAGE, "base/gpu/Dockerfile"),
    "gpu-openroad": (LOCAL_GPU_OPENROAD_BASE_IMAGE, "base/gpu-openroad/Dockerfile"),
    "gpu-blackwell": (
        LOCAL_GPU_BLACKWELL_BASE_IMAGE,
        "base/gpu-blackwell/Dockerfile",
    ),
    "cuda-graphics": (
        LOCAL_CUDA_GRAPHICS_BASE_IMAGE,
        "base/cuda-graphics/Dockerfile",
    ),
    "tpu": (LOCAL_TPU_BASE_IMAGE, "base/tpu/Dockerfile"),
}


@dataclass(frozen=True)
class LocalBaseImage:
    image: str
    tag: str
    dockerfile: Path

    @property
    def ref(self) -> str:
        return f"{self.image}:{self.tag}"


def local_base_image_for_problem(problem_dir: Path) -> LocalBaseImage:
    """Return the repo-local base image required by a task (by base flavor)."""
    from alignerr_plugin.base_image import resolve_base_flavor_for_resource

    task_toml = load_task_toml(problem_dir)
    env = task_toml.environment
    flavor = resolve_base_flavor_for_resource(
        getattr(env, "base_flavor", "auto"), env.required_resources
    )
    image, dockerfile = _LOCAL_BASE_BY_FLAVOR[flavor]
    return LocalBaseImage(image=image, tag=LOCAL_BASE_TAG, dockerfile=Path(dockerfile))


def ensure_local_base_image(repo_root: Path, problem_dir: Path) -> LocalBaseImage:
    """Build the template's local base image when it is not already present."""
    if shutil.which("docker") is None:
        raise RuntimeError("docker is required for local harness runs")

    base = local_base_image_for_problem(problem_dir)
    if _docker_image_exists(base.ref):
        return base
    _ensure_parent_base_image(repo_root, base)

    dockerfile = repo_root / base.dockerfile
    if not dockerfile.exists():
        raise FileNotFoundError(f"local base Dockerfile not found: {dockerfile}")

    print(f"Building local base image {base.ref} from {dockerfile}...", flush=True)
    completed = subprocess.run(
        [
            "docker",
            "build",
            "--progress",
            "plain",
            "--platform",
            LOCAL_PLATFORM,
            "--file",
            str(dockerfile),
            "--tag",
            base.ref,
            str(repo_root),
        ],
        check=False,
        timeout=3600,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"local base image build failed for {base.ref}. "
            "See the Docker output above for the failing install-common phase. "
            "If the failure happened while downloading or exporting packages, "
            "check Docker disk usage with `docker system df`."
        )
    return base


def _ensure_parent_base_image(repo_root: Path, base: LocalBaseImage) -> None:
    """Build a local parent base before overlay flavors that FROM it."""
    from alignerr_plugin.base_image import BASE_FLAVORS

    flavor = next(
        (
            name
            for name, (image, dockerfile) in _LOCAL_BASE_BY_FLAVOR.items()
            if image == base.image and Path(dockerfile) == base.dockerfile
        ),
        None,
    )
    if flavor is None:
        return
    parent = BASE_FLAVORS[flavor].parent
    if not parent:
        return
    parent_image, parent_dockerfile = _LOCAL_BASE_BY_FLAVOR[parent]
    parent_base = LocalBaseImage(
        image=parent_image, tag=base.tag, dockerfile=Path(parent_dockerfile)
    )
    if _docker_image_exists(parent_base.ref):
        return

    dockerfile = repo_root / parent_base.dockerfile
    if not dockerfile.exists():
        raise FileNotFoundError(f"local parent base Dockerfile not found: {dockerfile}")

    print(
        f"Building local parent base image {parent_base.ref} from {dockerfile}...",
        flush=True,
    )
    completed = subprocess.run(
        [
            "docker",
            "build",
            "--progress",
            "plain",
            "--platform",
            LOCAL_PLATFORM,
            "--file",
            str(dockerfile),
            "--tag",
            parent_base.ref,
            str(repo_root),
        ],
        check=False,
        timeout=3600,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"local parent base image build failed for {parent_base.ref}"
        )


def _docker_image_exists(image_ref: str) -> bool:
    completed = subprocess.run(
        ["docker", "image", "inspect", image_ref],
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )
    return completed.returncode == 0
