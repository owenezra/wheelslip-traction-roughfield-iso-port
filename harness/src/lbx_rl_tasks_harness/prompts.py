from __future__ import annotations

LOCAL_SYSTEM_PROMPT = (
    "You are an autonomous agent inside a sandboxed Linux task container. "
    "Solve the task described in the user message by exploring available files, "
    "writing code, and producing the required output artifacts under /tmp/output. "
    "Use bash for short shell commands, str_replace_based_edit_tool for file edits, "
    "and tmux for long-running processes. "
    "For numerical-solver tasks, use the real solver stack when the task asks for "
    "solver-backed evidence or diagnostics. OpenFOAM commands are available from "
    "Bash after activation, e.g. `bash -lc 'source /etc/solver-envs.d/openfoam.sh "
    "&& blockMesh -help'`; then tools like `blockMesh`, `checkMesh`, `simpleFoam`, "
    "and `pimpleFoam` can be invoked. "
    "OpenSees tasks use OpenSeesPy through Python, for example "
    "`import openseespy.opensees as ops`; do not assume a native `OpenSees` CLI exists. "
    "For structural calibration, modal, drift, damping, frame, or retrofit tasks, "
    "run an OpenSeesPy import or small check model before the final answer when "
    "OpenSeesPy is available, and use that result to validate the submitted parameters."
)

TASK_TYPE_PREPROMPT_MARKER = "<!-- lbx-task-type-preprompt -->"

_STRUCTURES_PREPROMPT = """\
## Required OpenSeesPy Use

This is a `structures` numerical-solver task. You must run OpenSeesPy as part
of solving the task, not only reason from static files.

Your work should clearly show agent-side OpenSeesPy evidence, such as:

- importing `openseespy.opensees`;
- building a model with calls like `ops.model`, `ops.node`, `ops.element`, and
  related setup calls;
- running analysis with `ops.analyze` or modal analysis with `ops.eigen`;
- inspecting numerical outputs such as convergence status, return code, drift,
  displacement, period, eigenvalue, or base shear.

Use the OpenSeesPy result to check or refine your final `/tmp/output` artifact.
If the task asks for a solver-evidence artifact, write it exactly at the
requested path and include the command, return code, and relevant output
summary. If no evidence artifact is requested, your trajectory should still
show the OpenSeesPy code you ran and the numerical output you inspected.
"""

_CFD_PREPROMPT = """\
## Required OpenFOAM Use

This is a `cfd` numerical-solver task. You must run OpenFOAM as part of solving
the task, not only reason from static files.

Your work should clearly show agent-side OpenFOAM evidence, such as:

- sourcing the OpenFOAM environment, for example
  `source /etc/solver-envs.d/openfoam.sh`;
- running commands such as `blockMesh`, `checkMesh`, `snappyHexMesh`,
  `simpleFoam`, `pimpleFoam`, `potentialFoam`, or `foamRun`;
- inspecting command output such as mesh quality checks, `Mesh OK`,
  geometry/topology checks, solver residuals, Courant number, time steps, or
  execution time.

Use the OpenFOAM result to check or refine your final `/tmp/output` artifact.
If the task asks for a solver-evidence artifact, write it exactly at the
requested path and include the command, return code, and relevant output
summary. If no evidence artifact is requested, your trajectory should still
show the OpenFOAM command you ran and the numerical output you inspected.
"""


def task_type_preprompt(task_type: str) -> str:
    normalized = task_type.strip().lower()
    if normalized == "structures":
        return _STRUCTURES_PREPROMPT
    if normalized == "cfd":
        return _CFD_PREPROMPT
    return ""


def prompt_with_task_type_preprompt(prompt: str, task_type: str) -> str:
    preprompt = task_type_preprompt(task_type)
    clean_prompt = prompt
    if clean_prompt.startswith(TASK_TYPE_PREPROMPT_MARKER):
        clean_prompt = clean_prompt.removeprefix(TASK_TYPE_PREPROMPT_MARKER).lstrip("\n")
    if not preprompt or clean_prompt.startswith(preprompt.rstrip()):
        return clean_prompt
    return f"{preprompt.rstrip()}\n\n---\n\n{clean_prompt}"
