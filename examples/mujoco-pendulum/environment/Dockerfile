ARG BASE_IMAGE=lbx-tasks-base
ARG BASE_TAG=runtime-ml-core-py313-local
ARG PROBLEM_DIR=examples/mujoco-pendulum
FROM ${BASE_IMAGE}:${BASE_TAG}

ARG PROBLEM_DIR=examples/mujoco-pendulum

WORKDIR /workdir

RUN mkdir -p /mcp_server/data /workdir /tmp/output && chown -R 1000:1000 /workdir /tmp/output
COPY ${PROBLEM_DIR}/data/ /data/
COPY --chmod=0700 ${PROBLEM_DIR}/scorer/data/ /mcp_server/data/
COPY --chmod=0700 ${PROBLEM_DIR}/scorer/ /mcp_server/grader/
RUN chmod 0755 /mcp_server && chmod -R 0700 /mcp_server/data /mcp_server/grader

COPY ${PROBLEM_DIR}/task.toml ${PROBLEM_DIR}/instruction.md /task/
