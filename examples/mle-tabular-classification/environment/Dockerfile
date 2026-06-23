ARG BASE_IMAGE=lbx-tasks-base
ARG BASE_TAG=runtime-ml-core-py313-local
ARG PROBLEM_DIR=examples/mle-tabular-classification
FROM ${BASE_IMAGE}:${BASE_TAG}

ARG PROBLEM_DIR=examples/mle-tabular-classification

WORKDIR /workdir

RUN mkdir -p /mcp_server/data /workdir /tmp/output && chown -R 1000:1000 /workdir /tmp/output
RUN env -u UV_SYSTEM_PYTHON uv pip install --python /mcp_server/.venv/bin/python --no-cache numpy pandas pyarrow scikit-learn
COPY ${PROBLEM_DIR}/data/ /data/
COPY --chmod=0700 ${PROBLEM_DIR}/scorer/data/ /mcp_server/data/
COPY --chmod=0700 ${PROBLEM_DIR}/scorer/ /mcp_server/grader/
RUN chmod 0755 /mcp_server && chmod -R 0700 /mcp_server/data /mcp_server/grader

COPY ${PROBLEM_DIR}/task.toml ${PROBLEM_DIR}/instruction.md /task/
