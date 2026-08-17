#!/usr/bin/env bash
set -euo pipefail

uv sync --extra dev
uv run ruff check .
uv run mypy src/ph_hgnn
uv run pytest
uv run ph-hgnn --quick
uv run ph-hgnn \
  --config configs/server_single_gpu.yaml \
  --dataset local_global_witnesses \
  --quick
