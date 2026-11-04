#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv-gpu
.venv-gpu/bin/pip install -q uv
.venv-gpu/bin/uv pip install --quiet --python .venv-gpu/bin/python \
  'jax[cuda13]==0.12.0.1' 'flax==0.12.12' 'netket==3.23.0' \
  'numpy==2.5.4' 'safetensors==0.8.0' 'msgpack==1.2.4' 'einops==0.8.2'
mkdir -p runs
.venv-gpu/bin/uv pip freeze --python .venv-gpu/bin/python > runs/gpu_environment.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
env -u LD_LIBRARY_PATH .venv-gpu/bin/python -c 'import jax; print(jax.devices()); assert jax.default_backend() == "gpu"'
