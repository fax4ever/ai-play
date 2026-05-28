#!/usr/bin/env bash

export PATH="$HOME/openmpi4/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/openmpi4/lib:/opt/rocm-7.2.2/lib:${LD_LIBRARY_PATH:-}"
export OPAL_PREFIX="$HOME/openmpi4"

exec "/home/fax/code/vllm/.venv/bin/python" "$@"
