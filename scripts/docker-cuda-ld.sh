#!/bin/sh
# TorchCodec CUDA wheels dlopen NVIDIA libs shipped in pip wheels under
# site-packages/nvidia/*/lib (see https://github.com/pytorch/torchcodec#installing-cuda-enabled-torchcodec).
# python:3.11-slim has no system CUDA; prepend all those dirs plus torch/lib.
set -eu
FW_VENV_ROOT="${FW_VENV_ROOT:-/app/.venv}"
SP="$FW_VENV_ROOT/lib/python3.11/site-packages"
LD_LIBRARY_PATH="$SP/torch/lib"
if [ -d "$SP/nvidia" ]; then
  nvidia_ld="$(find "$SP/nvidia" -type d -name lib 2>/dev/null | tr '\n' ':' | sed 's/:$//')"
  if [ -n "$nvidia_ld" ]; then
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${nvidia_ld}"
  fi
fi
export LD_LIBRARY_PATH
exec "$@"
