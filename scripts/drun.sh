#!/usr/bin/env bash
# Run any command inside the omnigda-manip:jazzy GPU + ROS2 + uv container
# Preserves current working directory (supports main repo AND any git worktree!)
set -euo pipefail

CACHE_DIR="${HOME}/.cache/omnigda"
mkdir -p "${CACHE_DIR}/huggingface" "${CACHE_DIR}/torch" "${CACHE_DIR}/maniskill"

PROXY_URL="${OMNIGDA_PROXY:-http://127.0.0.1:7897}"

exec docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e PYTHONPATH="/opt/omnigda-venv/lib/python3.12/site-packages:${PWD}/src" \
  -e HF_HOME=/cache/huggingface \
  -e TORCH_HOME=/cache/torch \
  -e MS_ASSET_DIR=/cache/maniskill \
  -e HTTP_PROXY="${PROXY_URL}" \
  -e HTTPS_PROXY="${PROXY_URL}" \
  -e NO_PROXY="localhost,127.0.0.1" \
  -v "${HOME}/code:${HOME}/code" \
  -v "${CACHE_DIR}:/cache" \
  -v /usr/share/vulkan/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/nvidia_icd.json:ro \
  -w "${PWD}" \
  omnigda-manip:jazzy \
  "$@"
