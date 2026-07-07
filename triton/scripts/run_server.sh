#!/usr/bin/env bash
# Launch Triton Inference Server (Docker) serving the nemotron_asr model(s).
#
# Prereqs:
#   - NVIDIA Container Toolkit (docker --gpus)
#   - The TRT engines on disk (encoder-nemotron.engine, decoder_joint-nemotron.engine,
#     and streaming_encoder-nemotron.engine for the streaming model)
#
# The backend runtime deps (tensorrt-cu13==10.16, torch cu13, numpy) are baked
# into a local image built from triton/Dockerfile, so the Python backend uses the
# container's own interpreter — no conda-pack / EXECUTION_ENV_PATH involved.
# Pick a base Triton release on Python 3.12 (e.g. 25.x-py3) via TRITON_IMAGE.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${HERE}/model_repository"

TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.10-py3}"
LOCAL_IMAGE="${LOCAL_IMAGE:-nemotron-asr-triton:latest}"
ENGINE_DIR="${ENGINE_DIR:-${HERE}/../nemotron-asr/nemotron-3.5-asr-streaming-0.6b/trt_engines}"

if [[ ! -f "${ENGINE_DIR}/encoder-nemotron.engine" ]]; then
  echo "ERROR: encoder-nemotron.engine not found under ENGINE_DIR=${ENGINE_DIR}"
  exit 1
fi

# Build the runtime image (deps baked in). Docker layer cache makes reruns cheap.
echo ">> building ${LOCAL_IMAGE} from ${TRITON_IMAGE}"
docker build -t "${LOCAL_IMAGE}" \
  --build-arg "TRITON_IMAGE=${TRITON_IMAGE}" \
  -f "${HERE}/Dockerfile" "${HERE}"

echo "Image:     ${LOCAL_IMAGE}"
echo "Repo:      ${REPO}"
echo "Engines:   ${ENGINE_DIR} -> /engines (matches config.pbtxt ENGINE_DIR)"

exec docker run --rm --gpus all --shm-size=1g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "${REPO}:/models:ro" \
  -v "${ENGINE_DIR}:/engines:ro" \
  "${LOCAL_IMAGE}" \
  tritonserver --model-repository=/models --log-verbose=1
