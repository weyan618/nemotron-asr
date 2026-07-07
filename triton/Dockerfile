# Triton Inference Server + ASR backend runtime deps baked in.
#
# The Python backend uses the container's own Python interpreter, so we just
# install the three runtime packages (NeMo-free pipeline) directly here. No
# conda-pack / EXECUTION_ENV_PATH needed.
#
# Pick a Triton release on Python 3.12 (25.x-py3). Override with:
#   docker build --build-arg TRITON_IMAGE=nvcr.io/nvidia/tritonserver:25.10-py3 ...
ARG TRITON_IMAGE=nvcr.io/nvidia/tritonserver:25.10-py3
FROM ${TRITON_IMAGE}

COPY backend_env/requirements.txt /tmp/requirements.txt

# torch's cu130 wheel is not on default PyPI -> install from the PyTorch cu130
# index first; then tensorrt-cu13 + numpy (tensorrt wheels live on pypi.nvidia.com).
RUN python3 -m pip install --no-cache-dir \
        torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130 \
 && python3 -m pip install --no-cache-dir \
        -r /tmp/requirements.txt --extra-index-url https://pypi.nvidia.com \
 && python3 -c "import torch, tensorrt as trt, numpy; \
        print('torch', torch.__version__, '| trt', trt.__version__)"
