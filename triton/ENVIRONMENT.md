# Triton Python backend 运行时依赖 — 直接装进容器

不再使用 conda-pack / `EXECUTION_ENV_PATH` / `triton_env.tar.gz`。
Triton 的 Python backend 用的就是**容器自带的 Python 解释器**，所以把运行时依赖
直接装进 Triton 镜像即可。

运行时只需三个包（pipeline 是 NeMo-free 的），见
`[backend_env/requirements.txt](./backend_env/requirements.txt)`：


| 包               | 版本               | 说明                                                            |
| --------------- | ---------------- | ------------------------------------------------------------- |
| `torch`         | `2.12.0` (cu130) | 特征 STFT / prompt_kernel / 贪心解码；自动带 cudnn/cublas/nccl 等 CUDA 库 |
| `tensorrt-cu13` | `10.16.1.11`     | 加载并执行 engine（必须与构建 engine 的 TRT 版本一致）                         |
| `numpy`         | `2.4.6`          | I/O                                                           |


> 选基于 **Python 3.12** 的 Triton 版本（如 `nvcr.io/nvidia/tritonserver:25.x-py3`）。
> 由于 `tensorrt-cu13` 由 pip 装进 site-packages，**容器自带的 TRT 版本无需匹配**。

---

## 方式 A：构建镜像（推荐，可复现）

仓库已带 `[Dockerfile](./Dockerfile)`，在基础 Triton 镜像上把三个包装好。
`run_server.sh` 会自动构建并使用它：

```bash
bash triton/scripts/run_server.sh          # 自动 docker build + docker run
```

或手动构建：

```bash
docker build -t nemotron-asr-triton:latest \
  --build-arg TRITON_IMAGE=nvcr.io/nvidia/tritonserver:25.10-py3 \
  -f triton/Dockerfile triton/
```

Docker 层缓存会让重复构建很快。

---

## 方式 B：在已运行的容器里临时安装

不想构建镜像时，可先起官方容器，再 `pip install`（注意 `--rm` 退出即丢失）：

```bash
docker run -idt --gpus all -v "$PWD/triton/model_repository:/models:ro" -v "$PWD/nemotron-asr/:/workspace"   -v "$PWD/nemotron-asr/nemotron-3.5-asr-streaming-0.6b/trt_engines:/engines:ro"  --shm-size=20g --net=host --ipc=host docker.io/library/nemotron-asr-triton:lates

docker run --rm -it --gpus all --shm-size=1g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$PWD/triton/model_repository:/models:ro" \
  -v "<engine_dir>:/engines:ro" \
  nvcr.io/nvidia/tritonserver:25.10-py3 bash

# 容器内：
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r /models/../backend_env/requirements.txt --extra-index-url https://pypi.nvidia.com
# （或直接： pip install tensorrt-cu13==10.16.1.11 numpy==2.4.6）
tritonserver --model-repository=/models --log-verbose=1
```

自检：

```bash
python3 -c "import torch, tensorrt as trt, numpy; \
print('torch', torch.__version__, '| trt', trt.__version__, '| cuda', torch.cuda.is_available())"
```

---

## 说明

- `config.pbtxt` 里已**移除** `EXECUTION_ENV_PATH`：backend 直接用容器 Python。
- `torch` 的 cu130 轮子不在默认 PyPI，需走 `https://download.pytorch.org/whl/cu130`；
`tensorrt-cu13` 走 `https://pypi.nvidia.com`。
- `torch` 会自动拉取匹配的 `nvidia-`* CUDA 库，无需手动装。

