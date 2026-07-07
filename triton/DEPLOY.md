# Nemotron 3.5 ASR Triton 服务 — 部署与测试流程

一份**从零到验收**的操作清单。按顺序执行即可把服务跑起来并测通。
深入的接口/参数说明见 [`USAGE.md`](./USAGE.md)，依赖说明见 [`ENVIRONMENT.md`](./ENVIRONMENT.md)，
架构设计见 [`README.md`](./README.md)。

本机路径前缀以 `/home/ebcpc10/workspace/nemotron-asr` 为例。

---

## 0. 部署架构一览

```
客户端 ──HTTP:8000 / gRPC:8001──> Triton Server (Docker)
                                   ├── nemotron_asr            离线整段转写
                                   └── nemotron_asr_streaming  实时流式（序列批处理）
                                        └─ Python backend → asr_pipeline.py
                                             ├─ encoder / streaming_encoder (TRT)
                                             └─ decoder_joint (TRT, RNNT 贪心解码)
```

- 一个容器同时提供**离线**和**流式**两个模型，按模型名调用。
- 运行时依赖（`tensorrt-cu13` + `torch` + `numpy`）**直接装进镜像**，不用 conda-pack。
- TRT 引擎不打进镜像，启动时以 `-v` 挂载到容器 `/engines`。

---

## 1. 前置条件

- NVIDIA GPU + 驱动；Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)（`docker run --gpus all` 可用）
- 三个 TRT 引擎，默认位于 `nemotron-asr/nemotron-3.5-asr-streaming-0.6b/trt_engines/`：
  - `encoder-nemotron.engine`（离线 encoder）
  - `streaming_encoder-nemotron.engine`（流式 encoder，带 cache）
  - `decoder_joint-nemotron.engine`（两者共用）
- 客户端机器：Python 3.10+，能 `pip install tritonclient[all] soundfile`

> 引擎已随仓库构建好。若需从 `.nemo` 重建，见 [`USAGE.md` 第 1 节](./USAGE.md)。

快速确认引擎就位：

```bash
cd /home/ebcpc10/workspace/nemotron-asr
ls -lh nemotron-asr/nemotron-3.5-asr-streaming-0.6b/trt_engines/*.engine
```

---

## 2. 部署流程

### 2.1 启动服务（一条命令）

`run_server.sh` 会自动 `docker build`（首次构建镜像、装好依赖；之后走缓存秒级完成）
再 `docker run` 启动：

```bash
cd /home/ebcpc10/workspace/nemotron-asr
bash triton/scripts/run_server.sh
```

需要自定义基础镜像或引擎目录时：

```bash
TRITON_IMAGE=nvcr.io/nvidia/tritonserver:25.10-py3 \
ENGINE_DIR=/abs/path/to/trt_engines \
bash triton/scripts/run_server.sh
```

> 必须选基于 **Python 3.12** 的 Triton 版本（`25.x-py3`）。
> 后台常驻可改用 `docker run -d`（见 [第 5 节](#5-运维)）。

### 2.2 等待就绪

服务启动后，日志出现 `Started HTTPService at 0.0.0.0:8000` 即可。端口：
HTTP `8000` / gRPC `8001` / metrics `8002`。

```bash
curl -s localhost:8000/v2/health/ready -o /dev/null -w "server=%{http_code}\n"
curl -s localhost:8000/v2/models/nemotron_asr/ready          -o /dev/null -w "offline=%{http_code}\n"
curl -s localhost:8000/v2/models/nemotron_asr_streaming/ready -o /dev/null -w "streaming=%{http_code}\n"
```

三行都为 `200` 表示部署成功。

---

## 3. 测试流程

### 3.0 装客户端依赖（一次即可）

```bash
PY=/home/ebcpc10/miniconda3/envs/nemo-trt/bin/python   # 或任意 python3
$PY -m pip install "tritonclient[all]" soundfile
```

样例音频随仓库提供：

```bash
S=/home/ebcpc10/workspace/nemotron-asr/nemotron-asr/nemotron-3.5-asr-streaming-0.6b/.nemo_extracted/samples
# $S/sample1.flac, $S/sample2.flac （16kHz 单声道）
```

### 3.1 离线转写（HTTP）

```bash
cd /home/ebcpc10/workspace/nemotron-asr
$PY triton/client/asr_client.py --audio $S/sample1.flac --lang en-US
```

预期：一行完整转写，例如
`Going along slashy country roads ... immediately afterwards`。

### 3.2 流式转写（HTTP，逐步出字）

```bash
$PY triton/client/streaming_client.py --audio $S/sample2.flac --lang en-US --realtime
```

预期：文本随音频块逐步增长，最后打印 `FINAL: ...`：

```
[  5.4s] Before he had time to answer a much encumbered Vera burst into the room
...
FINAL: Before he had time to answer ... a lusty specimen of black red game cock. <en-US>
```

### 3.3 流式转写（gRPC，端口 8001，推荐实时）

```bash
$PY triton/client/grpc_streaming_client.py --audio $S/sample1.flac --lang en-US --realtime
```

预期：增量文本通过回调返回，最后打印 `FINAL: ...`。

### 3.4（可选）不经 Triton 的本地自检

绕开服务，直接验证流水线，并核对特征与 NeMo 的数值一致性：

```bash
$PY triton/test_pipeline.py --audio $S/sample1.flac --lang en-US
# 预期：[parity] max|diff| ≈ 0，然后打印 TRANSCRIPT
$PY triton/scripts/verify_session.py
# 预期：流式在线增量 vs 离线，特征逐位一致、转写一致性 ~0.975
```

---

## 4. 验收清单

- [ ] `ls .../trt_engines/*.engine` 三个引擎都在
- [ ] `run_server.sh` 构建并启动无报错
- [ ] 三个 `/ready` 端点都返回 `200`
- [ ] 离线客户端返回非空转写
- [ ] 流式客户端（HTTP）逐步出字且 `FINAL` 合理
- [ ] 流式客户端（gRPC）返回一致结果
- [ ]（可选）`test_pipeline.py` 特征 `max|diff| ≈ 0`

---

## 5. 运维

```bash
# 后台常驻启动（替代 run_server.sh 里的前台 docker run）
docker run -d --name nemotron-asr --restart unless-stopped --gpus all --shm-size=1g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /home/ebcpc10/workspace/nemotron-asr/triton/model_repository:/models:ro \
  -v /home/ebcpc10/workspace/nemotron-asr/nemotron-asr/nemotron-3.5-asr-streaming-0.6b/trt_engines:/engines:ro \
  nemotron-asr-triton:latest \
  tritonserver --model-repository=/models --log-verbose=1

docker logs -f nemotron-asr        # 看日志
docker stop nemotron-asr           # 停止
docker rm -f nemotron-asr          # 删除容器
curl -s localhost:8002/metrics     # Prometheus 指标
```

并发：流式默认最多 8 条并发流（`config.pbtxt` 的 `max_candidate_sequences`），
每条流用各自唯一且稳定的 `sequence_id`。

---

## 6. 故障排查（速查）

| 现象 | 排查 |
| --- | --- |
| `/ready` 非 200 / 容器退出 | `docker logs <容器>` 看报错；多为引擎路径或依赖问题 |
| `ENGINE_DIR must be set` / 找不到引擎 | 确认 `ENGINE_DIR` 指向含三个 `.engine` 的目录并挂到 `/engines` |
| 找不到 `streaming_encoder-nemotron.engine` | 按 [`USAGE.md` 第 1 节](./USAGE.md) 导出 + `build_streaming_trt.py` |
| backend `No module named torch/tensorrt` | 镜像未装依赖：用 `Dockerfile`（`run_server.sh` 自动构建），见 [`ENVIRONMENT.md`](./ENVIRONMENT.md) |
| `tensorrt` 版本与 engine 不符 | 须为 `tensorrt-cu13==10.16.1.11`（与构建 engine 一致） |
| 选错 Triton 版本 | 须基于 Python 3.12（`25.x-py3`） |
| 流式报 `batch size does not match` | 输入需带 batch 维：`AUDIO_CHUNK=[1,N]`、`TARGET_LANG=[1,1]`（仓库客户端已处理） |
| 转写全空白 | 多为缺 `prompt_kernel`；确认 `assets/prompt_kernel.pt` 存在 |
| 非 16kHz 音频 | 客户端先重采样（装了 `librosa` 时 `asr_client.py` 自动处理） |
| 多条流串话 | 每条流用各自唯一且稳定的 `sequence_id` |

更多接口细节与自定义调用代码见 [`USAGE.md`](./USAGE.md)。
