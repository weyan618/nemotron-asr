# Nemotron 3.5 ASR Triton 服务 — 使用指南

本文档给出**离线**与**流式**两种推理的具体操作步骤（接口/参数参考）。
照着走的端到端**部署+测试流程**见 `[DEPLOY.md](./DEPLOY.md)`；架构与设计说明见 `[README.md](./README.md)`。

- 离线模型：`nemotron_asr`（整段音频一次转写）
- 流式模型：`nemotron_asr_streaming`（音频分块、实时增量出字）

两个模型在同一个 Triton server 中并存，按模型名各自调用。

---

## 0. 前置条件

- NVIDIA GPU + 驱动；Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)（`docker run --gpus all` 可用）
- 构建引擎用的 `nemo-trt` conda 环境（Python 3.12，TensorRT 10.16，torch cu13）
- 三个 TRT 引擎（见第 1 节），默认位于
`nemotron-asr/nemotron-3.5-asr-streaming-0.6b/trt_engines/`：
  - `encoder-nemotron.engine`（离线 encoder）
  - `streaming_encoder-nemotron.engine`（流式 encoder，带 cache）
  - `decoder_joint-nemotron.engine`（两者共用）

路径前缀以本机为例：`/home/ebcpc10/workspace/nemotron-asr`。

---

## 1. 准备引擎（已就绪可跳过）

仓库已构建好三个引擎。如需从 `.nemo` 重建：

```bash
PY=/home/ebcpc10/miniconda3/envs/nemo-trt/bin/python
cd /home/ebcpc10/workspace/nemotron-asr

# 离线 encoder + decoder_joint（原有脚本）
$PY nemotron-asr/nemotron-3.5-asr-streaming-0.6b/export_trt_onnx.py
$PY nemotron-asr/nemotron-3.5-asr-streaming-0.6b/build_trt.py

# 流式 encoder（cache-aware，att_context_size=[56,3] → 320ms chunk）
$PY triton/scripts/export_streaming_encoder.py --att 56 3
$PY triton/scripts/build_streaming_trt.py

# （可选）从 .nemo 重新生成 assets：vocab / prompt_dictionary / feat / prompt_kernel
$PY triton/scripts/extract_assets.py
```

> 改 chunk 大小需三处一致：`export_streaming_encoder.py --att`、重建引擎、
> 以及 `model_repository/nemotron_asr/1/asr_pipeline.py` 里的 `STREAMING_CFG`。

构建后本地核验（不经 Triton，强烈建议先跑）：

```bash
# 特征与 NeMo 对齐 + 离线端到端转写
$PY triton/test_pipeline.py --audio <16k音频> --lang en-US
# 流式在线增量 vs 离线（特征逐位一致 max|diff|=0，转写一致性 ~0.975）
$PY triton/scripts/verify_session.py
```

---

## 2. 运行时依赖（直接装进 Triton 容器）

Triton 的 Python backend 用**容器自带的 Python**，所以把依赖直接装进 Triton 镜像即可，
**不用** conda-pack / `EXECUTION_ENV_PATH`。运行时只需 `tensorrt-cu13` + `torch` + `numpy`
（pipeline 是 NeMo-free 的）。完整说明见 `[ENVIRONMENT.md](./ENVIRONMENT.md)`，包列表见
`[backend_env/requirements.txt](./backend_env/requirements.txt)`。

仓库已带 `[Dockerfile](./Dockerfile)`，在基础 Triton 镜像上装好这三个包。
第 3 节的 `run_server.sh` 会自动 `docker build` 并启动，无需手动操作。手动构建：

```bash
docker build -t nemotron-asr-triton:latest \
  --build-arg TRITON_IMAGE=nvcr.io/nvidia/tritonserver:25.10-py3 \
  -f triton/Dockerfile triton/
```

> 选基于 **Python 3.12** 的 Triton 版本（`nvcr.io/nvidia/tritonserver:25.x-py3`）。
> 由于 `tensorrt-cu13` 由 pip 装入，**容器自带的 TRT 版本无需匹配**。
> 也可在已运行的容器里临时 `pip install`，见 ENVIRONMENT.md 方式 B。

---

## 3. 启动服务

`run_server.sh` 会先 `docker build`（按第 2 节装好依赖，Docker 缓存让重复构建很快），
再 `docker run` 启动。

```bash
# 默认 ENGINE_DIR 指向仓库内 trt_engines；按需覆盖镜像/引擎目录
bash triton/scripts/run_server.sh

# 或：
TRITON_IMAGE=nvcr.io/nvidia/tritonserver:25.10-py3 \
ENGINE_DIR=/abs/path/to/trt_engines \
bash triton/scripts/run_server.sh
```

端口：HTTP `8000` / gRPC `8001` / metrics `8002`。健康检查与模型状态：

```bash
curl -s localhost:8000/v2/health/ready && echo " READY"
curl -s localhost:8000/v2/models/nemotron_asr/ready && echo " offline READY"
curl -s localhost:8000/v2/models/nemotron_asr_streaming/ready && echo " streaming READY"
```

---

## 4. 安装客户端依赖

```bash
pip install -r triton/client/requirements.txt
```

---

## 5. 离线转写 `nemotron_asr`

适合一次性整段音频（建议 5–25s，上限 ~30s）。

### 5.1 Python 客户端

```bash
python triton/client/asr_client.py --audio sample1.flac --lang en-US
python triton/client/asr_client.py --audio clip.wav --lang zh-CN --url localhost:8000
```

不想装 `tritonclient` 时，可用纯 `requests` 直连 KServe v2 REST 的等价 demo
（`http_demo.py`，音频以二进制张量发送，更轻量）：

```bash
pip install requests soundfile numpy
python triton/client/http_demo.py --audio sample1.flac --lang en-US --url http://localhost:8000
```

### 5.2 接口


| 张量            | 方向     | 类型    | 形状     | 说明                    |
| ------------- | ------ | ----- | ------ | --------------------- |
| `AUDIO`       | in     | FP32  | `[-1]` | 16kHz 单声道，幅度 ∈ [-1,1] |
| `TARGET_LANG` | in(可选) | BYTES | `[1]`  | 语言码，缺省 `en-US`        |
| `TRANSCRIPT`  | out    | BYTES | `[1]`  | 完整转写文本                |


### 5.3 自定义最小调用（Python）

```python
import numpy as np, soundfile as sf, tritonclient.http as http

audio, sr = sf.read("sample1.flac")              # 需 16kHz 单声道，否则先重采样
audio = audio.astype("float32")
cli = http.InferenceServerClient("localhost:8000")

a = http.InferInput("AUDIO", [audio.shape[0]], "FP32"); a.set_data_from_numpy(audio)
l = http.InferInput("TARGET_LANG", [1], "BYTES")
l.set_data_from_numpy(np.array([b"en-US"], dtype=object))
r = cli.infer("nemotron_asr", [a, l], outputs=[http.InferRequestedOutput("TRANSCRIPT")])
print(r.as_numpy("TRANSCRIPT")[0].decode())
```

### 5.4 gRPC 调用（端口 8001）

离线模型同样支持 gRPC，接口名一致，仅换客户端模块与端口：

```python
import numpy as np, soundfile as sf, tritonclient.grpc as grpc

audio = sf.read("sample1.flac")[0].astype("float32")
cli = grpc.InferenceServerClient("localhost:8001")
a = grpc.InferInput("AUDIO", [audio.shape[0]], "FP32"); a.set_data_from_numpy(audio)
l = grpc.InferInput("TARGET_LANG", [1], "BYTES")
l.set_data_from_numpy(np.array([b"en-US"], dtype=object))
r = cli.infer("nemotron_asr", [a, l], outputs=[grpc.InferRequestedOutput("TRANSCRIPT")])
print(r.as_numpy("TRANSCRIPT")[0].decode())
```

---

## 6. 流式转写 `nemotron_asr_streaming`

适合实时/边说边出字。每条音频流是一个 Triton **序列**：用固定 `sequence_id` 连续发块，
首块 `sequence_start=True`、末块 `sequence_end=True`。服务端按 `correlation_id` 维护每条流的
encoder cache 与 RNNT 解码状态，每次只返回**自上一块以来新增的文本（增量）**。

### 6.1 Python 客户端

```bash
# 把文件切成 320ms 块顺序发送；--realtime 模拟真实节奏（每块间隔 320ms）
python triton/client/streaming_client.py --audio sample1.flac --lang en-US --realtime
python triton/client/streaming_client.py --audio clip.wav --chunk-ms 320 --url localhost:8000
```

输出示例（文本随音频到达逐步增长）：

```
[  0.3s] Going a
[  0.6s] Going along
...
[ 11.5s] Going along slushy country roads ... immediately afterwards. <en-US>
FINAL: Going along slushy country roads ... immediately afterwards. <en-US>
```

### 6.2 接口


| 张量                       | 方向     | 类型    | 形状     | 说明                                           |
| ------------------------ | ------ | ----- | ------ | -------------------------------------------- |
| `AUDIO_CHUNK`            | in     | FP32  | `[-1]` | 本次新增的 16kHz 单声道样本（任意长度，~320ms 最佳）            |
| `TARGET_LANG`            | in(可选) | BYTES | `[1]`  | 语言码，于 `sequence_start` 请求读取                  |
| `TRANSCRIPT`             | out    | BYTES | `[1]`  | 相对上一块的**新增**文本                               |
| `START/READY/END/CORRID` | in     | —     | —      | 序列控制输入，由 tritonclient 通过 `sequence_`* 参数自动注入 |


### 6.3 自定义最小调用（Python，实时麦克风思路）

```python
import random, numpy as np, tritonclient.http as http

cli = http.InferenceServerClient("localhost:8000")
seq_id = random.randint(1, 2**31)                # 每条流唯一
full = ""

def send(chunk_f32, start=False, end=False, lang="en-US"):
    global full
    ins = [http.InferInput("AUDIO_CHUNK", [chunk_f32.shape[0]], "FP32")]
    ins[0].set_data_from_numpy(chunk_f32.astype("float32"))
    if start:                                    # 仅首块带语言码
        li = http.InferInput("TARGET_LANG", [1], "BYTES")
        li.set_data_from_numpy(np.array([lang.encode()], dtype=object)); ins.append(li)
    r = cli.infer("nemotron_asr_streaming", ins,
                  outputs=[http.InferRequestedOutput("TRANSCRIPT")],
                  sequence_id=seq_id, sequence_start=start, sequence_end=end)
    delta = r.as_numpy("TRANSCRIPT")[0].decode()
    full += delta
    return delta

# 逐块喂入（如来自麦克风的 320ms = 5120 个 16k 样本）：
# send(chunk0, start=True)
# send(chunk1); send(chunk2); ...
# send(last_chunk, end=True)   # 结束并冲刷尾帧；会话随之清理
```

要点：

- 同一条流的所有请求必须用**同一个 `sequence_id`**；不同流用不同值（可并发多达 8 条）。
- `sequence_start` 用来创建会话并读取 `TARGET_LANG`；`sequence_end` 触发尾帧冲刷并释放该流状态。
- 块大小可任意（服务端在线累计并按 320ms 内部切块）；越接近 320ms 整数倍延迟越平滑。

### 6.4 gRPC 双向流（推荐用于实时）

gRPC 用一条**双向流**（`start_stream` + `async_stream_infer`）发送同一序列的所有块，
响应（增量文本）通过回调异步返回，延迟更低、更贴近实时麦克风场景。端口为 `8001`。

```bash
python triton/client/grpc_streaming_client.py --audio sample1.flac --lang en-US --realtime
python triton/client/grpc_streaming_client.py --audio clip.wav --chunk-ms 320 --url localhost:8001
```

最小自定义调用：

```python
import queue, random, numpy as np, tritonclient.grpc as grpc

cli = grpc.InferenceServerClient("localhost:8001")
seq_id = random.randint(1, 2**31)
out_q = queue.Queue()
cli.start_stream(callback=lambda result, error: out_q.put((result, error)))

def send(chunk_f32, start=False, end=False, lang="en-US", req_id="0"):
    ins = [grpc.InferInput("AUDIO_CHUNK", [chunk_f32.shape[0]], "FP32")]
    ins[0].set_data_from_numpy(chunk_f32.astype("float32"))
    if start:                                          # 仅首块带语言码
        li = grpc.InferInput("TARGET_LANG", [1], "BYTES")
        li.set_data_from_numpy(np.array([lang.encode()], dtype=object)); ins.append(li)
    cli.async_stream_infer("nemotron_asr_streaming", inputs=ins,
                           outputs=[grpc.InferRequestedOutput("TRANSCRIPT")],
                           request_id=req_id, sequence_id=seq_id,
                           sequence_start=start, sequence_end=end)

# 发送：send(chunk0, start=True); send(chunk1); ...; send(last, end=True)
# 接收（每发一块取一条响应，序列调度保证同一流内有序）：
#   result, error = out_q.get(); delta = result.as_numpy("TRANSCRIPT")[0].decode()
# 结束：cli.stop_stream()
```

> HTTP（第 6.1/6.3 节）逐块同步请求、实现简单；gRPC 双向流吞吐/延迟更优，适合生产实时流。
> 两者语义完全一致：都靠 `sequence_id` + `sequence_start/sequence_end` 标识一条流的起止。

---

## 7. 语言码

`TARGET_LANG` 取值见 `model_repository/nemotron_asr/1/assets/prompt_dictionary.json`，
含 `en-US`、`es-ES`、`de-DE`、`fr-FR`、`zh-CN`、`ja-JP`、`ko-KR` 等 40 语区，以及 `auto`。
传入未知码会报错并提示可选值。

---

## 8. 不经 Triton 直接调用（本地/集成）

`asr_pipeline.py` 与 Triton 解耦，可直接在 `nemo-trt` 环境里 import：

```python
import sys; sys.path.insert(0, "triton/model_repository/nemotron_asr/1")
import numpy as np, soundfile as sf
from asr_pipeline import NemotronASR

asr = NemotronASR(engine_dir="nemotron-asr/nemotron-3.5-asr-streaming-0.6b/trt_engines",
                  assets_dir="triton/model_repository/nemotron_asr/1/assets")
audio = sf.read("sample1.flac")[0].astype("float32")

# 离线
print(asr.transcribe(audio, "en-US"))
print(asr.transcribe_batch([audio, audio], ["en-US", "zh-CN"]))   # GPU 批量

# 流式
s = asr.new_stream("en-US")
step = 5120                                                       # 320ms @16k
for i in range(0, len(audio), step):
    delta = s.add_audio(audio[i:i+step])
    if delta: print(delta, end="", flush=True)
print(s.finalize())                                              # 冲刷尾帧
```

---

## 9. 常见问题


| 现象                                                | 排查                                                                              |
| ------------------------------------------------- | ------------------------------------------------------------------------------- |
| 启动报 `ENGINE_DIR must be set`                      | 确认 `run_server.sh` 的 `ENGINE_DIR` 指向含三个 `.engine` 的目录，并已挂载为 `/engines`          |
| 流式模型加载失败：找不到 `streaming_encoder-nemotron.engine`  | 先跑第 1 节的导出 + `build_streaming_trt.py`                                           |
| Python backend 报 `No module named torch/tensorrt` | 镜像没装依赖：用 `Dockerfile`（`run_server.sh` 自动构建）或在容器内 `pip install`，见 ENVIRONMENT.md |
| `tensorrt` 版本与 engine 不符                          | 确认装的是 `tensorrt-cu13==10.16.1.11`（与构建 engine 的 TRT 一致）                          |
| 选错 Triton 版本                                      | 需基于 Python 3.12（`25.x-py3`）                                                     |
| 转写全为空白                                            | 多为缺 `prompt_kernel` 语言条件；本服务已内置，确认 `assets/prompt_kernel.pt` 存在                 |
| 非 16kHz 音频                                        | 客户端先重采样（已装 `librosa` 时 `asr_client.py` 会自动处理）                                   |
| 流式各块返回空字符串                                        | 正常：需累计够一个 320ms 块才会出字；末块发 `sequence_end` 可冲刷剩余                                  |
| 多条流串话/状态错乱                                        | 检查每条流是否用了各自唯一且稳定的 `sequence_id`                                                 |


