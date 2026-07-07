# Nemotron 3.5 ASR — Triton 推理服务

基于 NVIDIA **Nemotron 3.5 ASR streaming 0.6B**（Cache-Aware FastConformer-RNNT，多语言）
的两个 TensorRT engine，封装成 [Triton Inference Server](https://github.com/triton-inference-server/server)
的推理服务。输入音频，输出文本转写。

> 📖 **端到端部署+测试流程见 [`DEPLOY.md`](./DEPLOY.md)；接口/调用细节见 [`USAGE.md`](./USAGE.md)；
> 运行时依赖见 [`ENVIRONMENT.md`](./ENVIRONMENT.md)。** 本文偏架构与设计说明。

服务提供两种模型：

- **`nemotron_asr`（离线/整段转写）**：导出的 encoder engine 仅接收 `audio_signal + length`，
  适合单条 ≤ 30s 的整段音频，吞吐高。
- **`nemotron_asr_streaming`（实时流式）**：使用**重新导出的 cache-aware streaming encoder**
  （带 `cache_last_channel/time` 输入输出），按 ~320ms 分块逐步推理，延迟有界，可做实时转写。

---

## 1. 整体架构

服务由一个 Triton **Python backend** 模型 `nemotron_asr` 编排完整流水线（核心逻辑在
`asr_pipeline.py`，与 Triton 解耦、可独立测试）：

```
AUDIO (16kHz mono, FP32)
   │
   ▼  LeanFeaturizer            log-mel 特征，数值与 NeMo AudioToMelSpectrogramPreprocessor 完全一致
   ▼  encoder-nemotron.engine   audio_signal,length -> outputs[B,1024,T]
   ▼  prompt_kernel (MLP)        语言条件：拼接 128 维语言 one-hot 后投影 (1152->2048->1024)
   │                            ⚠️ 该步骤未包含在导出的 engine 里，必须在服务侧补上，否则输出全为 blank
   ▼  decoder_joint engine      RNNT 贪心解码，逐步调用 (max_symbols=10)
   ▼  detokenize                joint 词表(13087) + sentencepiece "▁"->空格
TRANSCRIPT (text)
```

关键事实（均已从 checkpoint / engine 中核实）：

| 项 | 值 |
| --- | --- |
| 特征 | 128-dim log-mel，sr=16000，n_fft=512，win=400，hop=160，preemph=0.97，mag_power=2，log(add, 2⁻²⁴)，normalize=NA(无) |
| encoder 输出维度 | 1024 |
| 词表大小 | 13087；RNNT blank id = 13087；joint logits 维度 = 13088 |
| RNNT 预测网络 | LSTM 2 层，hidden 640；状态 `input_states_1/2 = [2,B,640]` |
| 语言 prompt | `num_prompts=128`，由 `prompt_dictionary` 把语言码映射到 one-hot 下标 |
| max symbols/step | 10 |

---

## 2. 目录结构

```
triton/
├── Dockerfile                       # Triton 基础镜像 + 运行时依赖（torch/tensorrt-cu13/numpy）
├── DEPLOY.md                        # 端到端部署 + 测试流程（照着走）
├── USAGE.md                         # 离线/流式接口与调用示例（HTTP/gRPC）
├── ENVIRONMENT.md                   # 运行时依赖说明（直接装进容器）
├── backend_env/
│   └── requirements.txt             # backend 运行时三件套锁版本（供 Dockerfile 安装）
├── model_repository/
│   ├── nemotron_asr/                 # 离线模型
│   │   ├── config.pbtxt              # Triton 模型配置（输入/输出/参数/执行环境）
│   │   └── 1/
│   │       ├── model.py             # Triton Python backend 入口
│   │       ├── asr_pipeline.py      # 核心流水线（含离线 + StreamingSession，无 Triton/NeMo 运行时依赖）
│   │       └── assets/
│   │           ├── vocab.json
│   │           ├── prompt_dictionary.json
│   │           ├── feat_assets.pt   # mel 滤波器 + 窗
│   │           └── prompt_kernel.pt # 语言条件 MLP 权重
│   └── nemotron_asr_streaming/       # 实时流式模型（sequence batching）
│       ├── config.pbtxt              # 含 START/READY/END/CORRID 序列控制输入
│       └── 1/model.py               # 复用上面的 asr_pipeline.py，按 correlation_id 维护会话
├── client/
│   ├── asr_client.py                # 离线 tritonclient 示例（HTTP）
│   ├── http_demo.py                 # 离线纯 requests 示例（KServe v2 REST，无需 tritonclient）
│   ├── streaming_client.py          # 流式客户端（HTTP，按序列分块发送）
│   ├── grpc_streaming_client.py     # 流式客户端（gRPC 双向流，推荐实时）
│   └── requirements.txt
├── scripts/
│   ├── extract_assets.py            # 从 .nemo 重新生成上面的 assets
│   ├── export_streaming_encoder.py  # 导出 cache-aware streaming encoder 为 ONNX
│   ├── build_streaming_trt.py       # 由该 ONNX 构建 streaming encoder TRT engine
│   ├── verify_streaming.py          # 用 NeMo 特征 buffer 驱动，验证 streaming engine
│   ├── verify_session.py            # 验证在线 StreamingSession（原始音频增量喂入）
│   └── run_server.sh                # docker build（装依赖）+ run tritonserver
└── test_pipeline.py                 # 独立验证：特征对齐 + 端到端转写
```

TRT engine（`encoder-nemotron.engine` 1.2G、`streaming_encoder-nemotron.engine` 1.2G、
`decoder_joint-nemotron.engine` 47M）不放进仓库，运行时通过 `ENGINE_DIR` 挂载引用。

---

## 3. 运行时环境

Python backend 直接用**容器自带的 Python**。运行时只需与 engine 匹配的
**tensorrt-cu13 10.16** + **torch (cu13)** + numpy，通过 [`Dockerfile`](./Dockerfile)
（包列表见 [`backend_env/requirements.txt`](./backend_env/requirements.txt)）直接装进 Triton 镜像，
**不用** conda-pack / `EXECUTION_ENV_PATH`。由于 `tensorrt-cu13` 由 pip 装入，
**容器自带的 TRT 版本无需匹配**。详见 [`ENVIRONMENT.md`](./ENVIRONMENT.md)。

> 注意：请选用基于 **Python 3.12** 的 Triton 版本（例如 `25.x-py3`）。

---

## 4. 快速开始

### 4.1 （可选）重新生成 assets

仓库里已包含 assets。如需从 `.nemo` 重建：

```bash
/home/ebcpc10/miniconda3/envs/nemo-trt/bin/python triton/scripts/extract_assets.py
```

### 4.2 本地核验流水线（不依赖 Triton，强烈建议先跑）

```bash
/home/ebcpc10/miniconda3/envs/nemo-trt/bin/python triton/test_pipeline.py \
    --audio <你的16k音频> --lang en-US
```

预期：特征与 NeMo `max|diff| ≈ 0`，并打印转写文本。

### 4.3 启动 Triton 服务（自动构建含依赖的镜像）

```bash
# ENGINE_DIR 默认指向仓库内的 trt_engines；如有需要可覆盖
bash triton/scripts/run_server.sh
# 或自定义镜像 / 引擎目录：
TRITON_IMAGE=nvcr.io/nvidia/tritonserver:25.10-py3 \
ENGINE_DIR=/abs/path/to/trt_engines \
bash triton/scripts/run_server.sh
```

启动后 HTTP=8000 / gRPC=8001 / metrics=8002。健康检查：

```bash
curl -s localhost:8000/v2/health/ready && echo READY
```

### 4.4 调用

```bash
pip install -r triton/client/requirements.txt
# 离线整段
python triton/client/asr_client.py --audio sample1.flac --lang en-US
# 实时流式（按 320ms 分块发送，逐步打印增长的转写；--realtime 模拟真实节奏）
python triton/client/streaming_client.py --audio sample1.flac --lang en-US --realtime
```

---

## 5. 流式（实时）转写

`nemotron_asr_streaming` 用 Triton **sequence batching** 把每条音频流当作一个序列：客户端用
固定的 `sequence_id` 连续发送音频块，首块带 `sequence_start`、末块带 `sequence_end`。服务端按
`correlation_id` 为每条流维护一个 `StreamingSession`（encoder cache + RNNT 解码状态），每次只处理
新到的音频块并返回**增量文本**。

```
原始音频块 (任意长度，建议 ~320ms)
   │ 累积到会话原始缓冲，在线计算 log-mel 特征
   ▼ 按 chunk(32 帧新特征 + 9 帧 pre-encode 重叠 = 41 帧窗) 切块
   ▼ streaming_encoder engine: 窗口 + cache_{channel,time} -> 4 个 encoder 帧 + 新 cache
   ▼ prompt_kernel (与离线一致)
   ▼ decoder_joint 持久贪心：last_label / LSTM 状态跨块延续
   ▼ detokenize -> 返回相对上一块的新增文本
```

- 块配置对应导出时的 `att_context_size=[56,3]`（320ms chunk，subsampling 8），由
  `export_streaming_encoder.py --att 56 3` 决定；改 chunk 大小需重新导出 + 重建 engine + 同步
  `asr_pipeline.STREAMING_CFG`。
- 流式与离线在样例上转写一致性 ~0.975（差异主要是有限右上下文带来的细节及标点/语言标签）。
- 验证：`python triton/scripts/verify_session.py`（在线增量喂入 vs 离线整段）。

---

## 6. 接口

### 离线 `nemotron_asr`（`max_batch_size=0`，每请求一条音频，可并发）

| 张量 | 方向 | 类型 | 形状 | 说明 |
| --- | --- | --- | --- | --- |
| `AUDIO` | in | FP32 | `[-1]` | 16kHz 单声道波形，幅度 ∈ [-1,1] |
| `TARGET_LANG` | in(可选) | BYTES | `[1]` | 语言码，如 `en-US`/`de-DE`/`zh-CN`，缺省 `en-US` |
| `TRANSCRIPT` | out | BYTES | `[1]` | 解码文本 |

### 流式 `nemotron_asr_streaming`（`sequence_batching`，每序列一条音频流）

| 张量 | 方向 | 类型 | 形状 | 说明 |
| --- | --- | --- | --- | --- |
| `AUDIO_CHUNK` | in | FP32 | `[-1]` | 本次新增的 16kHz 单声道样本（任意长度） |
| `TARGET_LANG` | in(可选) | BYTES | `[1]` | 语言码，于 `sequence_start` 请求读取 |
| `TRANSCRIPT` | out | BYTES | `[1]` | 自上一块以来**新增**的转写文本（增量） |

调用需带 `sequence_id` + `sequence_start/sequence_end`（见 `streaming_client.py`）。

支持的语言码见 `assets/prompt_dictionary.json`（含 `en-US`、`es-ES`、`de-DE`、`fr-FR`、
`zh-CN`、`ja-JP`、`ko-KR` 等 40 语区）。

---

## 7. 设计取舍 / 已知限制

- **语言 prompt 必须在服务侧施加。** 导出的 encoder engine 不含 `prompt_kernel`，
  若跳过该步会得到空转写。本服务用从 checkpoint 提取的 `prompt_kernel.pt` 补齐。
- **离线整段**：`encoder-nemotron.engine` 无 cache 状态输入，不是逐块流式，单条建议 5–25s、上限 ~30s；
  需要实时增量结果时改用 `nemotron_asr_streaming`。
- **流式末块**：导出的 streaming encoder 每块固定输出 `valid_out_len=4` 帧，末尾不足一块时右侧补零并按
  真实长度 mask；极少量尾帧的输出与离线略有差异（已计入上面 ~0.975 的一致性）。
- **在线特征为增量计算**：每块只对其所需的 41 帧窗口做 STFT（按绝对帧对齐 + 零填充边界处理），与全量
  featurizer **逐位一致**（`verify_session.py` 中 `max|diff|=0`）。每块耗时与滚动原始缓冲均为 **O(1)**
  （实测 164s 流每块 ~11ms、缓冲恒 ~426ms），不随流长度增长。
- **GPU 端批量贪心解码**：encoder 与 RNNT 贪心循环都按 batch 执行。Triton 默认调度器会把
  同时排队的多个请求作为一个列表传入 `execute`，`model.py` 将其合并为一个 batch（按
  `MAX_BATCH=8` 分块，对齐 engine profile 上限）；不等长音频自动右侧补零并按真实长度 mask，
  每条独立维护 time 指针 / LSTM 状态 / 已发射符号数，逐步对整批调用一次 `decoder_joint`。
  实测 8 条并发相比逐条顺序约 **5.3×** 加速，且各 batch 大小结果与单条逐位一致。
- 单流执行：所有 torch 算子与两个 TRT engine 共用同一条 CUDA 流，避免跨流竞争。
- 特征实现已与 NeMo 逐位对齐（`test_pipeline.py` 中 `max|diff|=0`）。
