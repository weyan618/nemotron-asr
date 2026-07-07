---
license: other
license_name: openmdw-1.1
license_link: >-
  https://openmdw.ai/license/1-1/
library_name: nemo
language:
- en
- es
- de
- fr
- it
- ar
- ja
- ko
- pt
- ru
- hi
- zh
- vi
- he
- nl
- cs
- da
- pl
- 'no'
- sv
- th
- tr
- bg
- el
- et
- fi
- hr
- hu
- lt
- lv
- ro
- sk
- uk
- mt
- sl
datasets:
- nvidia/Granary
- multilingual_librispeech
- fleurs
- mozilla-foundation/common_voice_8_0
- voxpopuli
- europarl
thumbnail: null
tags:
- speech-recognition
- cache-aware ASR
- automatic-speech-recognition
- streaming-asr
- multilingual
- speech
- audio
- FastConformer
- RNNT
- Parakeet
- ASR
- pytorch
- NeMo
widget:
- example_title: Librispeech sample 1
  src: https://cdn-media.huggingface.co/speech_samples/sample1.flac
- example_title: Librispeech sample 2
  src: https://cdn-media.huggingface.co/speech_samples/sample2.flac
model-index:
- name: nemotron-asr-streaming-multilingual-0.6b
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: FLEURS (English)
      type: google/fleurs
      config: en_us
      split: test
    metrics:
    - name: WER (1.12s frame size, LangID)
      type: wer
      value: 7.91
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: FLEURS (Spanish)
      type: google/fleurs
      config: es_419
      split: test
    metrics:
    - name: WER (1.12s frame size, LangID)
      type: wer
      value: 4.11
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: FLEURS (French)
      type: google/fleurs
      config: fr_fr
      split: test
    metrics:
    - name: WER (1.12s frame size, LangID)
      type: wer
      value: 9.03
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: FLEURS (Italian)
      type: google/fleurs
      config: it_it
      split: test
    metrics:
    - name: WER (1.12s frame size, LangID)
      type: wer
      value: 4.25
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: FLEURS (Portuguese)
      type: google/fleurs
      config: pt_br
      split: test
    metrics:
    - name: WER (1.12s frame size, LangID)
      type: wer
      value: 5.48
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: FLEURS (German)
      type: google/fleurs
      config: de_de
      split: test
    metrics:
    - name: WER (1.12s frame size, LangID)
      type: wer
      value: 8.31
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: FLEURS (Hindi)
      type: google/fleurs
      config: hi_in
      split: test
    metrics:
    - name: WER (1.12s frame size, LangID)
      type: wer
      value: 6.81
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: FLEURS (Korean)
      type: google/fleurs
      config: ko_kr
      split: test
    metrics:
    - name: WER (1.12s frame size, LangID)
      type: wer
      value: 7.12
metrics:
- wer
pipeline_tag: automatic-speech-recognition
---

# Nemotron 3.5 ASR

<style>
h1, h2, h3, h4, h5, h6 {
  color: #76b900; /* NVIDIA green */
  font-weight: 700;
}

hr {
  border: none;
  border-top: 1px solid #e5e7eb;
  margin: 2rem 0;
}

/* Improve list spacing */
ul, ol {
  margin-top: 0.5rem;
  margin-bottom: 0.5rem;
}

/* Badge alignment consistency */
img {
  display: inline;
  vertical-align: middle;
}
</style>

<p align="center">
  <a href="#model-architecture"><img src="https://img.shields.io/badge/Model_Arch-FastConformer--CacheAware--RNNT-lightgrey#model-badge" alt="Model architecture"/></a>
  &nbsp;
  <a href="#model-architecture"><img src="https://img.shields.io/badge/Params-600M-lightgrey#model-badge" alt="Model size"/></a>
  &nbsp;
  <a href="#supported-languages"><img src="https://img.shields.io/badge/Language-Multilingual-lightgrey#model-badge" alt="Language"/></a>
</p>

<p align="center">
  <img src="model_overview.png" alt="Nemotron 3.5 ASR overview: multilingual audio across 40 language-locales is transcribed by a cache-aware FastConformer-RNNT model with language-ID prompting into punctuated text with an automatic language tag" width="900"/>
</p>

> [!Note]
> This model is the multilingual extension of [nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b), adding language-ID prompt conditioning to support transcription across **40 language-locales** from a single model.

**Nemotron 3.5 ASR** is a multilingual, streaming Automatic Speech Recognition (ASR) model engineered to deliver high-quality multilingual transcription across both low-latency streaming and high-throughput batch workloads. Developed by NVIDIA, this 600M parameter model transcribes speech into text with native support for punctuation and capitalization, and offers runtime flexibility with configurable chunk sizes, including 80ms, 160ms, 320ms, 560ms, and 1120ms.

By leveraging a state-of-the-art **Cache-Aware FastConformer-RNNT** architecture, the model eliminates redundant overlapping computations common in traditional "buffered" streaming. This allows it to process only new audio chunks while reusing cached encoder context, significantly improving computational efficiency and minimizing end-to-end delay without sacrificing accuracy.

It was trained on a massive ASR dataset and is engineered to perform across diverse and challenging acoustic conditions.

This model is ready for commercial use.

---

## License/Terms of Use

Governing Terms: Use of the model is governed by the [OpenMDW-1.1](https://openmdw.ai/license/1-1/) license.

## Deployment Geography

Global

## Use Case

This model is for transcription of multilingual audio.

## Release Date

- Hugging Face [06/04/2026] via https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b

## References

<a id="ref-1"></a>[1] [Stateful Conformer with Cache-based Inference for Streaming Automatic Speech Recognition](https://arxiv.org/abs/2312.17279)

<a id="ref-2"></a>[2] [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084)

<a id="ref-3"></a>[3] [NVIDIA Granary](https://huggingface.co/datasets/nvidia/Granary)

<a id="ref-4"></a>[4] [NVIDIA NeMo Framework](https://github.com/NVIDIA/NeMo)

## Why Choose Nemotron 3.5 ASR?

- 🌍 **Single Multilingual Model:** Transcribes 40 language-locales from one model through language-ID prompt conditioning, with optional automatic language detection.
- ⚡ **Native Streaming Architecture:** Cache-aware design enables efficient processing of continuous audio streams, designed and optimized for low-latency voice agent applications.
- 💰 **Improved Operational Efficiency:** Delivers superior throughput compared to traditional buffered streaming approaches. This allows for a higher number of parallel streams within the same GPU memory constraints, directly reducing operational costs for production environments.
- 🎛️ **Dynamic Runtime Flexibility:** Choose the optimal operating point on the latency-accuracy Pareto curve at inference time. No re-training is required to adjust for different use-case requirements.
- 📝 **Punctuation & Capitalization:** Built-in support for punctuation and capitalization in output text.

---

## Supported Languages

The model supports **40 language-locales** in total, across three tiers:

- **Transcription-ready (19 locales):** highest-accuracy ASR, ready out of the box.
- **Broad-coverage (13 locales):** production ASR across an additional 13 locales.
- **Adaptation-ready (8 locales):** recognized by the tokenizer; fine-tune on in-domain data to unlock full transcription.

| Tier | Languages (locales) |
| :--- | :--- |
| **Transcription-ready (19 locales)** | English (en-US, en-GB), Spanish (es-US, es-ES), French (fr-FR, fr-CA), Italian (it-IT), Portuguese (pt-BR, pt-PT), Dutch (nl-NL), German (de-DE), Turkish (tr-TR), Russian (ru-RU), Arabic (ar-AR), Hindi (hi-IN), Japanese (ja-JP), Korean (ko-KR), Vietnamese (vi-VN), Ukrainian (uk-UA) |
| **Broad-coverage (13 locales)** | Polish (pl-PL), Swedish (sv-SE), Czech (cs-CZ), Norwegian Bokmål (nb-NO), Danish (da-DK), Bulgarian (bg-BG), Finnish (fi-FI), Croatian (hr-HR), Slovak (sk-SK), Mandarin (zh-CN), Hungarian (hu-HU), Romanian (ro-RO), Estonian (et-EE) |
| **Adaptation-ready (8 locales)** | Greek (el-GR), Lithuanian (lt-LT), Latvian (lv-LV), Maltese (mt-MT), Slovenian (sl-SI), Hebrew (he-IL), Thai (th-TH), Norwegian Nynorsk (nn-NO) |

> **Note:** Transcription-ready and broad-coverage locales (**32 total**) produce ASR transcription out of the box; adaptation-ready locales require fine-tuning on in-domain data to enable full transcription. The model supports uppercase and lowercase letters, punctuation, spaces, and apostrophes.

> **Note:** We would recommend [Nemotron ASR Streaming (English)](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) model for English-only transcription use cases. For all other transcription ready locales, we recommend Nemotron 3.5 ASR to leverage its expanded multilingual capabilities.

> [!Tip]
> **Automatic language detection / language tagging:** When run with `target_lang=auto`, the model detects the spoken language and emits the corresponding **language code/tag** in the output following the terminal punctuation. This lets a single deployment transcribe mixed-language traffic and automatically label each utterance with its detected language — no separate language-ID component required.

---

## Model Architecture

**Architecture Type:** FastConformer-CacheAware-RNNT with Prompt

This model consists of a cache-aware streaming Parakeet (FastConformer) encoder with an RNN-T decoder and language-ID prompt conditioning. It is based on the Cache-Aware [\[1\]](#ref-1) FastConformer [\[2\]](#ref-2) architecture with 24 encoder layers and an RNNT (Recurrent Neural Network Transducer) decoder. The cache-aware streaming design enables efficient processing of audio in chunks while maintaining context from previous frames. Unlike buffered inference, this model maintains caches for all encoder self-attention and convolution layers. This enables reuse of hidden states at every streaming step, where cached activations eliminate redundant computations. As a result, there are no overlapping computations; each processed frame is strictly non-overlapping. This model leverages prompts to guide the transcription process, enabling language-specific transcription from a single ASR model through language ID conditioning.

<p align="center">
  <img src="model_architecture.png" alt="Nemotron 3.5 ASR architecture: FastConformer encoder and language-ID encoding are concatenated, projected, and fed to the RNNT decoder" width="900"/>
</p>

The language-ID prompt is fused with the acoustic representation as follows:

- **FastConformer encoder** processes audio into an acoustic embedding of shape (D=512, T).
- **Language Encoding** expands a 128-dim one-hot language vector across the time axis → (K=128, T), broadcasting the language identity to every frame.
- **Concatenation** along the feature axis → fused tensor (D + K, T).
- **Projection layer** maps the fused features to the RNNT decoder.

**Network Architecture:**
- Encoder: Cache-Aware FastConformer with 24 layers
- Decoder: RNNT (Recurrent Neural Network Transducer)
- Parameters: 600M

**This model was developed based on [nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b).**

---

## Results at a Glance

ASR performance is measured using Word Error Rate (WER) on the **FLEURS** test sets. Accuracy stays strong across both modes and improves as the chunk size grows, while remaining competitive even at the lowest-latency 80ms setting. Full tables are in [Performance](#performance).

<p align="center">
  <img src="fleurs_wer_vs_chunk_size.png" alt="FLEURS average WER vs streaming chunk size (LangID vs Auto-detect)" width="900"/>
</p>

<p align="center">
  <img src="fleurs_langid_vs_auto.png" alt="FLEURS WER by language: LangID vs Auto-detect at 320ms chunk" width="900"/>
</p>

> **Note:** Japanese and Korean are measured using Character Error Rate (CER) rather than WER, as is standard for these languages.

---

## Throughput & Efficiency

Despite being **roughly half the size** (0.6B vs. 1.1B), Nemotron 3.5 ASR serves **far more concurrent streams at far lower latency** than the [Parakeet RNNT 1.1B multilingual model](https://build.nvidia.com/nvidia/parakeet-1_1b-rnnt-multilingual-asr), which runs on buffered streaming. The cache-aware streaming design avoids the redundant recomputation of buffered inference, so a single H100 can sustain dramatically higher concurrency at every chunk size — directly lowering the cost per stream in production. At the lowest-latency 80ms setting, Nemotron sustains **~17× more concurrent streams** (240 vs. 14); at the 1120ms setting it sustains **6× more** (2,400 vs. 400). The latency-vs-concurrency curves tell the same story: Nemotron (solid green) holds low final-token latency well past 1,000 parallel requests, while Parakeet RNNT 1.1B (dashed blue) saturates after only a few hundred.

<p align="center">
  <img src="throughput_vs_chunk.png" alt="Concurrent streams supported on a single H100: Nemotron ASR streaming vs Parakeet RNNT, across chunk sizes" width="900"/>
</p>

<p align="center">
  <img src="latency_vs_parallel.png" alt="Median final-token latency vs number of parallel requests on a single H100, Nemotron vs Parakeet RNNT across chunk sizes" width="900"/>
</p>

> Measured on a single NVIDIA H100. Throughput is the number of real-time streams sustainable in parallel; latency is the median final-token latency at a given level of concurrency.

---

## Explore more from NVIDIA

For documentation, deployment guides, enterprise-ready APIs, and the latest open models—including Nemotron and other cutting-edge speech, translation, and generative AI—visit the NVIDIA Developer Portal at [developer.nvidia.com](https://developer.nvidia.com/).
Join the community to access tools, support, and resources to accelerate your development with NVIDIA's NeMo, Speech NIM, and foundation models.

- What is [Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/)?
- NVIDIA Developer [Nemotron](https://developer.nvidia.com/nemotron)
- [NVIDIA Speech NIM](https://docs.nvidia.com/nim/speech/latest/about/index.html)
- [NeMo Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html)

Also, check out the following NVIDIA speech models:
- Nemotron ASR Streaming (English) (Nemotron 3 ASR) - https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
- Multitalker Parakeet Streaming - https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1
- Parakeet Realtime EOU - https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1

---

## NVIDIA NeMo

To train, fine-tune or perform inference with this model, you will need to install [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) [\[4\]](#ref-4). We recommend you install it after you've installed Cython and latest PyTorch version.

```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython packaging
pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
```

## How to Use this Model

The model is available for use in the NeMo Framework, and can be used as a pre-trained checkpoint for inference or for fine-tuning on another dataset.

### Loading the Model

```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/nemotron-3.5-asr-streaming-0.6b")
```

### Streaming Inference

You can use the cache-aware streaming inference script from NeMo - [NeMo/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py](https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py)

This is a prompt-conditioned multilingual model: pass the target language with `target_lang` (e.g. `en-US`, `es-ES`, `de-DE`), or use `target_lang=auto` for automatic language detection.

```bash
cd NeMo
python examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
    model_path=<model_path> \
    dataset_manifest=<dataset_manifest> \
    batch_size=<batch_size> \
    target_lang=<lang_id> \ #language key (e.g. en-US) or "auto" for automatic language detection
    att_context_size="[56,13]" \ #set the second value to the desired right context from {0,1,3,6,13}
    strip_lang_tags=true \ #true: remove the detected language tag from the text; false: keep it in the output
    output_path=<output_folder>
```

**`strip_lang_tags`** controls how the detected language tag is handled in the output. The model appends a language tag (e.g. `<en-US>`) after the transcript's terminal punctuation:
- `strip_lang_tags=false` (keep): the tag is left in the output, so you can read the detected language directly from each utterance — useful for mixed-language traffic and language labeling.
- `strip_lang_tags=true` (remove): the tag is stripped, leaving only the clean transcript text — useful when you only need the spoken words.

### Setting up Streaming Configuration

Latency is defined by the `att_context_size` param, where att_context_size = `{num_frames_left_context, num_frame_right_context}`, all measured in **80ms frames**:

* [56, 0]: Chunk size = 1 (1 × 80ms = 0.08s)
* [56, 1]: Chunk size = 2 (2 × 80ms = 0.16s)
* [56, 3]: Chunk size = 4 (4 × 80ms = 0.32s)
* [56, 6]: Chunk size = 7 (7 × 80ms = 0.56s)
* [56, 13]: Chunk size = 14 (14 × 80ms = 1.12s)

Here, chunk size = current frame + right context; each chunk is processed in non-overlapping fashion.

### Input(s): <br>

**Input Type(s):** Audio, Lang ID <br>

**Input Format(s):** wav, string <br>

**Input Parameters:** One-Dimensional (1D) for audio and One-Dimensional (1D) for Lang ID <br>

**Other Properties Related to Input:** Maximum Length in seconds specific to GPU Memory, No Pre-Processing Needed, Mono channel is required. 

By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br>

### Output

**Output Type(s):** Text String in Input Language <br>

**Output Format(s):** String <br>

**Output Parameters:** One-Dimensional (1D) <br>

**Other Properties Related to Output:** No Maximum Character Length, transcribe punctuation and capitalization. 

By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br>

---

## Software Integration

**Runtime Engine:** NeMo 26.06

**Supported Hardware Microarchitecture Compatibility:**
- NVIDIA Ampere
- NVIDIA Blackwell
- NVIDIA Hopper
- NVIDIA Jetson
- NVIDIA Lovelace
- NVIDIA Turing
- NVIDIA Volta

**Supported Operating System(s):**
* Linux <br>
* Linux 4 Tegra <br>

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.<vr>



---


## Model Version(s):
nemotron-3.5-asr-streaming-0.6b-v1 <br>

## Training and Evaluation Datasets:

### Training Datasets

It was trained on speech data across 40 language-locales. The training data is a dynamic blend of public and proprietary internal datasets normalized to have spoken forms in text with punctuation and capitalization, including:


- NVIDIA Riva multilingual ASR training set (Proprietary)
- NVIDIA Granary [\[3\]](#ref-3)
- Multilingual LibriSpeech (MLS)
- Mozilla Common Voice
- FLEURS
- VoxPopuli / Europarl-ASR

** Data Modality: Audio <br>

** Audio Training Data Size: 10,000 to 1 Million Hours <br>

** Data Collection Method by dataset <br>
* Human <br>

** Labeling Method by dataset <br>
* Human <br>
* Synthetic: Synthetic labels were generated from an ensemble of ASR models ([NVIDIA Canary](https://build.nvidia.com/nvidia/canary-1b-asr), [Parakeet Multilingual 1.1B RNNT](https://build.nvidia.com/nvidia/parakeet-1_1b-rnnt-multilingual-asr), [Parakeet CTC 1.1B](https://build.nvidia.com/nvidia/parakeet-ctc-1_1b-asr), [OpenAI Whisper](https://huggingface.co/openai/whisper-large-v3), and [FunASR](https://github.com/modelscope/FunASR)), with punctuation and capitalization (PnC) generated from [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B).




### Evaluation Datasets

The model was evaluated on multilingual ASR benchmarks:

- FLEURS
- Mozilla Common Voice (MCV)
- Multilingual LibriSpeech (MLS)
- NVIDIA internal multilingual evaluation sets

** Data Collection Method by dataset <br>
* Human <br>

** Labeling Method by dataset <br>
* Human <br>

---

## Performance

ASR performance is measured using the Word Error Rate (WER). The tables below report WER (%) on the **FLEURS** test sets across configurable streaming chunk sizes, in two modes:
- **Language Input (LangID):** the target language is provided to the model.
- **Auto-detect:** the model automatically detects the spoken language.

> **Note:** Japanese, Korean, and Mandarin are evaluated using Character Error Rate (CER) rather than WER, as is standard for these languages.
> **Note on text normalization:** WER/CER are computed after text normalization that aligns the reference and hypothesis (e.g., casing, punctuation, numerals, and formatting conventions). Normalization is not perfect across all 40 language-locales, and residual mismatches between normalized text can inflate the reported error rates — actual transcription quality may be somewhat better than the numbers suggest.

### Transcription-ready (19 locales)

_Languages are ordered by accuracy (lowest WER first)._

<table>
  <thead>
    <tr><th rowspan="2" align="left">Language</th><th colspan="5" align="center" style="background-color:#76b900;color:#ffffff">Language Input (LangID)</th><th colspan="5" align="center" style="background-color:#6b7280;color:#ffffff;border-left:2px solid #cbd5e1;">Auto-detect</th></tr>
    <tr><th align="center" style="background-color:#eef6e0">80ms</th><th align="center" style="background-color:#eef6e0">160ms</th><th align="center" style="background-color:#eef6e0">320ms</th><th align="center" style="background-color:#eef6e0">560ms</th><th align="center" style="background-color:#eef6e0">1.12s</th><th align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">80ms</th><th align="center" style="background-color:#f3f4f6;">160ms</th><th align="center" style="background-color:#f3f4f6;">320ms</th><th align="center" style="background-color:#f3f4f6;">560ms</th><th align="center" style="background-color:#f3f4f6;">1.12s</th></tr>
  </thead>
  <tbody>
    <tr><td align="left">Spanish (es-US, es-ES)</td><td align="center" style="background-color:#eef6e0;">4.87</td><td align="center" style="background-color:#eef6e0;">4.64</td><td align="center" style="background-color:#eef6e0;">4.39</td><td align="center" style="background-color:#eef6e0;">4.26</td><td align="center" style="background-color:#eef6e0;">4.11</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">5.04</td><td align="center" style="background-color:#f3f4f6;">4.82</td><td align="center" style="background-color:#f3f4f6;">4.48</td><td align="center" style="background-color:#f3f4f6;">4.34</td><td align="center" style="background-color:#f3f4f6;">4.13</td></tr>
    <tr><td align="left">Italian (it-IT)</td><td align="center" style="background-color:#eef6e0;">5.23</td><td align="center" style="background-color:#eef6e0;">4.85</td><td align="center" style="background-color:#eef6e0;">4.83</td><td align="center" style="background-color:#eef6e0;">4.41</td><td align="center" style="background-color:#eef6e0;">4.25</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">5.28</td><td align="center" style="background-color:#f3f4f6;">4.89</td><td align="center" style="background-color:#f3f4f6;">4.84</td><td align="center" style="background-color:#f3f4f6;">4.47</td><td align="center" style="background-color:#f3f4f6;">4.32</td></tr>
    <tr><td align="left">Portuguese (pt-BR, pt-PT)</td><td align="center" style="background-color:#eef6e0;">6.29</td><td align="center" style="background-color:#eef6e0;">6.10</td><td align="center" style="background-color:#eef6e0;">5.81</td><td align="center" style="background-color:#eef6e0;">5.65</td><td align="center" style="background-color:#eef6e0;">5.48</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">6.41</td><td align="center" style="background-color:#f3f4f6;">6.19</td><td align="center" style="background-color:#f3f4f6;">5.82</td><td align="center" style="background-color:#f3f4f6;">5.57</td><td align="center" style="background-color:#f3f4f6;">5.47</td></tr>
    <tr><td align="left">Hindi (hi-IN)</td><td align="center" style="background-color:#eef6e0;">8.13</td><td align="center" style="background-color:#eef6e0;">7.97</td><td align="center" style="background-color:#eef6e0;">7.41</td><td align="center" style="background-color:#eef6e0;">7.05</td><td align="center" style="background-color:#eef6e0;">6.81</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">11.47</td><td align="center" style="background-color:#f3f4f6;">10.83</td><td align="center" style="background-color:#f3f4f6;">9.88</td><td align="center" style="background-color:#f3f4f6;">9.26</td><td align="center" style="background-color:#f3f4f6;">8.23</td></tr>
    <tr><td align="left">Korean (ko-KR)</td><td align="center" style="background-color:#eef6e0;">7.59</td><td align="center" style="background-color:#eef6e0;">7.70</td><td align="center" style="background-color:#eef6e0;">7.27</td><td align="center" style="background-color:#eef6e0;">7.18</td><td align="center" style="background-color:#eef6e0;">7.12</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">8.31</td><td align="center" style="background-color:#f3f4f6;">8.18</td><td align="center" style="background-color:#f3f4f6;">7.81</td><td align="center" style="background-color:#f3f4f6;">7.49</td><td align="center" style="background-color:#f3f4f6;">7.30</td></tr>
    <tr><td align="left">English (en-US, en-GB)</td><td align="center" style="background-color:#eef6e0;">9.43</td><td align="center" style="background-color:#eef6e0;">8.88</td><td align="center" style="background-color:#eef6e0;">8.27</td><td align="center" style="background-color:#eef6e0;">7.99</td><td align="center" style="background-color:#eef6e0;">7.91</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">9.72</td><td align="center" style="background-color:#f3f4f6;">9.34</td><td align="center" style="background-color:#f3f4f6;">8.84</td><td align="center" style="background-color:#f3f4f6;">8.80</td><td align="center" style="background-color:#f3f4f6;">8.84</td></tr>
    <tr><td align="left">German (de-DE)</td><td align="center" style="background-color:#eef6e0;">9.81</td><td align="center" style="background-color:#eef6e0;">9.21</td><td align="center" style="background-color:#eef6e0;">8.83</td><td align="center" style="background-color:#eef6e0;">8.42</td><td align="center" style="background-color:#eef6e0;">8.31</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">9.90</td><td align="center" style="background-color:#f3f4f6;">9.37</td><td align="center" style="background-color:#f3f4f6;">8.87</td><td align="center" style="background-color:#f3f4f6;">8.58</td><td align="center" style="background-color:#f3f4f6;">8.22</td></tr>
    <tr><td align="left">French (fr-FR, fr-CA)</td><td align="center" style="background-color:#eef6e0;">10.97</td><td align="center" style="background-color:#eef6e0;">10.60</td><td align="center" style="background-color:#eef6e0;">9.79</td><td align="center" style="background-color:#eef6e0;">9.45</td><td align="center" style="background-color:#eef6e0;">9.03</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">11.03</td><td align="center" style="background-color:#f3f4f6;">10.60</td><td align="center" style="background-color:#f3f4f6;">9.84</td><td align="center" style="background-color:#f3f4f6;">9.46</td><td align="center" style="background-color:#f3f4f6;">9.02</td></tr>
    <tr><td align="left">Russian (ru-RU)</td><td align="center" style="background-color:#eef6e0;">10.84</td><td align="center" style="background-color:#eef6e0;">10.73</td><td align="center" style="background-color:#eef6e0;">9.87</td><td align="center" style="background-color:#eef6e0;">9.60</td><td align="center" style="background-color:#eef6e0;">9.17</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">12.47</td><td align="center" style="background-color:#f3f4f6;">12.09</td><td align="center" style="background-color:#f3f4f6;">11.01</td><td align="center" style="background-color:#f3f4f6;">10.57</td><td align="center" style="background-color:#f3f4f6;">10.03</td></tr>
    <tr><td align="left">Turkish (tr-TR)</td><td align="center" style="background-color:#eef6e0;">12.34</td><td align="center" style="background-color:#eef6e0;">12.33</td><td align="center" style="background-color:#eef6e0;">12.05</td><td align="center" style="background-color:#eef6e0;">11.34</td><td align="center" style="background-color:#eef6e0;">11.17</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">12.61</td><td align="center" style="background-color:#f3f4f6;">12.28</td><td align="center" style="background-color:#f3f4f6;">11.93</td><td align="center" style="background-color:#f3f4f6;">11.51</td><td align="center" style="background-color:#f3f4f6;">11.32</td></tr>
    <tr><td align="left">Vietnamese (vi-VN)</td><td align="center" style="background-color:#eef6e0;">13.41</td><td align="center" style="background-color:#eef6e0;">12.87</td><td align="center" style="background-color:#eef6e0;">12.29</td><td align="center" style="background-color:#eef6e0;">11.78</td><td align="center" style="background-color:#eef6e0;">11.18</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">13.59</td><td align="center" style="background-color:#f3f4f6;">13.02</td><td align="center" style="background-color:#f3f4f6;">12.40</td><td align="center" style="background-color:#f3f4f6;">12.02</td><td align="center" style="background-color:#f3f4f6;">11.22</td></tr>
    <tr><td align="left">Dutch (nl-NL)</td><td align="center" style="background-color:#eef6e0;">14.03</td><td align="center" style="background-color:#eef6e0;">13.43</td><td align="center" style="background-color:#eef6e0;">12.17</td><td align="center" style="background-color:#eef6e0;">11.97</td><td align="center" style="background-color:#eef6e0;">11.46</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">14.09</td><td align="center" style="background-color:#f3f4f6;">13.80</td><td align="center" style="background-color:#f3f4f6;">12.62</td><td align="center" style="background-color:#f3f4f6;">12.24</td><td align="center" style="background-color:#f3f4f6;">11.70</td></tr>
    <tr><td align="left">Japanese (ja-JP)</td><td align="center" style="background-color:#eef6e0;">13.87</td><td align="center" style="background-color:#eef6e0;">12.90</td><td align="center" style="background-color:#eef6e0;">12.22</td><td align="center" style="background-color:#eef6e0;">11.91</td><td align="center" style="background-color:#eef6e0;">11.48</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">14.97</td><td align="center" style="background-color:#f3f4f6;">13.85</td><td align="center" style="background-color:#f3f4f6;">13.00</td><td align="center" style="background-color:#f3f4f6;">12.38</td><td align="center" style="background-color:#f3f4f6;">11.66</td></tr>
    <tr><td align="left">Arabic (ar-AR)</td><td align="center" style="background-color:#eef6e0;">13.17</td><td align="center" style="background-color:#eef6e0;">12.65</td><td align="center" style="background-color:#eef6e0;">12.55</td><td align="center" style="background-color:#eef6e0;">12.13</td><td align="center" style="background-color:#eef6e0;">12.03</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">13.47</td><td align="center" style="background-color:#f3f4f6;">12.85</td><td align="center" style="background-color:#f3f4f6;">12.67</td><td align="center" style="background-color:#f3f4f6;">12.18</td><td align="center" style="background-color:#f3f4f6;">12.06</td></tr>
    <tr><td align="left">Ukrainian (uk-UA)</td><td align="center" style="background-color:#eef6e0;">15.70</td><td align="center" style="background-color:#eef6e0;">15.21</td><td align="center" style="background-color:#eef6e0;">14.55</td><td align="center" style="background-color:#eef6e0;">13.67</td><td align="center" style="background-color:#eef6e0;">13.07</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">18.81</td><td align="center" style="background-color:#f3f4f6;">17.96</td><td align="center" style="background-color:#f3f4f6;">16.79</td><td align="center" style="background-color:#f3f4f6;">15.60</td><td align="center" style="background-color:#f3f4f6;">14.59</td></tr>
    <tr><td align="left"><strong>Average</strong></td><td align="center" style="background-color:#eef6e0;"><strong>10.38</strong></td><td align="center" style="background-color:#eef6e0;"><strong>10.00</strong></td><td align="center" style="background-color:#eef6e0;"><strong>9.49</strong></td><td align="center" style="background-color:#eef6e0;"><strong>9.12</strong></td><td align="center" style="background-color:#eef6e0;"><strong>8.84</strong></td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;"><strong>11.14</strong></td><td align="center" style="background-color:#f3f4f6;"><strong>10.67</strong></td><td align="center" style="background-color:#f3f4f6;"><strong>10.05</strong></td><td align="center" style="background-color:#f3f4f6;"><strong>9.63</strong></td><td align="center" style="background-color:#f3f4f6;"><strong>9.21</strong></td></tr>
  </tbody>
</table>

### Broad-coverage (13 locales)

_Languages are ordered by accuracy (lowest WER first)._

<table>
  <thead>
    <tr><th rowspan="2" align="left">Language</th><th colspan="5" align="center" style="background-color:#76b900;color:#ffffff">Language Input (LangID)</th><th colspan="5" align="center" style="background-color:#6b7280;color:#ffffff;border-left:2px solid #cbd5e1;">Auto-detect</th></tr>
    <tr><th align="center" style="background-color:#eef6e0">80ms</th><th align="center" style="background-color:#eef6e0">160ms</th><th align="center" style="background-color:#eef6e0">320ms</th><th align="center" style="background-color:#eef6e0">560ms</th><th align="center" style="background-color:#eef6e0">1.12s</th><th align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">80ms</th><th align="center" style="background-color:#f3f4f6;">160ms</th><th align="center" style="background-color:#f3f4f6;">320ms</th><th align="center" style="background-color:#f3f4f6;">560ms</th><th align="center" style="background-color:#f3f4f6;">1.12s</th></tr>
  </thead>
  <tbody>
    <tr><td align="left">Polish (pl-PL)</td><td align="center" style="background-color:#eef6e0;">19.88</td><td align="center" style="background-color:#eef6e0;">18.92</td><td align="center" style="background-color:#eef6e0;">17.48</td><td align="center" style="background-color:#eef6e0;">16.61</td><td align="center" style="background-color:#eef6e0;">15.15</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">22.65</td><td align="center" style="background-color:#f3f4f6;">21.63</td><td align="center" style="background-color:#f3f4f6;">20.05</td><td align="center" style="background-color:#f3f4f6;">18.52</td><td align="center" style="background-color:#f3f4f6;">16.55</td></tr>
    <tr><td align="left">Norwegian Bokmål (nb-NO)</td><td align="center" style="background-color:#eef6e0;">20.43</td><td align="center" style="background-color:#eef6e0;">20.07</td><td align="center" style="background-color:#eef6e0;">18.90</td><td align="center" style="background-color:#eef6e0;">18.44</td><td align="center" style="background-color:#eef6e0;">18.10</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">20.91</td><td align="center" style="background-color:#f3f4f6;">20.19</td><td align="center" style="background-color:#f3f4f6;">19.29</td><td align="center" style="background-color:#f3f4f6;">18.76</td><td align="center" style="background-color:#f3f4f6;">18.01</td></tr>
    <tr><td align="left">Finnish (fi-FI)</td><td align="center" style="background-color:#eef6e0;">21.19</td><td align="center" style="background-color:#eef6e0;">20.57</td><td align="center" style="background-color:#eef6e0;">20.05</td><td align="center" style="background-color:#eef6e0;">18.94</td><td align="center" style="background-color:#eef6e0;">18.34</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">21.61</td><td align="center" style="background-color:#f3f4f6;">20.88</td><td align="center" style="background-color:#f3f4f6;">20.40</td><td align="center" style="background-color:#f3f4f6;">19.36</td><td align="center" style="background-color:#f3f4f6;">18.72</td></tr>
    <tr><td align="left">Mandarin (zh-CN)</td><td align="center" style="background-color:#eef6e0;">20.56</td><td align="center" style="background-color:#eef6e0;">20.22</td><td align="center" style="background-color:#eef6e0;">20.03</td><td align="center" style="background-color:#eef6e0;">19.51</td><td align="center" style="background-color:#eef6e0;">19.28</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">22.45</td><td align="center" style="background-color:#f3f4f6;">21.07</td><td align="center" style="background-color:#f3f4f6;">20.59</td><td align="center" style="background-color:#f3f4f6;">20.40</td><td align="center" style="background-color:#f3f4f6;">19.87</td></tr>
    <tr><td align="left">Czech (cs-CZ)</td><td align="center" style="background-color:#eef6e0;">24.18</td><td align="center" style="background-color:#eef6e0;">23.20</td><td align="center" style="background-color:#eef6e0;">22.41</td><td align="center" style="background-color:#eef6e0;">21.04</td><td align="center" style="background-color:#eef6e0;">20.41</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">25.81</td><td align="center" style="background-color:#f3f4f6;">25.12</td><td align="center" style="background-color:#f3f4f6;">23.68</td><td align="center" style="background-color:#f3f4f6;">22.55</td><td align="center" style="background-color:#f3f4f6;">21.45</td></tr>
    <tr><td align="left">Bulgarian (bg-BG)</td><td align="center" style="background-color:#eef6e0;">24.50</td><td align="center" style="background-color:#eef6e0;">23.58</td><td align="center" style="background-color:#eef6e0;">22.80</td><td align="center" style="background-color:#eef6e0;">21.70</td><td align="center" style="background-color:#eef6e0;">20.53</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">28.28</td><td align="center" style="background-color:#f3f4f6;">27.22</td><td align="center" style="background-color:#f3f4f6;">25.54</td><td align="center" style="background-color:#f3f4f6;">24.05</td><td align="center" style="background-color:#f3f4f6;">21.84</td></tr>
    <tr><td align="left">Slovak (sk-SK)</td><td align="center" style="background-color:#eef6e0;">25.08</td><td align="center" style="background-color:#eef6e0;">24.14</td><td align="center" style="background-color:#eef6e0;">23.73</td><td align="center" style="background-color:#eef6e0;">22.51</td><td align="center" style="background-color:#eef6e0;">21.28</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">27.59</td><td align="center" style="background-color:#f3f4f6;">26.06</td><td align="center" style="background-color:#f3f4f6;">25.61</td><td align="center" style="background-color:#f3f4f6;">24.15</td><td align="center" style="background-color:#f3f4f6;">22.68</td></tr>
    <tr><td align="left">Swedish (sv-SE)</td><td align="center" style="background-color:#eef6e0;">25.61</td><td align="center" style="background-color:#eef6e0;">24.85</td><td align="center" style="background-color:#eef6e0;">23.63</td><td align="center" style="background-color:#eef6e0;">22.72</td><td align="center" style="background-color:#eef6e0;">22.17</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">26.28</td><td align="center" style="background-color:#f3f4f6;">25.56</td><td align="center" style="background-color:#f3f4f6;">24.18</td><td align="center" style="background-color:#f3f4f6;">23.57</td><td align="center" style="background-color:#f3f4f6;">22.53</td></tr>
    <tr><td align="left">Croatian (hr-HR)</td><td align="center" style="background-color:#eef6e0;">27.92</td><td align="center" style="background-color:#eef6e0;">27.09</td><td align="center" style="background-color:#eef6e0;">25.79</td><td align="center" style="background-color:#eef6e0;">24.92</td><td align="center" style="background-color:#eef6e0;">23.97</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">32.13</td><td align="center" style="background-color:#f3f4f6;">31.20</td><td align="center" style="background-color:#f3f4f6;">29.65</td><td align="center" style="background-color:#f3f4f6;">28.95</td><td align="center" style="background-color:#f3f4f6;">27.46</td></tr>
    <tr><td align="left">Romanian (ro-RO)</td><td align="center" style="background-color:#eef6e0;">31.52</td><td align="center" style="background-color:#eef6e0;">30.93</td><td align="center" style="background-color:#eef6e0;">29.04</td><td align="center" style="background-color:#eef6e0;">27.77</td><td align="center" style="background-color:#eef6e0;">25.90</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">34.22</td><td align="center" style="background-color:#f3f4f6;">33.26</td><td align="center" style="background-color:#f3f4f6;">30.97</td><td align="center" style="background-color:#f3f4f6;">29.84</td><td align="center" style="background-color:#f3f4f6;">26.88</td></tr>
    <tr><td align="left">Estonian (et-EE)</td><td align="center" style="background-color:#eef6e0;">29.95</td><td align="center" style="background-color:#eef6e0;">29.66</td><td align="center" style="background-color:#eef6e0;">28.59</td><td align="center" style="background-color:#eef6e0;">27.37</td><td align="center" style="background-color:#eef6e0;">26.35</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">30.58</td><td align="center" style="background-color:#f3f4f6;">30.09</td><td align="center" style="background-color:#f3f4f6;">28.72</td><td align="center" style="background-color:#f3f4f6;">28.03</td><td align="center" style="background-color:#f3f4f6;">27.19</td></tr>
    <tr><td align="left">Danish (da-DK)</td><td align="center" style="background-color:#eef6e0;">32.62</td><td align="center" style="background-color:#eef6e0;">31.51</td><td align="center" style="background-color:#eef6e0;">30.00</td><td align="center" style="background-color:#eef6e0;">28.92</td><td align="center" style="background-color:#eef6e0;">27.49</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">33.15</td><td align="center" style="background-color:#f3f4f6;">31.77</td><td align="center" style="background-color:#f3f4f6;">30.22</td><td align="center" style="background-color:#f3f4f6;">29.33</td><td align="center" style="background-color:#f3f4f6;">27.81</td></tr>
    <tr><td align="left">Hungarian (hu-HU)</td><td align="center" style="background-color:#eef6e0;">32.70</td><td align="center" style="background-color:#eef6e0;">32.03</td><td align="center" style="background-color:#eef6e0;">30.92</td><td align="center" style="background-color:#eef6e0;">29.72</td><td align="center" style="background-color:#eef6e0;">28.68</td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;">33.40</td><td align="center" style="background-color:#f3f4f6;">32.39</td><td align="center" style="background-color:#f3f4f6;">31.49</td><td align="center" style="background-color:#f3f4f6;">30.20</td><td align="center" style="background-color:#f3f4f6;">29.18</td></tr>
    <tr><td align="left"><strong>Average</strong></td><td align="center" style="background-color:#eef6e0;"><strong>25.86</strong></td><td align="center" style="background-color:#eef6e0;"><strong>25.14</strong></td><td align="center" style="background-color:#eef6e0;"><strong>24.11</strong></td><td align="center" style="background-color:#eef6e0;"><strong>23.09</strong></td><td align="center" style="background-color:#eef6e0;"><strong>22.13</strong></td><td align="center" style="background-color:#f3f4f6;border-left:2px solid #cbd5e1;"><strong>27.62</strong></td><td align="center" style="background-color:#f3f4f6;"><strong>26.65</strong></td><td align="center" style="background-color:#f3f4f6;"><strong>25.41</strong></td><td align="center" style="background-color:#f3f4f6;"><strong>24.44</strong></td><td align="center" style="background-color:#f3f4f6;"><strong>23.09</strong></td></tr>
  </tbody>
</table>

### Adaptation-ready languages (fine-tune to enable)

These **8 language-locales** are recognized by the tokenizer but are not tuned for production transcription out of the box: **Greek (el-GR), Hebrew (he-IL), Lithuanian (lt-LT), Slovenian (sl-SI), Latvian (lv-LV), Maltese (mt-MT), Thai (th-TH), and Norwegian Nynorsk (nn-NO)**. Fine-tuning on in-domain data is recommended to bring them to production quality.

Check our [blog post](https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr) of **how to fine-tune Nemotron 3.5 ASR to improve these languages**, including before/after results.


---

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

---
