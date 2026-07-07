"""Standalone validation of the Triton ASR pipeline core (no Triton needed).

Run with the `nemo-trt` conda env (has tensorrt 10.16 + torch + nemo):

    /home/ebcpc10/miniconda3/envs/nemo-trt/bin/python triton/test_pipeline.py \
        --audio <wav/flac> --lang en-US

It:
  1. checks LeanFeaturizer parity against NeMo's AudioToMelSpectrogramPreprocessor
  2. runs the full engine pipeline and prints the transcript
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model_repository", "nemotron_asr", "1")
sys.path.insert(0, MODEL_DIR)
from asr_pipeline import LeanFeaturizer, NemotronASR  # noqa: E402

DEFAULT_ENGINE_DIR = os.path.join(
    os.path.dirname(HERE), "nemotron-asr", "nemotron-3.5-asr-streaming-0.6b", "trt_engines"
)
ASSETS_DIR = os.path.join(MODEL_DIR, "assets")


def check_featurizer_parity(audio_np):
    try:
        import yaml
        from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
    except Exception as e:  # noqa: BLE001
        print(f"[parity] SKIP (NeMo unavailable): {e!r}")
        return
    cfg_path = os.path.join(
        os.path.dirname(HERE), "nemotron-asr", "nemotron-3.5-asr-streaming-0.6b",
        ".nemo_extracted", "model_config.yaml",
    )
    if not os.path.exists(cfg_path):
        print("[parity] SKIP (model_config.yaml not extracted)")
        return
    pcfg = yaml.safe_load(open(cfg_path))["preprocessor"]
    pcfg.pop("_target_", None)
    pcfg["dither"] = 0.0
    pcfg["pad_to"] = 0
    nemo_pre = AudioToMelSpectrogramPreprocessor(**pcfg).cuda().eval()

    audio = torch.as_tensor(audio_np, dtype=torch.float32, device="cuda").reshape(1, -1)
    alen = torch.tensor([audio.shape[1]], dtype=torch.int64, device="cuda")
    with torch.no_grad():
        ref, ref_len = nemo_pre(input_signal=audio, length=alen)

    lean = LeanFeaturizer(os.path.join(ASSETS_DIR, "feat_assets.pt"))
    got, got_len = lean(audio, alen)

    t = min(ref.shape[-1], got.shape[-1])
    diff = (ref[..., :t] - got[..., :t]).abs()
    print(f"[parity] feat shapes ref={tuple(ref.shape)} got={tuple(got.shape)} "
          f"len ref={ref_len.tolist()} got={got_len.tolist()}")
    print(f"[parity] max|diff|={diff.max().item():.3e} mean|diff|={diff.mean().item():.3e}")
    assert diff.max().item() < 1e-3, "featurizer parity check failed"
    print("[parity] OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=os.path.join(
        os.path.dirname(HERE), "nemotron-asr", "nemotron-3.5-asr-streaming-0.6b",
        ".nemo_extracted", "samples", "sample1.flac"))
    ap.add_argument("--lang", default="en-US")
    ap.add_argument("--engine-dir", default=DEFAULT_ENGINE_DIR)
    args = ap.parse_args()

    audio_np, sr = sf.read(args.audio)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    assert sr == 16000, f"expected 16 kHz, got {sr}"
    audio_np = audio_np.astype(np.float32)

    check_featurizer_parity(audio_np)

    asr = NemotronASR(engine_dir=args.engine_dir, assets_dir=ASSETS_DIR)
    text = asr.transcribe(audio_np, target_lang=args.lang)
    print("\n[transcribe] lang=", args.lang)
    print("TRANSCRIPT:\n", text)


if __name__ == "__main__":
    main()
