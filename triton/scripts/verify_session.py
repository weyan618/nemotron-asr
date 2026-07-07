"""Validate StreamingSession (online, raw-audio fed incrementally) vs offline."""
import difflib
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(HERE), "model_repository", "nemotron_asr", "1")
sys.path.insert(0, MODEL_DIR)
from asr_pipeline import NemotronASR  # noqa: E402

MODEL_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(HERE)), "nemotron-asr", "nemotron-3.5-asr-streaming-0.6b"
)
ENG = os.path.join(MODEL_ROOT, "trt_engines")
ASSETS = os.path.join(MODEL_DIR, "assets")
SAMPLE = os.path.join(MODEL_ROOT, ".nemo_extracted", "samples", "sample1.flac")


def main():
    asr = NemotronASR(engine_dir=ENG, assets_dir=ASSETS)
    audio, sr = sf.read(SAMPLE)
    audio = (audio.mean(1) if audio.ndim > 1 else audio).astype(np.float32)

    offline = asr.transcribe(audio, "en-US")
    print("OFFLINE  :", offline)

    # --- feature parity: incremental _chunk_features vs full-stream featurizer ---
    import torch
    sess0 = asr.new_stream("en-US")
    sess0.raw = torch.from_numpy(audio).to(asr.device)
    sess0.raw_base = 0
    sess0.total = audio.shape[0]
    full_feats, full_len = asr.featurizer(
        torch.from_numpy(audio).to(asr.device).unsqueeze(0),
        torch.tensor([audio.shape[0]], device=asr.device),
    )
    Ft = int(full_len[0].item())
    inc = sess0._chunk_features(0, Ft)               # incremental path, all frames
    diff = (inc[0] - full_feats[0, :, :Ft]).abs().max().item()
    print(f"feature parity max|diff| (incremental vs full) = {diff:.3e}")

    # feed in 320 ms raw chunks (simulating a live mic stream)
    step = int(0.32 * sr)
    sess = asr.new_stream("en-US")
    deltas = []
    for i in range(0, len(audio), step):
        d = sess.add_audio(audio[i : i + step])
        if d:
            deltas.append(d)
    d = sess.finalize()
    if d:
        deltas.append(d)
    streaming = sess.text
    print("STREAMING:", streaming)
    print("\nincremental deltas:")
    print(" | ".join(x.strip() for x in deltas if x.strip()))
    print(f"\nsimilarity(offline, streaming) = {difflib.SequenceMatcher(None, offline, streaming).ratio():.3f}")


if __name__ == "__main__":
    main()
