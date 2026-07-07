"""Minimal HTTP demo for `nemotron_asr` using only `requests` (no tritonclient).

This talks to Triton's KServe v2 REST inference API directly, using the
**binary tensor data extension** so the audio travels as raw float32 bytes
instead of a giant JSON number array (much smaller/faster).

Wire format (one HTTP POST):
    [ JSON header ][ AUDIO float32 bytes ][ TARGET_LANG BYTES bytes ]
    └ length given by the `Inference-Header-Content-Length` request header ┘

Run:
    pip install requests soundfile numpy
    python http_demo.py --audio sample1.flac --lang en-US --url http://localhost:8000
"""
import argparse
import json
import struct

import numpy as np
import requests
import soundfile as sf


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path)
    if audio.ndim > 1:                       # stereo -> mono
        audio = audio.mean(axis=1)
    assert sr == 16000, f"expected 16 kHz mono, got {sr} Hz (resample first)"
    return audio.astype("<f4")               # little-endian float32


def pack_bytes_tensor(strings) -> bytes:
    """KServe BYTES layout: per element -> uint32 LE length prefix + raw utf-8."""
    out = b""
    for s in strings:
        b = s.encode("utf-8")
        out += struct.pack("<I", len(b)) + b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--lang", default="en-US", help="e.g. en-US, de-DE, zh-CN")
    ap.add_argument("--url", default="http://localhost:8000", help="HTTP endpoint")
    ap.add_argument("--model", default="nemotron_asr")
    args = ap.parse_args()

    audio = load_audio(args.audio)
    audio_bin = audio.tobytes()
    lang_bin = pack_bytes_tensor([args.lang])

    # 1) Build the JSON header. Inputs declare binary_data_size and omit "data";
    #    the actual bytes are appended after the header in the same order.
    req = {
        "inputs": [
            {"name": "AUDIO", "datatype": "FP32", "shape": [audio.shape[0]],
             "parameters": {"binary_data_size": len(audio_bin)}},
            {"name": "TARGET_LANG", "datatype": "BYTES", "shape": [1],
             "parameters": {"binary_data_size": len(lang_bin)}},
        ],
        "outputs": [
            {"name": "TRANSCRIPT", "parameters": {"binary_data": True}},
        ],
    }
    header = json.dumps(req).encode("utf-8")
    body = header + audio_bin + lang_bin

    # 2) POST. Inference-Header-Content-Length marks where JSON ends / binary starts.
    resp = requests.post(
        f"{args.url}/v2/models/{args.model}/infer",
        data=body,
        headers={
            "Inference-Header-Content-Length": str(len(header)),
            "Content-Type": "application/octet-stream",
        },
        timeout=60,
    )
    resp.raise_for_status()

    # 3) Parse response: first N bytes are JSON metadata, the rest is binary.
    n = int(resp.headers["Inference-Header-Content-Length"])
    meta = json.loads(resp.content[:n])
    binary = resp.content[n:]

    out = meta["outputs"][0]
    raw = binary[: out["parameters"]["binary_data_size"]]
    (strlen,) = struct.unpack_from("<I", raw, 0)   # decode one BYTES element
    print(raw[4 : 4 + strlen].decode("utf-8"))


if __name__ == "__main__":
    main()
