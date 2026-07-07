"""Streaming client for the Triton `nemotron_asr_streaming` model.

Splits an audio file into fixed chunks and sends them as one Triton sequence
(same sequence_id), printing the transcript as it grows in real time.

    python streaming_client.py --audio sample1.flac --lang en-US
    python streaming_client.py --audio clip.wav --chunk-ms 320 --url localhost:8000
"""
import argparse
import random
import time

import numpy as np
import soundfile as sf
import tritonclient.http as httpclient


def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
    return audio.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--lang", default="en-US")
    ap.add_argument("--url", default="localhost:8000")
    ap.add_argument("--model", default="nemotron_asr_streaming")
    ap.add_argument("--chunk-ms", type=int, default=320, help="audio chunk size in ms")
    ap.add_argument("--realtime", action="store_true", help="sleep chunk-ms between sends")
    args = ap.parse_args()

    audio = load_audio(args.audio)
    step = int(args.chunk_ms * 16)  # samples per chunk at 16 kHz
    seq_id = random.randint(1, 2**31)
    client = httpclient.InferenceServerClient(url=args.url)

    n = max(1, (len(audio) + step - 1) // step)
    full = ""
    for i in range(n):
        chunk = audio[i * step : (i + 1) * step]
        if chunk.size == 0:
            chunk = np.zeros(1, dtype=np.float32)
        is_start = i == 0
        is_end = i == n - 1

        # model has max_batch_size>0, so every input needs a leading batch dim of 1.
        inputs = [httpclient.InferInput("AUDIO_CHUNK", [1, chunk.shape[0]], "FP32")]
        inputs[0].set_data_from_numpy(chunk.reshape(1, -1))
        if is_start:
            lang_in = httpclient.InferInput("TARGET_LANG", [1, 1], "BYTES")
            lang_in.set_data_from_numpy(
                np.array([[args.lang.encode("utf-8")]], dtype=object)
            )
            inputs.append(lang_in)

        result = client.infer(
            args.model,
            inputs=inputs,
            outputs=[httpclient.InferRequestedOutput("TRANSCRIPT")],
            sequence_id=seq_id,
            sequence_start=is_start,
            sequence_end=is_end,
        )
        delta = result.as_numpy("TRANSCRIPT").reshape(-1)[0]
        if isinstance(delta, bytes):
            delta = delta.decode("utf-8")
        if delta:
            full += delta
            print(f"[{(i + 1) * args.chunk_ms / 1000:5.1f}s] {full}")
        if args.realtime and not is_end:
            time.sleep(args.chunk_ms / 1000.0)

    print("\nFINAL:", full)


if __name__ == "__main__":
    main()
