"""gRPC streaming client for `nemotron_asr_streaming`.

Uses Triton's bidirectional gRPC stream (start_stream + async_stream_infer): all
chunks of one audio stream are sent on the same stream with a shared sequence_id;
responses (incremental transcript deltas) arrive via a callback. This is the
idiomatic low-latency path for real-time ASR.

    python grpc_streaming_client.py --audio sample1.flac --lang en-US --realtime
    python grpc_streaming_client.py --audio clip.wav --chunk-ms 320 --url localhost:8001
"""
import argparse
import queue
import random
import time

import numpy as np
import soundfile as sf
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


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
    ap.add_argument("--url", default="localhost:8001")
    ap.add_argument("--model", default="nemotron_asr_streaming")
    ap.add_argument("--chunk-ms", type=int, default=320)
    ap.add_argument("--realtime", action="store_true", help="sleep chunk-ms between sends")
    args = ap.parse_args()

    audio = load_audio(args.audio)
    step = int(args.chunk_ms * 16)                 # samples per chunk at 16 kHz
    n = max(1, (len(audio) + step - 1) // step)
    seq_id = random.randint(1, 2**31)

    results = queue.Queue()

    def callback(result, error):
        results.put((result, error))

    client = grpcclient.InferenceServerClient(url=args.url)
    client.start_stream(callback=callback)

    # producer: send chunks on the bidirectional stream
    for i in range(n):
        chunk = audio[i * step : (i + 1) * step]
        if chunk.size == 0:
            chunk = np.zeros(1, dtype=np.float32)
        is_start, is_end = i == 0, i == n - 1

        # model has max_batch_size>0, so every input needs a leading batch dim of 1.
        inputs = [grpcclient.InferInput("AUDIO_CHUNK", [1, chunk.shape[0]], "FP32")]
        inputs[0].set_data_from_numpy(chunk.reshape(1, -1))
        if is_start:
            lang_in = grpcclient.InferInput("TARGET_LANG", [1, 1], "BYTES")
            lang_in.set_data_from_numpy(
                np.array([[args.lang.encode("utf-8")]], dtype=object)
            )
            inputs.append(lang_in)

        client.async_stream_infer(
            model_name=args.model,
            inputs=inputs,
            outputs=[grpcclient.InferRequestedOutput("TRANSCRIPT")],
            request_id=str(i),
            sequence_id=seq_id,
            sequence_start=is_start,
            sequence_end=is_end,
        )
        if args.realtime and not is_end:
            time.sleep(args.chunk_ms / 1000.0)

    # consumer: collect exactly n responses (sequence batching preserves order)
    full = ""
    for _ in range(n):
        result, error = results.get()
        if error is not None:
            if isinstance(error, InferenceServerException):
                print("stream error:", error)
            else:
                print("stream error:", repr(error))
            continue
        delta = result.as_numpy("TRANSCRIPT").reshape(-1)[0]
        if isinstance(delta, bytes):
            delta = delta.decode("utf-8")
        if delta:
            full += delta
            print(f"+ {full}")

    client.stop_stream()
    print("\nFINAL:", full)


if __name__ == "__main__":
    main()
