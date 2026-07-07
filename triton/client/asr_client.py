"""Client for the Triton `nemotron_asr` model.

Examples:
    python asr_client.py --audio sample1.flac --lang en-US
    python asr_client.py --audio clip.wav --lang auto-fallback --url localhost:8000
"""
import argparse

import numpy as np
import soundfile as sf
import tritonclient.http as httpclient


def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        try:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Audio is {sr} Hz; install librosa to auto-resample. {e}")
    return audio.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--lang", default="en-US", help="language code, e.g. en-US, de-DE, zh-CN")
    ap.add_argument("--url", default="localhost:8000")
    ap.add_argument("--model", default="nemotron_asr")
    args = ap.parse_args()

    audio = load_audio(args.audio)

    client = httpclient.InferenceServerClient(url=args.url)

    audio_in = httpclient.InferInput("AUDIO", [audio.shape[0]], "FP32")
    audio_in.set_data_from_numpy(audio)

    lang_in = httpclient.InferInput("TARGET_LANG", [1], "BYTES")
    lang_in.set_data_from_numpy(np.array([args.lang.encode("utf-8")], dtype=object))

    out = httpclient.InferRequestedOutput("TRANSCRIPT")
    result = client.infer(args.model, inputs=[audio_in, lang_in], outputs=[out])

    transcript = result.as_numpy("TRANSCRIPT")[0]
    if isinstance(transcript, bytes):
        transcript = transcript.decode("utf-8")
    print(transcript)


if __name__ == "__main__":
    main()
