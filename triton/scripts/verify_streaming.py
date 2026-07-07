"""Validate the cache-aware streaming encoder TRT engine end-to-end.

Drives sample audio chunk-by-chunk through:
    NeMo feature buffer (pad_and_drop_preencoded=True, matches the exported graph)
    -> streaming_encoder TRT engine (with cache in/out)
    -> prompt_kernel (language conditioning)
    -> persistent RNNT greedy (decoder_joint engine, state carried across chunks)
and compares the streaming transcript to the known-correct offline transcript.
"""
import os
import sys

import numpy as np
import soundfile as sf
import tensorrt as trt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(HERE), "model_repository", "nemotron_asr", "1")
sys.path.insert(0, MODEL_DIR)
from asr_pipeline import TRTEngine, NemotronASR  # noqa: E402

MODEL_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(HERE)), "nemotron-asr", "nemotron-3.5-asr-streaming-0.6b"
)
ENG = os.path.join(MODEL_ROOT, "trt_engines")
ASSETS = os.path.join(MODEL_DIR, "assets")
SAMPLE = os.path.join(MODEL_ROOT, ".nemo_extracted", "samples", "sample1.flac")
ATT = [56, 3]


def main():
    dev = "cuda"
    # offline reference transcript (already validated correct)
    asr = NemotronASR(engine_dir=ENG, assets_dir=ASSETS)
    audio_np, sr = sf.read(SAMPLE)
    audio_np = (audio_np.mean(1) if audio_np.ndim > 1 else audio_np).astype(np.float32)
    offline = asr.transcribe(audio_np, "en-US")
    print("OFFLINE :", offline[:90])

    # NeMo model only for the streaming feature buffer + initial cache state
    from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt import (
        EncDecHybridRNNTCTCBPEModelWithPrompt,
    )
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    m = EncDecHybridRNNTCTCBPEModelWithPrompt.restore_from(
        os.path.join(MODEL_ROOT, "nemotron-3.5-asr-streaming-0.6b.nemo"),
        map_location="cpu", strict=False,
    ).eval()
    m.encoder.set_default_att_context_size(ATT)
    m.encoder.export_cache_support = True

    buf = CacheAwareStreamingAudioBuffer(m, online_normalization=False, pad_and_drop_preencoded=True)
    buf.append_audio(audio_np, stream_id=-1)

    # initial caches in export layout [B, L, T, D]
    c_chan, c_time, c_len = m.encoder.get_initial_cache_state(batch_size=1, device="cpu")
    c_chan = c_chan.transpose(0, 1).contiguous().to(dev)        # [B,L,Tcache,D]
    c_time = c_time.transpose(0, 1).contiguous().to(dev)        # [B,L,D,Tconv]
    c_len = c_len.to(dev)

    senc = TRTEngine(os.path.join(ENG, "streaming_encoder-nemotron.engine"), trt.Logger(trt.Logger.ERROR))
    prompt_id = asr.resolve_prompt_id("en-US")

    # persistent RNNT greedy state
    state1 = torch.zeros((2, 1, 640), device=dev)
    state2 = torch.zeros((2, 1, 640), device=dev)
    last = asr.blank_id
    tokens = []
    nchunks = 0
    for audio_chunk, chunk_len in buf:
        nchunks += 1
        feats = audio_chunk.to(dev).float()                    # [1,128,Tc]
        flen = chunk_len.to(dev).to(torch.int64)
        out = senc({
            "audio_signal": feats.contiguous(),
            "length": flen.contiguous(),
            "cache_last_channel": c_chan,
            "cache_last_time": c_time,
            "cache_last_channel_len": c_len.to(torch.int64).contiguous(),
        })
        encoded = out["outputs"]                                # [1,1024,~4]
        c_chan = out["cache_last_channel_next"].contiguous()
        c_time = out["cache_last_time_next"].contiguous()
        c_len = out["cache_last_channel_next_len"].contiguous()
        elen = int(out["encoded_lengths"][0].item())
        if elen <= 0:
            continue

        # prompt_kernel
        enc_bt = encoded[:, :, :elen].transpose(1, 2)          # [1,T,1024]
        prompt = torch.zeros((1, enc_bt.shape[1], 128), device=dev)
        prompt[:, :, prompt_id] = 1.0
        enc_p = asr.prompt_kernel(torch.cat([enc_bt, prompt], dim=-1)).transpose(1, 2).contiguous()

        # persistent greedy over the new frames
        for t in range(elen):
            enc_t = enc_p[:, :, t:t + 1].contiguous()
            sym = 0
            while sym < 10:
                tg = torch.tensor([[last]], dtype=torch.int32, device=dev)
                tl = torch.tensor([1], dtype=torch.int32, device=dev)
                o = asr.decoder_joint({"encoder_outputs": enc_t, "targets": tg, "target_length": tl,
                                       "input_states_1": state1, "input_states_2": state2})
                k = int(o["outputs"].reshape(-1, o["outputs"].shape[-1])[-1].argmax().item())
                if k == asr.blank_id:
                    break
                tokens.append(k); last = k
                state1 = o["output_states_1"].contiguous(); state2 = o["output_states_2"].contiguous()
                sym += 1

    streaming = asr.detokenize(tokens)
    print(f"STREAMING ({nchunks} chunks):", streaming[:90])
    print("\n--- full streaming ---\n", streaming)
    # crude similarity
    import difflib
    ratio = difflib.SequenceMatcher(None, offline, streaming).ratio()
    print(f"\nsimilarity(offline, streaming) = {ratio:.3f}")


if __name__ == "__main__":
    main()
