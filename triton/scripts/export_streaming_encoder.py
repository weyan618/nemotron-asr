"""Re-export the Nemotron 3.5 ASR encoder as a CACHE-AWARE STREAMING ONNX graph.

Unlike the original offline encoder (audio_signal + length only), this graph
takes/returns the encoder caches so it can be driven chunk-by-chunk with bounded
latency:

    in : audio_signal[B,128,Tchunk], length[B],
         cache_last_channel[B,L,Tcache,D], cache_last_time[B,L,D,Tconv], cache_last_channel_len[B]
    out: outputs[B,1024,Tout], encoded_lengths[B],
         cache_last_channel_next, cache_last_time_next, cache_last_channel_next_len

Run on CPU (GB10/Blackwell tracing can be flaky on GPU), e.g.:

    /home/ebcpc10/miniconda3/envs/nemo-trt/bin/python \
        triton/scripts/export_streaming_encoder.py --att 56 3
"""
import argparse
import os

import torch


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    model_root = os.path.join(
        os.path.dirname(repo), "nemotron-asr", "nemotron-3.5-asr-streaming-0.6b"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", default=os.path.join(model_root, "nemotron-3.5-asr-streaming-0.6b.nemo"))
    ap.add_argument("--att", type=int, nargs=2, default=[56, 3],
                    help="att_context_size [left right]; right in {0,1,3,6,13} -> chunk=right+1")
    ap.add_argument("--out", default=os.path.join(model_root, "onnx_streaming", "streaming_encoder.onnx"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt import (
        EncDecHybridRNNTCTCBPEModelWithPrompt,
    )

    print("Loading model on CPU ...")
    m = EncDecHybridRNNTCTCBPEModelWithPrompt.restore_from(
        args.nemo, map_location="cpu", strict=False
    ).eval()

    enc = m.encoder
    enc.set_default_att_context_size(list(args.att))   # also calls setup_streaming_params()
    enc.export_cache_support = True
    cfg = enc.streaming_cfg
    print("att_context_size :", enc.att_context_size)
    print("streaming_cfg    : chunk_size=", cfg.chunk_size, "shift=", cfg.shift_size,
          "valid_out_len=", cfg.valid_out_len, "pre_encode_cache=", cfg.pre_encode_cache_size,
          "last_channel_cache=", cfg.last_channel_cache_size,
          "drop_extra_pre_encoded=", cfg.drop_extra_pre_encoded)

    ex = enc.input_example()
    print("input_example shapes:", [tuple(t.shape) if torch.is_tensor(t) else t for t in ex])

    print("Exporting ONNX ->", args.out)
    enc.export(args.out, onnx_opset_version=17, check_trace=False)

    # report ONNX IO
    try:
        import onnx
        g = onnx.load(args.out).graph
        print("\nONNX inputs:")
        for i in g.input:
            dims = [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]
            print(f"  {i.name}: {dims}")
        print("ONNX outputs:")
        for o in g.output:
            dims = [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]
            print(f"  {o.name}: {dims}")
    except Exception as e:  # noqa: BLE001
        print("onnx introspection skipped:", repr(e))


if __name__ == "__main__":
    main()
