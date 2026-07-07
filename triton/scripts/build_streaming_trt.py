"""Build a TensorRT engine for the cache-aware streaming encoder ONNX.

For att_context_size=[56,3] (320 ms chunk, subsampling 8):
  audio_signal           [B,128,T]   T up to chunk(32)+pre_encode_cache(9)=41 (first chunk=25)
  length                 [B]
  cache_last_channel     [B,24,56,1024]
  cache_last_time        [B,24,1024,8]
  cache_last_channel_len [B]
Only batch (and audio T) are dynamic; cache dims are fixed by the chunk config.
"""
import argparse
import os

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build(onnx_path, engine_path, max_batch, t_max, l_cache, t_conv, n_layers, d_model,
          fp16=True, workspace_gb=8):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    print(f"[parse] {onnx_path}")
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print("  ONNX parse error:", parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[config] FP16 enabled")

    profile = builder.create_optimization_profile()
    shapes = {
        "audio_signal": ((1, 128, 1), (1, 128, t_max), (max_batch, 128, t_max)),
        "length": ((1,), (1,), (max_batch,)),
        "cache_last_channel": ((1, n_layers, l_cache, d_model),
                               (1, n_layers, l_cache, d_model),
                               (max_batch, n_layers, l_cache, d_model)),
        "cache_last_time": ((1, n_layers, d_model, t_conv),
                            (1, n_layers, d_model, t_conv),
                            (max_batch, n_layers, d_model, t_conv)),
        "cache_last_channel_len": ((1,), (1,), (max_batch,)),
    }
    for name, (mn, opt, mx) in shapes.items():
        print(f"[profile] {name}: min={mn} opt={opt} max={mx}")
        profile.set_shape(name, mn, opt, mx)
    config.add_optimization_profile(profile)

    print(f"[build] serializing -> {engine_path}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[done] {engine_path} ({os.path.getsize(engine_path) // (1 << 20)} MB)")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    model_root = os.path.join(
        os.path.dirname(os.path.dirname(here)), "nemotron-asr", "nemotron-3.5-asr-streaming-0.6b"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(model_root, "onnx_streaming", "streaming_encoder.onnx"))
    ap.add_argument("--engine", default=os.path.join(model_root, "trt_engines", "streaming_encoder-nemotron.engine"))
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--t-max", type=int, default=41)
    ap.add_argument("--l-cache", type=int, default=56)
    ap.add_argument("--t-conv", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=24)
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args()
    print("TensorRT", trt.__version__)
    build(args.onnx, args.engine, args.max_batch, args.t_max, args.l_cache, args.t_conv,
          args.n_layers, args.d_model, fp16=not args.no_fp16)


if __name__ == "__main__":
    main()
