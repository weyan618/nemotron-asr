import argparse
import os
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# Dynamic-shape optimization profiles per input (min / opt / max).
# Encoder consumes log-mel features [B, 128, T]; T is the number of 10ms frames.
# decoder_joint is invoked per greedy step, so time/label dims stay tiny.
PROFILES = {
    "encoder-nemotron.onnx": {
        "audio_signal": ((1, 128, 16), (1, 128, 1000), (8, 128, 3000)),
        "length": ((1,), (1,), (8,)),
    },
    "decoder_joint-nemotron.onnx": {
        "encoder_outputs": ((1, 1024, 1), (1, 1024, 1), (8, 1024, 4)),
        "targets": ((1, 1), (1, 1), (8, 4)),
        "target_length": ((1,), (1,), (8,)),
        "input_states_1": ((2, 1, 640), (2, 1, 640), (2, 8, 640)),
        "input_states_2": ((2, 1, 640), (2, 1, 640), (2, 8, 640)),
    },
}


def build(onnx_path, engine_path, profile_spec, fp16=True, workspace_gb=8):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"[parse] {onnx_path}")
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print("  ONNX parse error:", parser.get_error(i))
        raise RuntimeError(f"Failed to parse {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30)
    )
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[config] FP16 enabled")

    profile = builder.create_optimization_profile()
    for name, (mn, opt, mx) in profile_spec.items():
        print(f"[profile] {name}: min={mn} opt={opt} max={mx}")
        profile.set_shape(name, mn, opt, mx)
    config.add_optimization_profile(profile)

    print(f"[build] serializing engine -> {engine_path}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Engine build failed for {onnx_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[done] {engine_path} ({os.path.getsize(engine_path)//(1<<20)} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default="onnx")
    ap.add_argument("--engine-dir", default="trt_engines")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--workspace-gb", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.engine_dir, exist_ok=True)
    print("TensorRT", trt.__version__)
    for onnx_name, spec in PROFILES.items():
        onnx_path = os.path.join(args.onnx_dir, onnx_name)
        engine_path = os.path.join(
            args.engine_dir, onnx_name.replace(".onnx", ".engine")
        )
        build(
            onnx_path,
            engine_path,
            spec,
            fp16=not args.no_fp16,
            workspace_gb=args.workspace_gb,
        )


if __name__ == "__main__":
    main()
