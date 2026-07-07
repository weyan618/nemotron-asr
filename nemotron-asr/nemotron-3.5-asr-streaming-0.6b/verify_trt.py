import tensorrt as trt
import torch

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.BOOL: torch.bool,
    trt.DataType.INT8: torch.int8,
}

# Concrete shapes used to exercise each engine (within the built profiles).
TEST_INPUTS = {
    "trt_engines/encoder-nemotron.engine": {
        "audio_signal": (1, 128, 1000),
        "length": (1,),
    },
    "trt_engines/decoder_joint-nemotron.engine": {
        "encoder_outputs": (1, 1024, 1),
        "targets": (1, 1),
        "target_length": (1,),
        "input_states_1": (2, 1, 640),
        "input_states_2": (2, 1, 640),
    },
}


def make_tensor(shape, dtype, name):
    if dtype in (torch.int32, torch.int64):
        # length-like inputs must be valid; use the time dim where relevant.
        val = shape[-1] if "length" in name else 1
        return torch.full(shape, val, dtype=dtype, device="cuda")
    return torch.randn(shape, dtype=dtype, device="cuda")


def run(engine_path, input_shapes):
    print(f"\n=== {engine_path} ===")
    with open(engine_path, "rb") as f:
        data = f.read()
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(data)
    assert engine is not None, "deserialize failed"
    ctx = engine.create_execution_context()

    buffers = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = TRT_TO_TORCH[engine.get_tensor_dtype(name)]
        if mode == trt.TensorIOMode.INPUT:
            shape = input_shapes[name]
            ctx.set_input_shape(name, shape)
            t = make_tensor(shape, dtype, name)
            buffers[name] = t
            ctx.set_tensor_address(name, t.data_ptr())
            print(f"  IN  {name:18s} {tuple(shape)} {dtype}")

    # Outputs: shapes now resolvable from the chosen input shapes.
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shape = tuple(ctx.get_tensor_shape(name))
            dtype = TRT_TO_TORCH[engine.get_tensor_dtype(name)]
            t = torch.empty(shape, dtype=dtype, device="cuda")
            buffers[name] = t
            ctx.set_tensor_address(name, t.data_ptr())

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ok = ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    assert ok, "execution failed"

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            t = buffers[name]
            print(f"  OUT {name:18s} {tuple(t.shape)} {t.dtype}  "
                  f"finite={torch.isfinite(t).all().item()}")
    print("  -> inference OK")


if __name__ == "__main__":
    print("TensorRT", trt.__version__, "| torch", torch.__version__,
          "| device", torch.cuda.get_device_name(0))
    for path, shapes in TEST_INPUTS.items():
        run(path, shapes)
    print("\nALL ENGINES VERIFIED")
