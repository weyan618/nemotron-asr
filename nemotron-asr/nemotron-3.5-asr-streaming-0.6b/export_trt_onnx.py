import os
import glob
import json
import torch

from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt import (
    EncDecHybridRNNTCTCBPEModelWithPrompt,
)

MODEL = "nemotron-3.5-asr-streaming-0.6b.nemo"
OUT_DIR = "onnx"
os.makedirs(OUT_DIR, exist_ok=True)

# Export on CPU to avoid GB10/Blackwell kernel issues during tracing.
device = "cpu"
print("Restoring model on", device)
# strict=False: the trained model is RNNT-only with prompt; the hybrid class
# also builds an (unused) CTC head whose weights are absent from the checkpoint.
m = EncDecHybridRNNTCTCBPEModelWithPrompt.restore_from(
    MODEL, map_location=device, strict=False
)
m = m.to(device).eval()
m.freeze()

print("MODEL CLASS:", type(m).__name__)
for name in ["encoder", "decoder", "joint", "ctc_decoder"]:
    comp = getattr(m, name, None)
    print("  ", name, "->", type(comp).__name__ if comp is not None else None)
try:
    print("export subnets:", m.list_export_subnets())
except Exception as e:
    print("list_export_subnets err:", repr(e))

out = os.path.join(OUT_DIR, "nemotron.onnx")
print("Exporting (native NeMo) to", out)
exported = m.export(out, onnx_opset_version=17)
print("export() returned:", exported)

print("=== files in", OUT_DIR, "===")
for f in sorted(glob.glob(os.path.join(OUT_DIR, "*"))):
    print(f, os.path.getsize(f) // 1024, "KB")
