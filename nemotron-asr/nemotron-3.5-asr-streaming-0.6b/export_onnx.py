import os
import glob
import torch
import nemo.collections.asr as nemo_asr

MODEL = "nemotron-3.5-asr-streaming-0.6b.nemo"
OUT_DIR = "onnx"
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

m = nemo_asr.models.ASRModel.restore_from(MODEL, map_location=device)
m = m.to(device).eval()
m.freeze()

out = os.path.join(OUT_DIR, "nemotron.onnx")
print("Exporting (native NeMo) to", out)
exported = m.export(out)
print("export() returned:", exported)

print("=== files in", OUT_DIR, "===")
for f in sorted(glob.glob(os.path.join(OUT_DIR, "*"))):
    print(f, os.path.getsize(f) // 1024, "KB")
