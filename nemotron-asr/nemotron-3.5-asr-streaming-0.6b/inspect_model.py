import nemo.collections.asr as nemo_asr

MODEL = "nemotron-3.5-asr-streaming-0.6b.nemo"

m = nemo_asr.models.ASRModel.restore_from(MODEL, map_location="cpu")
print("=" * 60)
print("MODEL CLASS:", type(m).__name__)
print("=" * 60)
for name in ["encoder", "decoder", "joint"]:
    comp = getattr(m, name, None)
    print(name, "->", type(comp).__name__ if comp is not None else None)
print("concat:", getattr(m, "concat", "N/A"))
print("num_prompts:", getattr(m, "num_prompts", "N/A"))
print("has prompt_kernel:", hasattr(m, "prompt_kernel"))
print("=" * 60)
try:
    print("export subnets:", m.list_export_subnets())
except Exception as e:
    print("list_export_subnets err:", repr(e))
enc = m.encoder
for attr in ["att_context_size", "att_context_style", "streaming_cfg"]:
    print("encoder.", attr, "->", getattr(enc, attr, "N/A"))
print("=" * 60)
# prompt dictionary
pd = m.cfg.model_defaults.get("prompt_dictionary", None)
if pd:
    items = list(pd.items())
    print("prompt_dictionary size:", len(items), "sample:", items[:5])
print("subsampling_factor:", m.cfg.get("subsampling_factor", "N/A"))
print("encoder d_model:", m.cfg.encoder.get("d_model", "N/A"))
