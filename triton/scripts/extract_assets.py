"""Reproduce the serving assets from the original .nemo checkpoint.

Produces, into model_repository/nemotron_asr/1/assets/:
    vocab.json            joint vocabulary (id -> piece), len 13087, blank = 13087
    prompt_dictionary.json language code -> prompt index (0..127)
    feat_assets.pt        {mel_fb[128,257], window[400]} for the log-mel featurizer
    prompt_kernel.pt      language-conditioning MLP weights (1152->2048->1024)

Run with the env that can load the model (e.g. nemo-trt):
    /path/to/nemo-trt/bin/python triton/scripts/extract_assets.py \
        --nemo <model>.nemo
"""
import argparse
import json
import os

import torch
import yaml


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    default_nemo = os.path.join(
        os.path.dirname(repo), "nemotron-asr", "nemotron-3.5-asr-streaming-0.6b",
        "nemotron-3.5-asr-streaming-0.6b.nemo",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", default=default_nemo)
    ap.add_argument("--out", default=os.path.join(
        repo, "model_repository", "nemotron_asr", "1", "assets"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt import (
        EncDecHybridRNNTCTCBPEModelWithPrompt,
    )
    from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor

    m = EncDecHybridRNNTCTCBPEModelWithPrompt.restore_from(
        args.nemo, map_location="cpu", strict=False
    ).eval()

    # vocabulary (id -> piece)
    vocab = list(m.joint.vocabulary)
    json.dump(vocab, open(os.path.join(args.out, "vocab.json"), "w"), ensure_ascii=False)
    print("vocab.json:", len(vocab), "tokens; blank_id =", len(vocab))

    # prompt dictionary
    pd = dict(m.cfg.model_defaults.get("prompt_dictionary"))
    json.dump(pd, open(os.path.join(args.out, "prompt_dictionary.json"), "w"),
              ensure_ascii=False, indent=0)
    print("prompt_dictionary.json:", len(pd), "entries")

    # prompt_kernel MLP weights
    sd = {k: v.cpu().float() for k, v in m.prompt_kernel.state_dict().items()}
    torch.save(sd, os.path.join(args.out, "prompt_kernel.pt"))
    print("prompt_kernel.pt:", {k: tuple(v.shape) for k, v in sd.items()})

    # featurizer assets (mel filterbank + window), with inference settings
    pcfg = dict(m.cfg.preprocessor)
    pcfg.pop("_target_", None)
    pcfg["dither"] = 0.0
    pcfg["pad_to"] = 0
    fb = AudioToMelSpectrogramPreprocessor(**pcfg).featurizer
    torch.save({"mel_fb": fb.fb.cpu().float(), "window": fb.window.cpu().float()},
               os.path.join(args.out, "feat_assets.pt"))
    print("feat_assets.pt: mel_fb", tuple(fb.fb.shape), "window", tuple(fb.window.shape))
    print("Done ->", args.out)


if __name__ == "__main__":
    main()
