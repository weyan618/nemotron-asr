"""Triton Python backend for Nemotron 3.5 ASR streaming 0.6B (TRT engines).

Inputs  (per request, no batch dim; max_batch_size=0):
    AUDIO        FP32   [-1]   16 kHz mono waveform samples in [-1, 1]
    TARGET_LANG  BYTES  [1]    language code, e.g. "en-US" (optional, default en-US)

Output:
    TRANSCRIPT   BYTES  [1]    decoded text

Engine + asset locations are resolved from model config `parameters`
(ENGINE_DIR, ASSETS_DIR) with sensible defaults relative to the model dir.
"""

import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils

from asr_pipeline import NemotronASR


class TritonPythonModel:
    def initialize(self, args):
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})

        def param(name, default):
            return params.get(name, {}).get("string_value", default) if name in params else default

        assets_dir = param("ASSETS_DIR", os.path.join(model_dir, "assets"))
        engine_dir = param("ENGINE_DIR", os.environ.get("ENGINE_DIR", ""))
        if not engine_dir:
            raise pb_utils.TritonModelException(
                "ENGINE_DIR must be set (model config parameter or env var) and point "
                "to the directory containing encoder-nemotron.engine and "
                "decoder_joint-nemotron.engine"
            )
        self.default_lang = param("DEFAULT_LANG", "en-US")

        self.logger = pb_utils.Logger
        self.logger.log_info(
            f"[nemotron_asr] loading engines from {engine_dir}, assets from {assets_dir}"
        )
        self.asr = NemotronASR(engine_dir=engine_dir, assets_dir=assets_dir)
        self.logger.log_info("[nemotron_asr] ready")

    def execute(self, requests):
        # Parse every request first; collect the valid ones into one batch so the
        # encoder and the RNNT greedy decode run batched on the GPU.
        responses = [None] * len(requests)
        audios, langs, batch_idx = [], [], []
        for i, request in enumerate(requests):
            try:
                audio_t = pb_utils.get_input_tensor_by_name(request, "AUDIO")
                audio = np.asarray(audio_t.as_numpy(), dtype=np.float32).reshape(-1)
                lang_t = pb_utils.get_input_tensor_by_name(request, "TARGET_LANG")
                if lang_t is not None:
                    raw = lang_t.as_numpy().reshape(-1)[0]
                    target_lang = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                else:
                    target_lang = self.default_lang
                audios.append(audio)
                langs.append(target_lang)
                batch_idx.append(i)
            except Exception as exc:  # noqa: BLE001
                responses[i] = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"{type(exc).__name__}: {exc}"),
                )

        if audios:
            try:
                texts = self.asr.transcribe_batch(audios, langs)
                for i, text in zip(batch_idx, texts):
                    out = pb_utils.Tensor(
                        "TRANSCRIPT", np.array([text.encode("utf-8")], dtype=object)
                    )
                    responses[i] = pb_utils.InferenceResponse(output_tensors=[out])
            except Exception as exc:  # noqa: BLE001
                err = pb_utils.TritonError(f"{type(exc).__name__}: {exc}")
                for i in batch_idx:
                    responses[i] = pb_utils.InferenceResponse(output_tensors=[], error=err)
        return responses

    def finalize(self):
        self.asr = None
