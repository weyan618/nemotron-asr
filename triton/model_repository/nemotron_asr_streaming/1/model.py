"""Triton Python backend for real-time cache-aware streaming ASR.

Uses Triton sequence batching: each audio stream is one sequence (constant
correlation_id). The backend keeps a StreamingSession per correlation_id holding
the encoder cache + RNNT decode state, so each request carries only the next
audio chunk and gets back the newly finalized transcript text (a delta).

Per request:
    AUDIO_CHUNK  FP32   [-1]   new 16 kHz mono samples (any length; ~320 ms is ideal)
    TARGET_LANG  BYTES  [1]    language code, read on the START request (optional)
    START/READY/END/CORRID     sequence control inputs (injected by Triton)

Output:
    TRANSCRIPT   BYTES  [1]    newly finalized text since the previous chunk
"""

import json
import os
import sys

import numpy as np
import triton_python_backend_utils as pb_utils

# Share the inference pipeline with the offline model (../../nemotron_asr/1).
_THIS = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.normpath(os.path.join(_THIS, "..", "..", "nemotron_asr", "1"))
for _p in (_THIS, _SHARED):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from asr_pipeline import NemotronASR  # noqa: E402


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})

        def param(name, default):
            return params.get(name, {}).get("string_value", default) if name in params else default

        assets_dir = param("ASSETS_DIR", os.path.join(_SHARED, "assets"))
        engine_dir = param("ENGINE_DIR", os.environ.get("ENGINE_DIR", ""))
        if not engine_dir:
            raise pb_utils.TritonModelException("ENGINE_DIR must be set")
        self.default_lang = param("DEFAULT_LANG", "en-US")

        self.logger = pb_utils.Logger
        self.logger.log_info(f"[nemotron_asr_streaming] loading engines from {engine_dir}")
        self.asr = NemotronASR(engine_dir=engine_dir, assets_dir=assets_dir)
        if self.asr.streaming_encoder is None:
            raise pb_utils.TritonModelException(
                "streaming_encoder-nemotron.engine not found in ENGINE_DIR"
            )
        self.sessions = {}  # correlation_id -> StreamingSession
        self.logger.log_info("[nemotron_asr_streaming] ready")

    @staticmethod
    def _ctrl(request, name):
        t = pb_utils.get_input_tensor_by_name(request, name)
        if t is None:
            return 0
        return int(np.asarray(t.as_numpy()).reshape(-1)[0])

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                corr = self._ctrl(request, "CORRID")
                start = self._ctrl(request, "START")
                end = self._ctrl(request, "END")

                if start or corr not in self.sessions:
                    lang_t = pb_utils.get_input_tensor_by_name(request, "TARGET_LANG")
                    if lang_t is not None:
                        raw = lang_t.as_numpy().reshape(-1)[0]
                        lang = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                    else:
                        lang = self.default_lang
                    self.sessions[corr] = self.asr.new_stream(lang)

                sess = self.sessions[corr]

                audio_t = pb_utils.get_input_tensor_by_name(request, "AUDIO_CHUNK")
                audio = (
                    np.asarray(audio_t.as_numpy(), dtype=np.float32).reshape(-1)
                    if audio_t is not None
                    else np.zeros(0, dtype=np.float32)
                )
                delta = sess.add_audio(audio)
                if end:
                    delta += sess.finalize()
                    self.sessions.pop(corr, None)

                out = pb_utils.Tensor("TRANSCRIPT", np.array([delta.encode("utf-8")], dtype=object))
                responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
            except Exception as exc:  # noqa: BLE001
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(f"{type(exc).__name__}: {exc}"),
                    )
                )
        return responses

    def finalize(self):
        self.sessions = {}
        self.asr = None
