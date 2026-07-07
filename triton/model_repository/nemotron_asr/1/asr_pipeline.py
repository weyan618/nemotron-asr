"""Core inference pipeline for the Nemotron 3.5 ASR streaming 0.6B TRT engines.

This module is deliberately free of any Triton / NeMo runtime dependency so that
it can be imported both by the Triton Python backend (`model.py`) and by the
standalone parity test (`test_pipeline.py`).

Pipeline stages (offline / full-utterance, ``target_lang`` conditioned):

    audio (16 kHz mono)
      -> log-mel features          (LeanFeaturizer, matches NeMo FilterbankFeatures)
      -> encoder engine            (audio_signal, length -> outputs[B,1024,T])
      -> prompt_kernel MLP         (language one-hot conditioning, NOT baked in the engine)
      -> RNNT greedy decode        (decoder_joint engine, per-step)
      -> detokenize                (joint vocabulary, sentencepiece "▁" -> space)
"""

import json
import os

import numpy as np
import tensorrt as trt
import torch

# RNNT prediction-network hidden size and number of layers (from model_config.yaml).
PRED_HIDDEN = 640
PRED_LAYERS = 2
NUM_PROMPTS = 128
ENC_HIDDEN = 1024
MAX_SYMBOLS_PER_STEP = 10
# Max batch the TRT engines were built for (encoder/decoder_joint profile max = 8).
MAX_BATCH = 8

# Cache-aware streaming config for att_context_size=[56,3] (320 ms chunk, subsampling 8).
# Must match how streaming_encoder-nemotron.engine was exported (see scripts/).
STREAMING_CFG = {
    "chunk_feat": 32,        # new log-mel frames consumed per step (32 * 10ms = 320ms)
    "pre_encode_cache": 9,   # left feature-frame overlap fed each step
    "n_layers": 24,
    "l_cache": 56,           # encoder last-channel cache length
    "d_model": 1024,
    "t_conv": 8,             # conv (last-time) cache length
}

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.BOOL: torch.bool,
}


class TRTEngine:
    """Thin wrapper around a TensorRT engine for execute_async_v3 with torch tensors."""

    def __init__(self, engine_path, logger):
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine: {engine_path}")
        self.ctx = self.engine.create_execution_context()
        self.input_names, self.output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def __call__(self, feeds):
        """Enqueue inference on the *current* torch CUDA stream. All torch ops and
        TRT executions share one stream, so ordering is implicit and no manual
        cross-stream synchronization is needed (avoids data races)."""
        for name, tensor in feeds.items():
            assert tensor.is_contiguous(), f"{name} must be contiguous"
            self.ctx.set_input_shape(name, tuple(tensor.shape))
            self.ctx.set_tensor_address(name, tensor.data_ptr())
        outputs = {}
        for name in self.output_names:
            shape = tuple(self.ctx.get_tensor_shape(name))
            dtype = _TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]
            out = torch.empty(shape, dtype=dtype, device="cuda")
            outputs[name] = out
            self.ctx.set_tensor_address(name, out.data_ptr())
        stream = torch.cuda.current_stream()
        if not self.ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TRT execute_async_v3 failed")
        return outputs


class LeanFeaturizer:
    """Log-mel feature extractor, numerically equivalent to NeMo's
    ``AudioToMelSpectrogramPreprocessor`` (FilterbankFeatures) for this model's
    config: sr=16000, n_fft=512, win=400, hop=160, 128 mels, preemph=0.97,
    mag_power=2.0, log(add, 2**-24), normalize=NA (none), dither disabled."""

    def __init__(self, assets_path, device="cuda"):
        assets = torch.load(assets_path, map_location=device)
        self.mel_fb = assets["mel_fb"].to(device).float()      # [1,128,257]
        if self.mel_fb.dim() == 3:
            self.mel_fb = self.mel_fb.squeeze(0)               # [128,257]
        self.window = assets["window"].to(device).float()      # [400]
        self.device = device
        self.n_fft = 512
        self.hop_length = 160
        self.win_length = 400
        self.preemph = 0.97
        self.mag_power = 2.0
        self.log_zero_guard = 2.0 ** -24

    def __call__(self, audio, audio_len):
        """audio: [B, T] float32 on device. audio_len: [B] int64 (num samples).
        Returns (features[B,128,Tf] float32, feat_len[B] int64)."""
        x = audio
        # output feature lengths: floor(samples / hop_length) with center padding n_fft/2
        feat_len = torch.div(audio_len, self.hop_length, rounding_mode="floor").to(torch.int64)

        # preemphasis (mask any padding beyond the true sample length to 0 first)
        time_mask = (
            torch.arange(x.shape[1], device=x.device).unsqueeze(0) < audio_len.unsqueeze(1)
        )
        x = torch.cat((x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
        x = x.masked_fill(~time_mask, 0.0)

        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            window=self.window,
            return_complex=True,
            pad_mode="constant",
        )
        mag = torch.sqrt(torch.view_as_real(stft).pow(2).sum(-1))
        power = mag.pow(self.mag_power)                          # [B,257,T]
        mel = torch.matmul(self.mel_fb, power)                  # [B,128,T]
        feats = torch.log(mel + self.log_zero_guard)

        # zero out frames beyond feat_len
        max_t = feats.shape[-1]
        fmask = torch.arange(max_t, device=feats.device).unsqueeze(0) >= feat_len.unsqueeze(1)
        feats = feats.masked_fill(fmask.unsqueeze(1), 0.0)
        return feats.contiguous(), feat_len


class NemotronASR:
    """Full offline ASR pipeline over the two TensorRT engines."""

    def __init__(self, engine_dir, assets_dir, device="cuda", trt_log_level=trt.Logger.ERROR):
        self.device = device
        torch.cuda.init()
        self.logger = trt.Logger(trt_log_level)

        self.encoder = TRTEngine(os.path.join(engine_dir, "encoder-nemotron.engine"), self.logger)
        self.decoder_joint = TRTEngine(
            os.path.join(engine_dir, "decoder_joint-nemotron.engine"), self.logger
        )

        # Optional cache-aware streaming encoder (chunk-by-chunk, with cache in/out).
        # Enables real-time streaming via new_stream(); offline path works without it.
        senc_path = os.path.join(engine_dir, "streaming_encoder-nemotron.engine")
        self.streaming_encoder = (
            TRTEngine(senc_path, self.logger) if os.path.exists(senc_path) else None
        )

        self.featurizer = LeanFeaturizer(os.path.join(assets_dir, "feat_assets.pt"), device)

        self.vocab = json.load(open(os.path.join(assets_dir, "vocab.json"), encoding="utf-8"))
        self.blank_id = len(self.vocab)  # 13087; joint logits have len(vocab)+1 classes
        self.prompt_dict = json.load(
            open(os.path.join(assets_dir, "prompt_dictionary.json"), encoding="utf-8")
        )

        # prompt_kernel MLP: Linear(1152,2048) -> ReLU -> Linear(2048,1024)
        sd = torch.load(os.path.join(assets_dir, "prompt_kernel.pt"), map_location=device)
        self.prompt_kernel = torch.nn.Sequential(
            torch.nn.Linear(NUM_PROMPTS + ENC_HIDDEN, ENC_HIDDEN * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(ENC_HIDDEN * 2, ENC_HIDDEN),
        ).to(device).eval()
        self.prompt_kernel.load_state_dict(
            {
                "0.weight": sd["0.weight"], "0.bias": sd["0.bias"],
                "2.weight": sd["2.weight"], "2.bias": sd["2.bias"],
            }
        )

    def resolve_prompt_id(self, target_lang):
        if target_lang in self.prompt_dict:
            return self.prompt_dict[target_lang]
        raise ValueError(
            f"Unknown target_lang '{target_lang}'. Examples: "
            f"{list(self.prompt_dict.keys())[:8]} ..."
        )

    @torch.no_grad()
    def _encode(self, audio, audio_len, prompt_ids):
        """audio: [B,T] float32. prompt_ids: list[int] length B.
        Returns (encoded[B,1024,T'], enc_len[B])."""
        feats, feat_len = self.featurizer(audio, audio_len)
        enc_out = self.encoder(
            {"audio_signal": feats.float(), "length": feat_len.to(torch.int64)}
        )
        encoded = enc_out["outputs"]                       # [B,1024,T]
        enc_len = enc_out["encoded_lengths"]               # [B]

        # language conditioning via prompt_kernel (concat + MLP), applied here
        # because the exported encoder engine does NOT include it.
        enc_bt = encoded.transpose(1, 2)                   # [B,T,1024]
        prompt = torch.zeros(
            (enc_bt.shape[0], enc_bt.shape[1], NUM_PROMPTS), device=self.device
        )
        idx = torch.tensor(prompt_ids, device=self.device, dtype=torch.long)
        prompt[torch.arange(enc_bt.shape[0], device=self.device), :, idx] = 1.0
        fused = self.prompt_kernel(torch.cat([enc_bt, prompt], dim=-1))
        encoded = fused.transpose(1, 2).contiguous().float()
        return encoded, enc_len

    @torch.no_grad()
    def _greedy_decode_batch(self, encoded, enc_len):
        """Batched RNNT greedy decode.

        encoded: [B,1024,Tmax]; enc_len: [B] int. Each utterance advances through
        its own time axis; the decoder_joint engine is invoked once per inner step
        for the whole (still-active) batch. Returns list[list[int]] of token ids.
        """
        B = encoded.shape[0]
        dev = self.device
        ar = torch.arange(B, device=dev)

        t = torch.zeros(B, dtype=torch.long, device=dev)              # time pointer per utt
        symbols = torch.zeros(B, dtype=torch.long, device=dev)        # symbols emitted at current t
        last_label = torch.full((B,), self.blank_id, dtype=torch.int32, device=dev)
        state1 = torch.zeros((PRED_LAYERS, B, PRED_HIDDEN), device=dev)
        state2 = torch.zeros((PRED_LAYERS, B, PRED_HIDDEN), device=dev)
        enc_len = enc_len.to(dev).long()
        Tmax = encoded.shape[2]

        tokens = [[] for _ in range(B)]
        while True:
            active = t < enc_len
            if not bool(active.any()):
                break
            idx = t.clamp(max=max(Tmax - 1, 0))
            enc_t = encoded[ar, :, idx].unsqueeze(-1).contiguous()  # [B,1024,1]
            targets = last_label.reshape(B, 1).contiguous()
            tlen = torch.ones(B, dtype=torch.int32, device=dev)
            out = self.decoder_joint(
                {
                    "encoder_outputs": enc_t,
                    "targets": targets,
                    "target_length": tlen,
                    "input_states_1": state1,
                    "input_states_2": state2,
                }
            )
            V = out["outputs"].shape[-1]
            logits = out["outputs"].reshape(B, -1, V)[:, -1, :]    # [B,V]
            k = logits.argmax(dim=-1).to(torch.int32)              # [B]

            # force blank (-> advance time) where we already hit max symbols
            force_blank = symbols >= MAX_SYMBOLS_PER_STEP
            is_blank = (k == self.blank_id) | force_blank
            emit = active & ~is_blank
            advance = active & is_blank

            if bool(emit.any()):
                emit_idx = emit.nonzero(as_tuple=True)[0].tolist()
                k_cpu = k.tolist()
                for b in emit_idx:
                    tokens[b].append(k_cpu[b])
                em = emit.view(1, B, 1)
                state1 = torch.where(em, out["output_states_1"], state1)
                state2 = torch.where(em, out["output_states_2"], state2)
                last_label = torch.where(emit, k, last_label)
                symbols = symbols + emit.long()

            if bool(advance.any()):
                t = t + advance.long()
                symbols = torch.where(advance, torch.zeros_like(symbols), symbols)
        return tokens

    def detokenize(self, token_ids):
        pieces = [self.vocab[i] for i in token_ids]
        return "".join(pieces).replace("\u2581", " ").strip()

    @torch.no_grad()
    def _transcribe_chunk(self, audio_list, lang_list):
        """Process up to MAX_BATCH utterances together. Returns list[str]."""
        B = len(audio_list)
        prompt_ids = [self.resolve_prompt_id(lang) for lang in lang_list]
        max_len = max(a.shape[0] for a in audio_list)
        audio = torch.zeros((B, max_len), dtype=torch.float32, device=self.device)
        audio_len = torch.zeros(B, dtype=torch.int64, device=self.device)
        for i, a in enumerate(audio_list):
            t = torch.as_tensor(a, dtype=torch.float32, device=self.device)
            audio[i, : t.shape[0]] = t
            audio_len[i] = t.shape[0]
        encoded, enc_len = self._encode(audio, audio_len, prompt_ids)
        tokens = self._greedy_decode_batch(encoded, enc_len)
        return [self.detokenize(tok) for tok in tokens]

    @torch.no_grad()
    def transcribe_batch(self, audio_list, lang_list=None):
        """audio_list: list of 1-D float32 numpy arrays (16 kHz mono).
        lang_list: list of language codes (or None -> en-US). Returns list[str].
        Internally chunked to the engines' max batch size."""
        if lang_list is None:
            lang_list = ["en-US"] * len(audio_list)
        results = []
        for i in range(0, len(audio_list), MAX_BATCH):
            results.extend(
                self._transcribe_chunk(audio_list[i : i + MAX_BATCH], lang_list[i : i + MAX_BATCH])
            )
        return results

    @torch.no_grad()
    def transcribe(self, audio_np, target_lang="en-US"):
        """Single-utterance convenience wrapper. Returns transcript string."""
        return self.transcribe_batch([np.asarray(audio_np, dtype=np.float32)], [target_lang])[0]

    def new_stream(self, target_lang="en-US"):
        """Create a real-time streaming session (one per audio stream)."""
        if self.streaming_encoder is None:
            raise RuntimeError(
                "streaming_encoder-nemotron.engine not found; build it with "
                "scripts/export_streaming_encoder.py + scripts/build_streaming_trt.py"
            )
        return StreamingSession(self, self.resolve_prompt_id(target_lang))


class StreamingSession:
    """Per-stream cache-aware streaming decode state.

    Feed raw 16 kHz mono audio incrementally via add_audio(); each call returns
    the newly finalized transcript text (delta). The encoder cache and RNNT
    decode state are carried across chunks so latency stays bounded (~320 ms).
    """

    def __init__(self, asr, prompt_id):
        self.asr = asr
        self.pid = prompt_id
        dev = asr.device
        cfg = STREAMING_CFG
        self.cfg = cfg
        self.dev = dev

        # rolling raw-audio buffer (on device) + absolute sample/frame bookkeeping.
        # We never recompute features over the whole stream: each chunk's 41-frame
        # window is computed from just the samples it needs (see _chunk_features),
        # so per-step cost is O(chunk), not O(stream length).
        self.raw = torch.zeros(0, device=dev)
        self.raw_base = 0          # absolute sample index of self.raw[0]
        self.total = 0             # total samples received so far
        self.buffer_idx = 0        # absolute feature-frame consume pointer

        # encoder caches in export layout [B, L, T, D] / [B, L, D, Tconv]
        self.c_chan = torch.zeros((1, cfg["n_layers"], cfg["l_cache"], cfg["d_model"]), device=dev)
        self.c_time = torch.zeros((1, cfg["n_layers"], cfg["d_model"], cfg["t_conv"]), device=dev)
        self.c_len = torch.zeros(1, dtype=torch.int64, device=dev)

        # persistent RNNT greedy state
        self.state1 = torch.zeros((PRED_LAYERS, 1, PRED_HIDDEN), device=dev)
        self.state2 = torch.zeros((PRED_LAYERS, 1, PRED_HIDDEN), device=dev)
        self.last = self.asr.blank_id

        self.tokens = []
        self._emitted = ""

    @torch.no_grad()
    def _run_window(self, window, valid_len):
        """window: [1,128,41] features. valid_len: number of real (non-pad) frames."""
        a = self.asr
        out = a.streaming_encoder(
            {
                "audio_signal": window.contiguous(),
                "length": torch.tensor([valid_len], dtype=torch.int64, device=self.dev),
                "cache_last_channel": self.c_chan,
                "cache_last_time": self.c_time,
                "cache_last_channel_len": self.c_len,
            }
        )
        self.c_chan = out["cache_last_channel_next"].contiguous()
        self.c_time = out["cache_last_time_next"].contiguous()
        self.c_len = out["cache_last_channel_next_len"].to(torch.int64).contiguous()
        elen = int(out["encoded_lengths"][0].item())
        if elen <= 0:
            return
        encoded = out["outputs"][:, :, :elen]                      # [1,1024,elen]

        # language conditioning (prompt_kernel), same as offline path
        enc_bt = encoded.transpose(1, 2)                           # [1,elen,1024]
        prompt = torch.zeros((1, enc_bt.shape[1], NUM_PROMPTS), device=self.dev)
        prompt[:, :, self.pid] = 1.0
        enc_p = a.prompt_kernel(torch.cat([enc_bt, prompt], dim=-1)).transpose(1, 2).contiguous()

        # persistent greedy over the new encoder frames
        for t in range(elen):
            enc_t = enc_p[:, :, t : t + 1].contiguous()
            sym = 0
            while sym < MAX_SYMBOLS_PER_STEP:
                o = a.decoder_joint(
                    {
                        "encoder_outputs": enc_t,
                        "targets": torch.tensor([[self.last]], dtype=torch.int32, device=self.dev),
                        "target_length": torch.tensor([1], dtype=torch.int32, device=self.dev),
                        "input_states_1": self.state1,
                        "input_states_2": self.state2,
                    }
                )
                V = o["outputs"].shape[-1]
                k = int(o["outputs"].reshape(-1, V)[-1].argmax().item())
                if k == a.blank_id:
                    break
                self.tokens.append(k)
                self.last = k
                self.state1 = o["output_states_1"].contiguous()
                self.state2 = o["output_states_2"].contiguous()
                sym += 1

    def _segment(self, s_lo, s_hi):
        """Absolute raw samples [s_lo, s_hi) as a 1-D tensor; out-of-range (before
        start, after end, or already-trimmed) positions are zero — which matches
        torch.stft's constant padding at the true signal boundaries."""
        seg = torch.zeros(s_hi - s_lo, device=self.dev)
        lo = max(s_lo, self.raw_base)
        hi = min(s_hi, self.raw_base + self.raw.shape[0])
        if hi > lo:
            seg[lo - s_lo : hi - s_lo] = self.raw[lo - self.raw_base : hi - self.raw_base]
        return seg

    @torch.no_grad()
    def _chunk_features(self, a, b):
        """Log-mel features for absolute frame indices [a, b), bit-identical to the
        full-stream featurizer for interior frames. Uses constant (zero) padding and
        preemphasis exactly like LeanFeaturizer, but only over the needed samples."""
        f = self.asr.featurizer
        nfft, hop, half = f.n_fft, f.hop_length, f.n_fft // 2
        # samples needed: window of frame a starts at a*hop-half; last frame (b-1)
        # ends at (b-1)*hop+half; one extra sample on the left for preemphasis.
        seg = self._segment(a * hop - half - 1, (b - 1) * hop + half)
        pre = seg[1:] - f.preemph * seg[:-1]                       # preemph; abs start a*hop-half
        # LeanFeaturizer zero-pads *after* preemphasis, so end-pad samples must be 0
        # (not the spurious -preemph*x[total-1] a continuous diff would produce).
        end_rel = self.total - (a * hop - half)
        if 0 <= end_rel < pre.shape[0]:
            pre[max(end_rel, 0):] = 0.0
        stft = torch.stft(
            pre.unsqueeze(0), n_fft=nfft, hop_length=hop, win_length=f.win_length,
            center=False, window=f.window, return_complex=True, pad_mode="constant",
        )                                                          # [1,257,b-a]
        mag = torch.sqrt(torch.view_as_real(stft).pow(2).sum(-1))
        mel = torch.matmul(f.mel_fb, mag.pow(f.mag_power))         # [1,128,b-a]
        return torch.log(mel + f.log_zero_guard)

    @torch.no_grad()
    def add_audio(self, samples, last=False):
        """Append raw float32 mono 16 kHz samples; returns newly finalized text (delta)."""
        cfg = self.cfg
        cf, pe = cfg["chunk_feat"], cfg["pre_encode_cache"]
        hop, half = self.asr.featurizer.hop_length, self.asr.featurizer.n_fft // 2

        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size:
            self.raw = torch.cat([self.raw, torch.from_numpy(samples).to(self.dev)])
            self.total += samples.size
        feat_len_total = self.total // hop                         # floor, matches LeanFeaturizer

        while True:
            bi = self.buffer_idx
            if last:
                if bi >= feat_len_total:
                    break
            else:
                # a chunk frame is final once its STFT window lies fully inside real
                # audio (constant end-pad no longer affects it): (bi+cf-1)*hop+half <= total
                if (bi + cf - 1) * hop + half > self.total:
                    break

            a = max(0, bi - pe)                                    # first real frame to compute
            b = min(bi + cf, feat_len_total) if last else bi + cf  # exclusive last frame
            n_chunk = min(cf, b - bi)
            if n_chunk <= 0:
                break

            feats = self._chunk_features(a, b)                     # [1,128,b-a]
            if bi < pe:  # left-pad zeros for the missing pre-encode frames (first chunk)
                pad = torch.zeros((1, feats.shape[1], pe - bi), device=self.dev)
                feats = torch.cat([pad, feats], dim=2)
            if feats.shape[2] < pe + cf:  # final partial chunk: right-pad to window width
                pad = torch.zeros((1, feats.shape[1], pe + cf - feats.shape[2]), device=self.dev)
                feats = torch.cat([feats, pad], dim=2)
            window = feats.contiguous()                            # [1,128,pe+cf]
            self._run_window(window, valid_len=pe + n_chunk)
            self.buffer_idx += cf

        # drop raw samples no longer needed by any future chunk
        keep_from = max(0, (self.buffer_idx - pe)) * hop - half - 1
        if keep_from > self.raw_base:
            self.raw = self.raw[keep_from - self.raw_base :]
            self.raw_base = keep_from

        full = self.asr.detokenize(self.tokens)
        delta = full[len(self._emitted):]
        self._emitted = full
        return delta

    @torch.no_grad()
    def finalize(self):
        """Flush trailing audio and return any remaining transcript text."""
        return self.add_audio(np.zeros(0, dtype=np.float32), last=True)

    @property
    def text(self):
        return self._emitted
