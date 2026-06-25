"""SidONNX speech restoration backend (denoise + dereverb + bandwidth extension).

Three-stage pipeline running on ONNX Runtime:
    16 kHz audio
      → mel_frontend.onnx (SeamlessM4T log-mel, bundled)
      → sidon-predictor.onnx (w2v-BERT 2.0, 8 layers)
      → sidon-vocoder.onnx (DAC decoder)
      → 48 kHz restored audio
"""

from __future__ import annotations

import importlib.resources
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from revospeech.registry import get

from ..tts.result import Audio
from .base import BaseUtil

logger = logging.getLogger(__name__)

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 48000
HOP_LENGTH = 160


def _resample_linear(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear-interpolation resample — matches revospeech.asr.audio convention."""
    if src_sr == dst_sr:
        return samples.astype(np.float32, copy=False)
    duration = len(samples) / src_sr
    target_len = int(duration * dst_sr)
    indices = np.linspace(0, len(samples) - 1, target_len)
    return np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)


def _bundled_mel_frontend_path() -> Path:
    assets = importlib.resources.files("revospeech.util.assets")
    return Path(str(assets / "mel_frontend.onnx"))


class SidonUtil(BaseUtil):
    """Speech restoration util model using Sidon ONNX."""

    def __init__(self, model_name: str = "sidon", device: str = "auto") -> None:
        super().__init__(model_name, device)
        self.manifest = get(model_name, "util")
        self._model_dir: Path | None = None
        self._mel_session: ort.InferenceSession | None = None
        self._predictor_session: ort.InferenceSession | None = None
        self._vocoder_session: ort.InferenceSession | None = None

    def _ensure_downloaded(self) -> Path:
        if self._model_dir is not None and (
            self._model_dir / self.manifest.files["predictor"]
        ).exists():
            return self._model_dir

        from revospeech.hf_utils import download_gated_model

        cache_dir = Path.home() / ".cache" / "revospeech" / self.manifest.name
        self._model_dir = cache_dir

        predictor_path = cache_dir / self.manifest.files["predictor"]
        if predictor_path.exists():
            return self._model_dir

        logger.info("Downloading Sidon ONNX model from %s", self.manifest.model_url)
        download_gated_model(self.manifest.model_url, cache_dir)
        return self._model_dir

    def _load_sessions(self) -> None:
        if self._predictor_session is not None:
            return

        model_dir = self._ensure_downloaded()
        predictor_path = model_dir / self.manifest.files["predictor"]
        vocoder_path = model_dir / self.manifest.files["vocoder"]
        mel_path = _bundled_mel_frontend_path()

        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._mel_session = ort.InferenceSession(
            str(mel_path), providers=providers
        )
        self._predictor_session = ort.InferenceSession(
            str(predictor_path), providers=providers
        )
        self._vocoder_session = ort.InferenceSession(
            str(vocoder_path), providers=providers
        )
        logger.info(
            "Sidon loaded (mel=%s, predictor=%s, vocoder=%s)",
            mel_path.name, predictor_path.name, vocoder_path.name,
        )

    def restore(self, audio: Audio) -> Audio:
        """Restore / enhance an Audio input. Returns Audio at 48 kHz."""
        self._load_sessions()

        samples = audio.samples.astype(np.float32, copy=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        if audio.sample_rate != INPUT_SAMPLE_RATE:
            samples = _resample_linear(samples, audio.sample_rate, INPUT_SAMPLE_RATE)

        # Pad so the framing yields a whole number of hop_length-aligned frames
        window_length = 400
        if len(samples) < window_length:
            samples = np.pad(samples, (0, window_length - len(samples)))
        remainder = (len(samples) - window_length) % HOP_LENGTH
        if remainder:
            samples = np.pad(samples, (0, HOP_LENGTH - remainder))

        # Stage 1: mel front-end
        mel_out = self._mel_session.run(
            ["features"], {"audio": samples},
        )[0]  # [1, T, 160]

        # Stage 2: predictor (w2v-BERT 2.0)
        pred_inputs = {"input_features": mel_out}
        try:
            pred_out = self._predictor_session.run(None, pred_inputs)
        except ort.capi.onnxruntime_pybind11_state.InvalidArgument:
            input_name = self._predictor_session.get_inputs()[0].name
            pred_out = self._predictor_session.run(None, {input_name: mel_out})
        # Predictor output: dict-like with last_hidden_state [1, T, 1024]
        if isinstance(pred_out, dict):
            hidden = pred_out.get("last_hidden_state", next(iter(pred_out.values())))
        elif len(pred_out) > 1:
            hidden = pred_out[0]
        else:
            hidden = pred_out[0]

        # Stage 3: DAC vocoder
        voc_inputs = {self._vocoder_session.get_inputs()[0].name: hidden}
        voc_out = self._vocoder_session.run(None, voc_inputs)[0]

        # Vocoder output shape: [1, 1, N'] or [1, N']
        enhanced = np.squeeze(voc_out)
        if enhanced.ndim > 1:
            enhanced = enhanced.mean(axis=0)
        enhanced = enhanced.astype(np.float32, copy=False)

        # Trim to original duration scaled to 48 kHz
        target_len = int(
            round(len(audio.samples) * OUTPUT_SAMPLE_RATE / audio.sample_rate)
        )
        if len(enhanced) > target_len:
            enhanced = enhanced[:target_len]
        elif len(enhanced) < target_len:
            enhanced = np.pad(enhanced, (0, target_len - len(enhanced)))

        # Smooth the speech-to-silence transition to suppress vocoder cutoff clicks.
        # The DAC vocoder can end mid-oscillation, producing a step from ~-0.7 to 0
        # in a single sample. Locate the last non-zero sample and fade-out across
        # the ~5 ms immediately preceding it so the natural decay ends at zero.
        abs_enh = np.abs(enhanced)
        nonzero_idx = np.where(abs_enh > 1e-6)[0]
        if nonzero_idx.size:
            last_nonzero = int(nonzero_idx[-1])
            fade_n = int(0.005 * OUTPUT_SAMPLE_RATE)  # 240 samples at 48 kHz
            fade_start = max(0, last_nonzero - fade_n + 1)
            n = last_nonzero - fade_start + 1
            if n > 1:
                fade = np.linspace(1.0, 0.0, n, dtype=np.float32)
                enhanced[fade_start : last_nonzero + 1] *= fade
            enhanced[last_nonzero + 1 :] = 0.0

        return Audio(samples=enhanced, sample_rate=OUTPUT_SAMPLE_RATE)
