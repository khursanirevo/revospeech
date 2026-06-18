"""VITS TTS backend for Malay via ONNX Runtime (Piper-compatible phonemization)."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort

from revospeech.registry import get

from .base import BaseTTS
from .result import Audio

logger = logging.getLogger(__name__)

PRODUCTION_SPEAKERS = ["sarah", "paan", "anwar"]

BOS = "^"
EOS = "$"
PAD = "_"
_TIE_BAR = "‍"


def _phonemize_espeak(text: str, language: str = "ms") -> list[list[str]]:
    """Phonemize text into sentences of phonemes using espeak-ng IPA output.

    Returns list of sentences, each sentence is a list of phoneme strings.
    Matches piper_phonemize.phonemize_espeak() behavior.
    """
    try:
        result = subprocess.run(
            ["espeak-ng", "-x", "--ipa=3", "-q", "-v", language, text],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        raise RuntimeError("espeak-ng not found. Install: sudo apt install espeak-ng")

    sentences: list[list[str]] = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        phonemes: list[str] = []
        for char in line:
            if char == _TIE_BAR:
                continue
            if char in (" ", "\t"):
                phonemes.append(" ")
                continue
            phonemes.append(char)
        if phonemes:
            sentences.append(phonemes)
    return sentences


def _phonemes_to_ids(
    phonemes: list[str],
    id_map: dict[str, list[int]],
) -> list[int]:
    """Convert phonemes to ID sequence with PAD between each (Piper-compatible)."""
    ids = list(id_map.get(BOS, [1]))
    pad_ids = id_map.get(PAD, [0])
    for phoneme in phonemes:
        phoneme_ids = id_map.get(phoneme)
        if phoneme_ids is None:
            logger.warning("Missing phoneme from id map: %s", phoneme)
            continue
        ids.extend(phoneme_ids)
        ids.extend(pad_ids)
    ids.extend(id_map.get(EOS, [2]))
    return ids


def _normalize_text_simple(text: str) -> str:
    try:
        from revo_norm import normalize_text

        return normalize_text(text, language="ms")
    except ImportError:
        logger.warning(
            "revo_norm not installed — text normalization skipped. "
            "Abbreviations and numbers may not render correctly. "
            "Install with: pip install revo-norm"
        )
        return text


def _audio_float_to_int16(audio: np.ndarray) -> np.ndarray:
    max_wav = 32767.0
    audio_norm = audio * (max_wav / max(0.01, np.max(np.abs(audio))))
    return np.clip(audio_norm, -max_wav, max_wav).astype("int16")


class VitsTTS(BaseTTS):
    """Malay TTS using VITS via ONNX Runtime."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        super().__init__(model_name, device)
        self.manifest = get(model_name, "tts")
        self._models_dir: Path | None = None
        self._loaded_speakers: dict[str, ort.InferenceSession] = {}
        self._phoneme_maps: dict[str, dict[str, list[int]]] = {}
        self._speaker_configs: dict[str, dict] = {}
        self._fallback_phoneme_map: dict[str, list[int]] | None = None

    def _ensure_repo(self) -> Path:
        if self._models_dir is not None:
            return self._models_dir

        from revospeech.hf_utils import download_gated_model

        cache_dir = Path.home() / ".cache" / "revospeech" / self.manifest.name
        self._models_dir = cache_dir

        if (cache_dir / "speakers.json").exists():
            return self._models_dir

        logger.info("Downloading VITS model from %s", self.manifest.model_url)
        download_gated_model(self.manifest.model_url, cache_dir)
        return self._models_dir

    def _get_fallback_phoneme_map(self, models_dir: Path) -> dict[str, list[int]]:
        if self._fallback_phoneme_map is not None:
            return self._fallback_phoneme_map
        for spk in PRODUCTION_SPEAKERS:
            cfg_path = models_dir / "speakers" / spk / "model.onnx.json"
            if not cfg_path.exists():
                continue
            with open(cfg_path) as f:
                config = json.load(f)
            raw = config.get("phoneme_id_map", {})
            if raw:
                self._fallback_phoneme_map = raw
                return self._fallback_phoneme_map
        return {}

    def _load_speaker(self, speaker: str):
        if speaker in self._loaded_speakers:
            return (
                self._loaded_speakers[speaker],
                self._phoneme_maps[speaker],
                self._speaker_configs.get(speaker, {}),
            )

        if speaker not in PRODUCTION_SPEAKERS:
            raise ValueError(
                f"Unknown speaker '{speaker}'. "
                f"Production speakers: {', '.join(PRODUCTION_SPEAKERS)}"
            )

        models_dir = self._ensure_repo()
        model_path = models_dir / "speakers" / speaker / "model.onnx"
        config_path = models_dir / "speakers" / speaker / "model.onnx.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Speaker model not found: {model_path}. "
                f"Ensure the model is downloaded first."
            )

        with open(config_path) as f:
            config = json.load(f)

        raw_pmap = config.get("phoneme_id_map", {})
        if not raw_pmap:
            raw_pmap = self._get_fallback_phoneme_map(models_dir)

        phoneme_map = {p: ids for p, ids in raw_pmap.items() if ids}

        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._loaded_speakers[speaker] = sess
        self._phoneme_maps[speaker] = phoneme_map
        self._speaker_configs[speaker] = config
        logger.info("Loaded VITS speaker '%s' (phonemes=%d)", speaker, len(phoneme_map))
        return sess, phoneme_map, config

    def _synthesize_ids(
        self,
        sess: ort.InferenceSession,
        phoneme_ids: list[int],
        config: dict,
        speed: float = 1.0,
    ) -> np.ndarray:
        inference = config.get("inference", {})
        noise_scale = inference.get("noise_scale", 0.667)
        length_scale = inference.get("length_scale", 1.0)
        noise_w = inference.get("noise_w", 0.8)

        input_ids = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        input_len = np.array([input_ids.shape[1]], dtype=np.int64)
        scales = np.array(
            [noise_scale, length_scale / speed, noise_w],
            dtype=np.float32,
        )

        output = sess.run(
            None,
            {"input": input_ids, "input_lengths": input_len, "scales": scales},
        )
        return output[0].squeeze()

    def synthesize(
        self,
        text: str,
        output_path: str | Path | None = None,
        *,
        speed: float = 1.0,
        speaker: str = "sarah",
        sentence_silence: float = 0.0,
        **kwargs,
    ) -> Audio:
        text = _normalize_text_simple(text)
        sentences = _phonemize_espeak(text)

        sess, phoneme_map, config = self._load_speaker(speaker)
        sr = self.manifest.sample_rate

        all_samples: list[np.ndarray] = []
        num_silence_samples = int(sentence_silence * sr)

        for phonemes in sentences:
            phoneme_ids = _phonemes_to_ids(phonemes, phoneme_map)
            chunk = self._synthesize_ids(sess, phoneme_ids, config, speed=speed)
            all_samples.append(chunk)
            if num_silence_samples > 0:
                all_samples.append(np.zeros(num_silence_samples, dtype=np.float32))

        if all_samples:
            samples = np.concatenate(all_samples).astype(np.float32)
        else:
            samples = np.zeros(0, dtype=np.float32)

        audio = Audio(samples=samples, sample_rate=sr)

        if output_path:
            audio.save(output_path)

        return audio
