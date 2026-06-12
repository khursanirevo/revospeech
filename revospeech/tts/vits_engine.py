"""VITS TTS backend for Malay via ONNX Runtime (no piper-tts dependency)."""

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

PRODUCTION_SPEAKERS = ["sarah", "pendakwah", "pendakwah_teknologi", "paan", "anwar"]

_ESPEAK_PHONEMES = {
    "a",
    "b",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "z",
    "A:",
    "E",
    "I",
    "O",
    "U",
    "Y",
    "_",
    "dZ",
    "tS",
    "N",
    "?",
    "x",
}


def _phonemize_espeak(text: str, language: str = "ms") -> list[str]:
    try:
        result = subprocess.run(
            ["espeak-ng", "-x", "-v", language, "--phonout", "/dev/stdout", text],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        raise RuntimeError("espeak-ng not found. Install: sudo apt install espeak-ng")

    raw = result.stdout.strip()
    phonemes = []
    i = 0
    while i < len(raw):
        if raw[i] == " ":
            i += 1
            continue
        if raw[i] == "_":
            phonemes.append("^")
            i += 1
            continue
        if raw[i] == "|":
            phonemes.append("_")
            i += 1
            continue
        matched = False
        for length in (3, 2, 1):
            if i + length <= len(raw):
                chunk = raw[i : i + length]
                if chunk in _ESPEAK_PHONEMES:
                    phonemes.append(chunk)
                    i += length
                    matched = True
                    break
        if not matched:
            i += 1

    return phonemes


def _normalize_text_simple(text: str) -> str:
    try:
        from revo_norm import normalize_text

        return normalize_text(text, language="ms")
    except ImportError:
        return text


class VitsTTS(BaseTTS):
    """Malay TTS using VITS via ONNX Runtime."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        super().__init__(model_name, device)
        self.manifest = get(model_name, "tts")
        self._models_dir: Path | None = None
        self._loaded_speakers: dict[str, ort.InferenceSession] = {}
        self._phoneme_maps: dict[str, dict[str, int]] = {}

    def _ensure_repo(self) -> Path:
        if self._models_dir is not None:
            return self._models_dir

        from huggingface_hub import snapshot_download

        cache_dir = Path.home() / ".cache" / "revospeech" / self.manifest.name
        self._models_dir = cache_dir

        if (cache_dir / "speakers.json").exists():
            return self._models_dir

        logger.info("Downloading VITS model from %s", self.manifest.model_url)
        snapshot_download(
            repo_id=self.manifest.model_url,
            local_dir=cache_dir,
        )
        return self._models_dir

    def _load_speaker(self, speaker: str):
        if speaker in self._loaded_speakers:
            return self._loaded_speakers[speaker], self._phoneme_maps[speaker]

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

        phoneme_map = {}
        for phoneme, ids in config.get("phoneme_id_map", {}).items():
            if ids:
                phoneme_map[phoneme] = ids[0]

        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._loaded_speakers[speaker] = sess
        self._phoneme_maps[speaker] = phoneme_map
        logger.info("Loaded VITS speaker '%s'", speaker)
        return sess, phoneme_map

    def synthesize(
        self,
        text: str,
        output_path: str | None = None,
        *,
        speed: float = 1.0,
        speaker: str = "sarah",
        **kwargs,
    ) -> Audio:
        text = _normalize_text_simple(text)
        phonemes = _phonemize_espeak(text)

        sess, phoneme_map = self._load_speaker(speaker)

        ids = [phoneme_map.get("^", 0)]
        for p in phonemes:
            ids.append(phoneme_map.get(p, 0))
        ids.append(phoneme_map.get("$", 0))

        input_ids = np.array([ids], dtype=np.int64)
        input_len = np.array([len(ids)], dtype=np.int64)
        scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)

        output = sess.run(
            None,
            {"input": input_ids, "input_lengths": input_len, "scales": scales},
        )

        samples = output[0].flatten().astype(np.float32)
        sr = self.manifest.sample_rate
        audio = Audio(samples=samples, sample_rate=sr)

        if output_path:
            audio.save(output_path)

        return audio
