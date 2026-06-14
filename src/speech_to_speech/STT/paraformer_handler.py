from __future__ import annotations

import logging
import re
import torch
from typing import Any, Iterator, Optional

import numpy as np
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()

# Matches SenseVoice output tokens like <|ja|>, <|NEUTRAL|>, <|Speech|>, <|withitn|>
_SENSEVOICE_TAG_RE = re.compile(r"<\|[^|]+\|>")


class ParaformerSTTHandler(BaseSTTHandler):
    """
    Handles the Speech To Text generation using a Paraformer or SenseVoice model.
    The default for this model is set to Chinese (paraformer-zh).
    For Japanese with live transcription support, use FunAudioLLM/SenseVoiceSmall with hub='hf'.
    This model was contributed by @wuhongsheng.
    """

    def setup(
        self,
        model_name: str = "paraformer-zh",
        device: str = "cuda",
        language: Optional[str] = None,
        hub: Optional[str] = None,
        gen_kwargs: dict[str, Any] = {},
    ) -> None:
        self.device = device
        self.language = language
        self.gen_kwargs = gen_kwargs
        try:
            from funasr import AutoModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Paraformer STT requires the optional 'paraformer' extra. "
                "Install it with `pip install speech-to-speech[paraformer]`."
            ) from exc

        model_kwargs: dict[str, Any] = {"model": model_name, "device": device, "disable_update": True}
        if hub:
            model_kwargs["hub"] = hub

        logger.info(f"Loading Paraformer/SenseVoice model: {model_name} (hub={hub}, language={language})")
        self.model = AutoModel(**model_kwargs)
        self.warmup()

    def _generate(self, audio: np.ndarray) -> str:
        kwargs: dict[str, Any] = {**self.gen_kwargs}
        if self.language:
            kwargs["language"] = self.language
            kwargs["use_itn"] = True
        raw = self.model.generate(audio, **kwargs)[0]["text"]
        # Strip SenseVoice special tokens (no-op for plain Paraformer output)
        return _SENSEVOICE_TAG_RE.sub("", raw).strip().replace(" ", "")

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")
        dummy_input = np.zeros(16000, dtype=np.float32)
        _ = self._generate(dummy_input)

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        logger.debug("infering paraformer...")

        pred_text = self._generate(vad_audio.audio)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        logger.debug("finished paraformer inference")
        console.print(f"[yellow]USER: {pred_text}")

        if vad_audio.mode == "progressive":
            yield PartialTranscription(
                text=pred_text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )
        else:
            yield Transcription(
                text=pred_text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                speech_stopped_at_s=vad_audio.created_at_s,
            )
