import numpy as np

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.STT import paraformer_handler
from speech_to_speech.STT.paraformer_handler import ParaformerSTTHandler


class _FakeParaformerModel:
    def generate(self, audio, **kwargs):
        return [{"text": " 今 天 天 气 不 错 "}]


class _FakeSenseVoiceModel:
    """Simulates SenseVoiceSmall output which includes language/emotion tags."""

    def generate(self, audio, **kwargs):
        return [{"text": "<|ja|><|NEUTRAL|><|Speech|><|withitn|>今日 は 良い 天気 ですね"}]


def _handler(language=None):
    handler = object.__new__(ParaformerSTTHandler)
    handler.model = _FakeParaformerModel()
    handler.gen_kwargs = {}
    handler.language = language
    return handler


def _sensevoice_handler():
    handler = object.__new__(ParaformerSTTHandler)
    handler.model = _FakeSenseVoiceModel()
    handler.gen_kwargs = {}
    handler.language = "ja"
    return handler


def _silence_console(monkeypatch):
    monkeypatch.setattr(paraformer_handler.console, "print", lambda *args, **kwargs: None)


def test_progressive_paraformer_transcription_is_partial(monkeypatch):
    _silence_console(monkeypatch)

    result = list(
        _handler().process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="progressive",
                turn_id="turn_1",
                turn_revision=2,
            )
        )
    )

    assert len(result) == 1
    assert isinstance(result[0], PartialTranscription)
    assert result[0].text == "今天天气不错"
    assert result[0].turn_id == "turn_1"
    assert result[0].turn_revision == 2


def test_final_paraformer_transcription_is_final(monkeypatch):
    _silence_console(monkeypatch)

    result = list(
        _handler().process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="final",
                turn_id="turn_1",
                turn_revision=2,
                created_at_s=123.0,
            )
        )
    )

    assert len(result) == 1
    assert isinstance(result[0], Transcription)
    assert result[0].text == "今天天气不错"
    assert result[0].turn_id == "turn_1"
    assert result[0].turn_revision == 2
    assert result[0].speech_stopped_at_s == 123.0


def test_sensevoice_tags_are_stripped_from_japanese(monkeypatch):
    _silence_console(monkeypatch)

    result = list(
        _sensevoice_handler().process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="final",
                turn_id="turn_1",
                turn_revision=1,
                created_at_s=1.0,
            )
        )
    )

    assert len(result) == 1
    assert isinstance(result[0], Transcription)
    assert result[0].text == "今日は良い天気ですね"


def test_sensevoice_progressive_japanese_is_partial(monkeypatch):
    _silence_console(monkeypatch)

    result = list(
        _sensevoice_handler().process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="progressive",
                turn_id="turn_2",
                turn_revision=3,
            )
        )
    )

    assert len(result) == 1
    assert isinstance(result[0], PartialTranscription)
    assert result[0].text == "今日は良い天気ですね"
    assert result[0].turn_id == "turn_2"
    assert result[0].turn_revision == 3
