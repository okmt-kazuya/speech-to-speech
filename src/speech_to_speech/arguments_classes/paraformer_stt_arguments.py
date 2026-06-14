from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParaformerSTTHandlerArguments:
    paraformer_stt_model_name: str = field(
        default="paraformer-zh",
        metadata={
            "help": "The pretrained model to use. Default is 'paraformer-zh'. "
            "For Japanese with live transcription, use 'FunAudioLLM/SenseVoiceSmall' with hub='hf'. "
            "See https://github.com/modelscope/FunASR for available models."
        },
    )
    paraformer_stt_device: str = field(
        default="cuda",
        metadata={"help": "The device type on which the model will run. Default is 'cuda' for GPU acceleration."},
    )
    paraformer_stt_language: Optional[str] = field(
        default=None,
        metadata={
            "help": "Language code for transcription (e.g. 'ja', 'zh', 'en', 'auto'). "
            "Required for SenseVoice multilingual models. Default is None (model default)."
        },
    )
    paraformer_stt_hub: Optional[str] = field(
        default=None,
        metadata={
            "help": "Model hub to load from. Use 'hf' to load from HuggingFace (e.g. for FunAudioLLM/SenseVoiceSmall). "
            "Default is None (ModelScope)."
        },
    )
