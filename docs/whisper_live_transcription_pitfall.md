# WhisperSTT と `--enable_live_transcription` の組み合わせは使用禁止

## 概要

`--stt whisper`（または `faster-whisper`）使用時に `--enable_live_transcription` を指定すると、
**ユーザーの発話全体がLLMに届かず、断片的な転写結果しか渡されない**バグが発生する。

---

## 正しい設定

### 日本語対応（Whisper）

```bash
speech-to-speech \
    --stt whisper \
    --stt_model_name openai/whisper-large-v3-turbo \
    --language ja \
    ...
    # --enable_live_transcription は指定しない
```

### ライブ文字起こしが必要な場合

ライブ文字起こしに対応しているのは `parakeet-tdt` のみ。ただし `parakeet-tdt` は日本語非対応。

```bash
speech-to-speech \
    --stt parakeet-tdt \
    ...
    --enable_live_transcription  # parakeet-tdt のみ有効
```

---

## 原因の詳細

### STTモデルのライブ文字起こし対応状況

| STTモデル | 日本語対応 | ライブ文字起こし対応 |
|-----------|-----------|-------------------|
| `parakeet-tdt` | ✗（欧州25言語のみ） | ✓ |
| `whisper` | ✓ | ✗ |
| `faster-whisper` | ✓ | ✗ |
| `paraformer` (SenseVoiceSmall) | ✓ | ✓ |

**日本語 × ライブ文字起こしを両立する唯一の選択肢は `paraformer` + `FunAudioLLM/SenseVoiceSmall`。**

```bash
speech-to-speech \
    --stt paraformer \
    --paraformer_stt_model_name FunAudioLLM/SenseVoiceSmall \
    --paraformer_stt_hub hf \
    --paraformer_stt_language ja \
    --enable_live_transcription \
    ...
```

SenseVoice は出力テキストに `<|ja|><|NEUTRAL|><|Speech|><|withitn|>` 等のタグを付加するが、
`ParaformerSTTHandler` 内の `_SENSEVOICE_TAG_RE` で自動除去される。

> **注意**: `hub` を省略すると ModelScope（中国サーバー）からダウンロードされ非常に遅い。
> `--paraformer_stt_hub hf` を必ず指定すること（HuggingFace 経由で高速ダウンロード）。

### バグの発生メカニズム

`--enable_live_transcription` を有効にすると、VAD (`vad_handler.py`) が
`enable_realtime_transcription=True` となり、発話中に短い区間の音声を
**`mode="progressive"`** でSTTキューへ送り続ける。
発話終了後に **`mode="final"`**（全体音声）を送る。

```
VAD → STTキュー:
  VADAudio(mode="progressive", turn_id="turn_1", turn_revision=0)  ← 0.5秒
  VADAudio(mode="progressive", turn_id="turn_1", turn_revision=0)  ← 1.0秒
  VADAudio(mode="final",       turn_id="turn_1", turn_revision=0)  ← 全体
```

**`WhisperSTTHandler.process()`** は `vad_audio.mode` を判定せず、
progressiveチャンクを最終音声と同様に `model.generate()` にかけて
`Transcription` オブジェクトを返す。

**`BaseSTTHandler.before_emit_output()`** (`base_stt_handler.py:73`) は
`Transcription` を出力するたびにその `(turn_id, turn_revision)` を
「処理済み」としてマークする。

```python
def before_emit_output(self, output: STTOut) -> None:
    if isinstance(output, Transcription):
        self._mark_completed_final_revision(output)
```

その結果、後から届く `mode="final"` の全体音声は
`should_process_input()` (`base_stt_handler.py:28`) で弾かれ**ドロップされる**。

```python
def should_process_input(self, item: STTIn) -> bool:
    if self._is_completed_final_revision(item):
        return False  # ← final音声がここで捨てられる
```

### 最終的な影響

- LLMに届くのは最初のprogressiveチャンク（0.5秒分）の転写結果のみ
- 発話全体の最終転写結果は捨てられる
- 応答が文脈を欠いたものになる、または無応答になる

---

## 関連ファイル

- `src/speech_to_speech/STT/whisper_stt_handler.py` — `mode` 未チェック
- `src/speech_to_speech/STT/base_stt_handler.py` — `_mark_completed_final_revision` / `_is_completed_final_revision`
- `src/speech_to_speech/VAD/vad_handler.py` — progressive/final送出ロジック
- `src/speech_to_speech/STT/parakeet_tdt_handler.py` — progressiveを正しく処理する参照実装
