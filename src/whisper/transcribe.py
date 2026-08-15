"""CUDA WhisperX transcription with batched ASR and word alignment."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.whisper.quality import strip_introductions


CONSERVATIVE_ASR_OPTIONS = {
    "beam_size": 5,
    "patience": 1.0,
    "length_penalty": 1.0,
    "repetition_penalty": 1.12,
    "no_repeat_ngram_size": 3,
    "temperatures": [0.0],
    "compression_ratio_threshold": 2.0,
    "log_prob_threshold": -0.8,
    "no_speech_threshold": 0.5,
    "condition_on_previous_text": False,
}
VAD_OPTIONS = {"chunk_size": 15}


class WhisperXSession:
    """Reusable CUDA WhisperX pipeline; load ASR and alignment models once."""

    def __init__(self, *, model: str, model_dir: Path, alignment_model_dir: Path,
                 batch_size: int = 16, force_align_words: bool = True,
                 language: str = "sv") -> None:
        import whisperx

        self._whisperx = whisperx
        self._batch_size = batch_size
        self._force_align_words = force_align_words
        self._asr_model = whisperx.load_model(
            model,
            device="cuda",
            compute_type="float16",
            language=language,
            vad_method="silero",
            vad_options=VAD_OPTIONS,
            asr_options=CONSERVATIVE_ASR_OPTIONS,
            download_root=str(model_dir),
        )
        self._alignment_model = None
        self._alignment_metadata = None
        if force_align_words:
            self._alignment_model, self._alignment_metadata = whisperx.load_align_model(
                language_code=language, device="cuda", model_dir=str(alignment_model_dir)
            )

    def transcribe(self, audio_path: Path) -> dict[str, Any]:
        audio = self._whisperx.load_audio(str(audio_path))
        transcript = self._asr_model.transcribe(audio, batch_size=self._batch_size)
        if self._force_align_words:
            transcript = self._whisperx.align(
                transcript["segments"], self._alignment_model,
                self._alignment_metadata, audio, device="cuda"
            )
        strip_introductions(transcript)
        return transcript


def transcribe_audio_whisperx(*, audio_path: Path, output_path: Path,
                              model: str, model_dir: Path, alignment_model_dir: Path,
                              batch_size: int = 16, force_align_words: bool = True,
                              language: str = "sv") -> dict[str, Any]:
    """Transcribe one episode with CUDA WhisperX and optional word alignment."""
    session = WhisperXSession(
        model=model, model_dir=model_dir, alignment_model_dir=alignment_model_dir,
        batch_size=batch_size, force_align_words=force_align_words, language=language,
    )
    transcript = session.transcribe(audio_path)
    transcript.setdefault("sommar_i_parquet", {}).update({
        "audio_path": str(audio_path),
        "created_at": datetime.now(UTC).isoformat(),
        "engine": "whisperx (KB-Whisper-large FP16 + Silero VAD + wav2vec2 alignment)"
        if force_align_words else "whisperx (KB-Whisper-large FP16 + Silero VAD)",
        "model": model,
        "model_dir": str(model_dir),
        "alignment_model_dir": str(alignment_model_dir),
        "batch_size": batch_size,
        "force_align_words": force_align_words,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(transcript, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return transcript
