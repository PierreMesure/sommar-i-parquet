"""WhisperMLX transcription with Silero VAD and optional word alignment."""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.whisper.quality import strip_boilerplate, strip_recommended_episode_extract


class WhisperMLXSession:
    """Reusable WhisperMLX pipeline; load the q4 model only once."""

    def __init__(self, *, model_path: Path, alignment_model_dir: Path,
                 force_align_words: bool = False, language: str = "sv") -> None:
        import mlx.core as mx

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            import whispermlx
            self._whispermlx = whispermlx
            self._force_align_words = force_align_words
            self._asr_model = whispermlx.load_model(
                str(model_path), device="cpu", language=language, vad_method="silero"
            )
            self._alignment_model = None
            self._alignment_metadata = None
            if force_align_words:
                self._alignment_model, self._alignment_metadata = whispermlx.load_align_model(
                    language_code=language, device="cpu", model_dir=str(alignment_model_dir)
                )
        mx.clear_cache()

    def transcribe(self, audio_path: Path) -> dict[str, Any]:
        transcript = self._asr_model.transcribe(str(audio_path))
        if self._force_align_words:
            transcript = self._whispermlx.align(
                transcript["segments"], self._alignment_model,
                self._alignment_metadata, str(audio_path), device="cpu"
            )
        strip_recommended_episode_extract(transcript)
        strip_boilerplate(transcript)
        return transcript


def transcribe_audio_whispermlx(*, audio_path: Path, output_path: Path,
                                model_path: Path, alignment_model_dir: Path,
                                force_align_words: bool = False,
                                language: str = "sv") -> dict[str, Any]:
    """Transcribe one episode with WhisperMLX and Silero VAD."""
    session = WhisperMLXSession(
        model_path=model_path, alignment_model_dir=alignment_model_dir,
        force_align_words=force_align_words, language=language,
    )
    transcript = session.transcribe(audio_path)
    transcript.setdefault("sommar_i_parquet", {}).update({
        "audio_path": str(audio_path),
        "created_at": datetime.now(UTC).isoformat(),
        "engine": "whispermlx (MLX ASR + Silero VAD + wav2vec2 alignment)"
        if force_align_words else "whispermlx (MLX ASR + Silero VAD)",
        "model_path": str(model_path),
        "alignment_model_dir": str(alignment_model_dir),
        "force_align_words": force_align_words,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(transcript, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return transcript
