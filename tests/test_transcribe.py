from src.whisper.transcribe import CONSERVATIVE_ASR_OPTIONS, VAD_OPTIONS


def test_cuda_transcription_uses_conservative_decoding():
    assert CONSERVATIVE_ASR_OPTIONS["condition_on_previous_text"] is False
    assert CONSERVATIVE_ASR_OPTIONS["repetition_penalty"] > 1
    assert CONSERVATIVE_ASR_OPTIONS["no_repeat_ngram_size"] == 3
    assert CONSERVATIVE_ASR_OPTIONS["temperatures"] == [0.0]
    assert VAD_OPTIONS["chunk_size"] == 15
