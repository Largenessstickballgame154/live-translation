import queue

from live_translate_overlay import post_source_and_enqueue_translation, release_mlx_whisper_model


class RecordingOverlay:
    def __init__(self):
        self.sources = []

    def post_source(self, source, pause_ms=0):
        self.sources.append((source, pause_ms))


def test_finalized_transcript_is_posted_before_translation_queue_drop():
    translation_q = queue.Queue(maxsize=1)
    translation_q.put_nowait({"source": "old untranslated block", "pause_ms": 0.0})
    overlay = RecordingOverlay()

    post_source_and_enqueue_translation(
        translation_q,
        overlay,
        "new finalized transcript",
        pause_ms=1500.0,
        source_language="en",
    )

    assert overlay.sources == [("new finalized transcript", 1500.0)]
    queued = translation_q.get_nowait()
    assert queued == {
        "source": "new finalized transcript",
        "pause_ms": 1500.0,
        "source_language": "en",
    }


def test_release_mlx_whisper_model_clears_matching_holder():
    from mlx_whisper.transcribe import ModelHolder

    previous_model = ModelHolder.model
    previous_path = ModelHolder.model_path
    try:
        ModelHolder.model = object()
        ModelHolder.model_path = "mlx-community/whisper-small-mlx"

        assert release_mlx_whisper_model("mlx-community/whisper-small-mlx") is True
        assert ModelHolder.model is None
        assert ModelHolder.model_path is None
    finally:
        ModelHolder.model = previous_model
        ModelHolder.model_path = previous_path
