import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_stt_result_has_fields():
    from jarvis.stt_engine import STTResult
    result = STTResult(text="hello world", segments=[("hello world", -0.3)], language="en")
    assert result.text == "hello world"
    assert result.segments == [("hello world", -0.3)]
    assert result.language == "en"


def test_stt_result_avg_logprob():
    from jarvis.stt_engine import STTResult
    result = STTResult(
        text="hello world",
        segments=[("hello", -0.2), ("world", -0.4)],
        language="en",
    )
    assert abs(result.avg_logprob - (-0.3)) < 0.01


def test_stt_result_empty():
    from jarvis.stt_engine import STTResult
    result = STTResult(text="", segments=[], language="en")
    assert result.text == ""
    assert result.avg_logprob == 0.0
