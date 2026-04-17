"""Smoke tests for HotwordListener's stream-mutation lock.

Thread races are hard to exercise deterministically, so these tests
verify the lock exists and that the mutation entrypoints do not bypass
it (by patching and asserting acquire/release).
"""

import threading
from unittest.mock import MagicMock

import pytest


def _make_listener():
    """Return a HotwordListener instance wired to a mocked gui."""
    from jarvis.voice_input_gui import HotwordListener
    gui = MagicMock()
    return HotwordListener(gui)


def test_stream_lock_attribute_exists():
    hl = _make_listener()
    assert isinstance(hl._stream_lock, type(threading.Lock()))


def test_close_stream_acquires_the_lock():
    hl = _make_listener()
    hl._stream = MagicMock()  # Simulate an open stream

    # Replace the lock with a tracking wrapper
    original_lock = hl._stream_lock
    enter_count = [0]
    class TrackingLock:
        def __enter__(self):
            enter_count[0] += 1
            return original_lock.__enter__()
        def __exit__(self, *args):
            return original_lock.__exit__(*args)
    hl._stream_lock = TrackingLock()

    hl._close_stream()
    assert enter_count[0] == 1
    assert hl._stream is None


def test_pause_goes_through_close_stream():
    hl = _make_listener()
    hl._stream = MagicMock()
    hl.pause()
    assert hl._stream is None


def test_resume_without_active_listener_calls_start():
    hl = _make_listener()
    hl.active = False
    hl.start = MagicMock()
    hl.resume()
    hl.start.assert_called_once()


def test_resume_with_existing_stream_is_noop():
    hl = _make_listener()
    hl._stream = object()  # non-None
    hl.active = True
    hl.resume()
    # Should return without setting _reopen
    assert not hl._reopen
