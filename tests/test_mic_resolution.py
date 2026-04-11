import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_resolve_mic_by_name_exact():
    from jarvis.audio_pipeline import resolve_mic_by_name
    devices = [
        {"name": "HDA NVidia: HDMI 0", "max_input_channels": 0},
        {"name": "USB Audio: Blue Snowball", "max_input_channels": 2},
        {"name": "pipewire", "max_input_channels": 64},
    ]
    assert resolve_mic_by_name("Blue Snowball", devices) == 1

def test_resolve_mic_by_name_substring():
    from jarvis.audio_pipeline import resolve_mic_by_name
    devices = [
        {"name": "HDA NVidia", "max_input_channels": 0},
        {"name": "USB Audio: - (hw:2,0)", "max_input_channels": 2},
        {"name": "default", "max_input_channels": 64},
    ]
    assert resolve_mic_by_name("USB Audio", devices) == 1

def test_resolve_mic_case_insensitive():
    from jarvis.audio_pipeline import resolve_mic_by_name
    devices = [{"name": "Blue Snowball iCE", "max_input_channels": 1}]
    assert resolve_mic_by_name("blue snowball", devices) == 0

def test_resolve_mic_not_found():
    from jarvis.audio_pipeline import resolve_mic_by_name
    devices = [{"name": "pipewire", "max_input_channels": 64}]
    assert resolve_mic_by_name("Blue Snowball", devices) is None

def test_resolve_mic_skips_output_only():
    from jarvis.audio_pipeline import resolve_mic_by_name
    devices = [
        {"name": "Blue Snowball Output", "max_input_channels": 0},
        {"name": "Blue Snowball Input", "max_input_channels": 2},
    ]
    assert resolve_mic_by_name("Blue Snowball", devices) == 1

def test_resolve_mic_default_returns_none():
    from jarvis.audio_pipeline import resolve_mic_by_name
    assert resolve_mic_by_name("Default", []) is None
    assert resolve_mic_by_name(None, []) is None
    assert resolve_mic_by_name("", []) is None
