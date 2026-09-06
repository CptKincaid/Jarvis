"""``presence.desk`` IS NOT A ROOM NAME. It is a boolean feature switch.

Found by reading his LIVE config through ``AssistantConfig.load()``
(2026-09-05): ``presence.desk`` is ``True`` on his box, because
``jarvis/deskpresence.py`` owns that key as its on/off switch
(``bool(_cfg_get(self._cfg, "presence.desk", True))``). The room name lives
under ``presence.desk_room``, which is what ``app._desk`` has always read
(``arrival.DEFAULT_DESK_ROOM`` = "office").

Reading the switch as a room name gives ``str(True)`` -> "True", and two
things quietly die on HIS box and nowhere else:

  * HIS DEPARTURE RULE never arms. ``DepartureSequence`` compares the
    RoomChanged room against ``desk_room``; "office" never equals "True",
    so the sequence can never reach its first stage and "office then
    kitchen then the phone drops" can never complete.
  * THE BEDROOM SPLIT loses its office branch. Cells 11 and 12 with the
    office seen last would answer BED ("he is in the bedroom") instead of
    HOME ("he never left the office and the radar dropped a still body")
    -- the one branch whose whole point is that it contradicts his rule 2.

This is the same shape of defect as the one that made an earlier session
report the phone leg as unconfigured: it read ``presence.phone`` when the
key is ``presence.phone_ip``. A default that looks sensible hides a key
that is never actually read.
"""
from __future__ import annotations

from jarvis import app as app_mod, arrival as arrival_mod, presence


class Cfg:
    """AssistantConfig's read shape, with HIS values."""

    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


def his_config(**extra):
    """His box: presence.desk is the BOOLEAN, and no desk_room is set."""
    values = {
        "presence.room_sensor_enabled": True,
        "presence.rooms": [{"name": "office", "url": "http://192.168.50.51"},
                           {"name": "kitchen", "url": "http://192.168.50.52"}],
        "presence.phone_ip": "192.168.50.34",
        "presence.desk": True,
        "presence.door_room": "kitchen",
    }
    values.update(extra)
    return Cfg(**values)


def test_the_voter_takes_the_desk_room_from_desk_room_not_the_switch():
    s = presence.PresenceSentinel(his_config(), publish=lambda e: None)
    assert s.legs is not None, "his config has rooms, so the voter must build"
    assert s.legs.desk_room == "office"
    assert s.legs.desk_room != "True"


def test_an_explicit_desk_room_is_honoured_by_the_voter():
    s = presence.PresenceSentinel(his_config(**{"presence.desk_room": "study"}),
                                  publish=lambda e: None)
    assert s.legs.desk_room == "study"


def test_the_departure_sequence_takes_the_desk_room_from_desk_room():
    a = object.__new__(app_mod.JarvisApp)
    a.assistant = his_config()
    seq = a._make_departure_seq()
    assert seq is not None
    assert seq.desk_room == "office"
    assert seq.desk_room != "True"


def test_his_departure_rule_actually_arms_on_his_own_config():
    """The end of the bug, stated as the behaviour it cost him: office,
    then kitchen, then the phone drops, on the config he really has."""
    a = object.__new__(app_mod.JarvisApp)
    a.assistant = his_config()
    seq = a._make_departure_seq()
    seq.room(room="office", at=0.0)
    seq.room(room="kitchen", at=20.0)
    assert seq.phone_gone(at=800.0) is True
    assert seq.left is True


def test_the_desk_room_default_is_the_one_arrival_already_owns():
    a = object.__new__(app_mod.JarvisApp)
    a.assistant = Cfg(**{"presence.rooms": [{"name": "office",
                                             "url": "http://a"}],
                         "presence.room_sensor_enabled": True})
    seq = a._make_departure_seq()
    assert seq.desk_room == arrival_mod.DEFAULT_DESK_ROOM


def test_the_boolean_switch_is_left_alone_for_its_real_owner(monkeypatch):
    """deskpresence.py reads presence.desk as a bool and must keep it.

    The env override is cleared so this asks about the CONFIG KEY and not
    about whatever ``JARVIS_DESK_PRESENCE`` happens to be set to here.
    """
    from jarvis import deskpresence
    monkeypatch.delenv(deskpresence.ENV_OFF, raising=False)
    watch = object.__new__(deskpresence.DeskSentinel)
    watch._cfg = his_config()
    assert watch.enabled is True
