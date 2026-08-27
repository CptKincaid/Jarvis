"""Tests for jarvis.events — thread-safe publish, ordering, subscriptions."""
import threading

from jarvis.events import Bus, JarvisReply, Status


def test_publish_without_root_delivers_inline():
    b = Bus()
    got = []
    b.subscribe(Status, got.append)
    b.publish(Status(text="hello", kind="ok"))
    assert len(got) == 1 and got[0].text == "hello"


def test_events_from_thread_drain_in_order():
    b = Bus()
    b._root = object()          # simulate an attached UI: defer delivery
    got = []
    b.subscribe(Status, got.append)

    def worker():
        for i in range(100):
            b.publish(Status(text=str(i)))

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert got == []            # nothing delivered until the main loop drains

    b.drain()                   # fake main-loop pump
    assert [e.text for e in got] == [str(i) for i in range(100)]


def test_interleaved_threads_all_delivered():
    b = Bus()
    b._root = object()
    got = []
    b.subscribe(Status, got.append)

    threads = [
        threading.Thread(
            target=lambda tid=tid: [b.publish(Status(text=f"{tid}-{i}"))
                                    for i in range(50)])
        for tid in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    b.drain()
    assert len(got) == 200
    # per-thread ordering is preserved even when interleaved
    for tid in range(4):
        mine = [e.text for e in got if e.text.startswith(f"{tid}-")]
        assert mine == [f"{tid}-{i}" for i in range(50)]


def test_subscribers_are_type_exact():
    b = Bus()
    statuses, replies = [], []
    b.subscribe(Status, statuses.append)
    b.subscribe(JarvisReply, replies.append)
    b.publish(JarvisReply(text="hi", speak=False))
    assert statuses == []
    assert len(replies) == 1


def test_failing_subscriber_does_not_break_others():
    b = Bus()
    got = []

    def bad(ev):
        raise RuntimeError("boom")

    b.subscribe(Status, bad)
    b.subscribe(Status, got.append)
    b.publish(Status(text="still delivered"))
    assert len(got) == 1
    # and the bus keeps working afterwards
    b.publish(Status(text="again"))
    assert len(got) == 2


def test_unsubscribe():
    b = Bus()
    got = []
    b.subscribe(Status, got.append)
    b.unsubscribe(Status, got.append)
    b.publish(Status(text="x"))
    assert got == []
    # unsubscribing a never-registered fn is harmless
    b.unsubscribe(Status, lambda e: None)
