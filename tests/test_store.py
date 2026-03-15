"""Tests for the in-memory history store."""

from app.store import HistoryStore


def test_save_and_all():
    store = HistoryStore()
    id1 = store.save("q1", "r1")
    id2 = store.save("q2", "r2")
    assert id1 == 1
    assert id2 == 2

    records = store.all()
    assert len(records) == 2
    assert records[0].query == "q1"
    assert records[1].response == "r2"


def test_all_returns_copy():
    store = HistoryStore()
    store.save("q", "r")
    records = store.all()
    records.clear()
    assert len(store.all()) == 1  # original unaffected


def test_empty_store():
    store = HistoryStore()
    assert store.all() == []
