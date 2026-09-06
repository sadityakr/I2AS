import pytest

from i2as.core.gates import Gate


def test_rejects_empty_name():
    with pytest.raises(ValueError):
        Gate("")


def test_rejects_negative_window():
    with pytest.raises(ValueError):
        Gate("g", check=lambda: True, window_s=-1.0)


def test_no_action_no_check_satisfied_immediately():
    gate = Gate("g")
    assert gate.step() is True


def test_action_only_runs_once_and_satisfies_same_step():
    calls = []
    gate = Gate("g", action=lambda: calls.append(1))
    assert gate.step() is True
    assert calls == [1]
    # Still satisfied, action still only ran once.
    assert gate.step() is True
    assert calls == [1]


def test_check_only_waits_for_continuous_window(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("i2as.core.gates.time.time", lambda: now[0])

    gate = Gate("g", check=lambda: True, window_s=5.0)
    assert gate.step() is False  # stability clock just started

    now[0] = 3.0
    assert gate.step() is False  # not yet 5 s

    now[0] = 5.0
    assert gate.step() is True  # held for exactly the window


def test_check_only_resets_clock_on_false(monkeypatch):
    now = [0.0]
    stable = [True]
    monkeypatch.setattr("i2as.core.gates.time.time", lambda: now[0])

    gate = Gate("g", check=lambda: stable[0], window_s=5.0)
    gate.step()  # stable_since = 0.0
    now[0] = 4.0
    assert gate.step() is False  # 4 s elapsed, not yet 5

    stable[0] = False
    now[0] = 4.5
    assert gate.step() is False  # False resets the clock

    stable[0] = True
    now[0] = 4.5
    assert gate.step() is False  # just became stable again, clock restarts here

    now[0] = 9.5
    assert gate.step() is True  # 5.0 s continuous since it went stable at 4.5


def test_action_then_check_runs_action_before_clock_starts(monkeypatch):
    now = [0.0]
    calls = []
    monkeypatch.setattr("i2as.core.gates.time.time", lambda: now[0])

    gate = Gate("g", check=lambda: True, window_s=2.0, action=lambda: calls.append(1))
    assert gate.step() is False  # action ran, clock just started
    assert calls == [1]

    now[0] = 2.0
    assert gate.step() is True
    assert calls == [1]  # action never runs again
