from i2as.gui.lifecycle_toggle import LifecycleToggleButton


def test_starts_in_standby_state(qtbot):
    w = LifecycleToggleButton("magnet_z", lambda action: None)
    qtbot.addWidget(w)

    assert w.is_initiated() is False
    assert w._btn.text() == "Initiate"
    assert w._dot.property("status") == "standby"


def test_click_calls_back_with_initiate_and_does_not_flip_state(qtbot):
    calls = []
    w = LifecycleToggleButton("magnet_z", calls.append)
    qtbot.addWidget(w)

    w._btn.click()

    assert calls == ["initiate"]
    # No optimistic flip: state only changes via set_initiated().
    assert w.is_initiated() is False
    assert w._btn.text() == "Initiate"


def test_set_initiated_true_updates_button_and_dot(qtbot):
    w = LifecycleToggleButton("magnet_z", lambda action: None)
    qtbot.addWidget(w)

    w.set_initiated(True)

    assert w.is_initiated() is True
    assert w._btn.text() == "Standby"
    assert w._dot.property("status") == "initiated"


def test_click_after_initiated_calls_back_with_standby(qtbot):
    calls = []
    w = LifecycleToggleButton("magnet_z", calls.append)
    qtbot.addWidget(w)
    w.set_initiated(True)

    w._btn.click()

    assert calls == ["standby"]
    assert w.is_initiated() is True  # still waiting for confirmation


def test_set_initiated_same_state_skips_the_repaint(qtbot):
    """Re-asserting the current state must not trigger an unpolish/polish cycle.

    Asserting only the resulting text/state would pass even if the early return
    were deleted, so this counts _render() calls: the guard is the behaviour
    under test, and a redundant repaint on every tick is the regression it
    prevents.
    """
    w = LifecycleToggleButton("magnet_z", lambda action: None)
    qtbot.addWidget(w)

    renders = []
    original_render = w._render
    w._render = lambda: (renders.append(1), original_render())[1]

    w.set_initiated(False)  # already False → guard should short-circuit
    assert renders == [], "set_initiated(False) on a standby widget re-rendered"

    w.set_initiated(True)  # genuine change → must render
    assert len(renders) == 1
    assert w._btn.text() == "Standby"
