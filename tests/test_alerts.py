"""The alert model and band (i2as/gui/alerts.py).

The old banner held one message, so a new warning replaced the last and a
cleared fault could erase an unrelated error. These tests pin the rules that
replace it: every alert has a key naming its cause, conditions are resolved
only by their cause, events are dismissed by the operator or expire, repeats
count instead of stacking, and the band draws exactly what the model holds.
"""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QApplication

from i2as.gui.alerts import (
    EXPIRY_S,
    MAX_ALERTS,
    VISIBLE_ROWS,
    AlertBand,
    AlertCenter,
)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def center(qtbot, clock):
    return AlertCenter(clock=clock)


def test_one_cause_ending_never_removes_another(center):
    """The bug the banner had: clearing a fault erased a save failure."""
    center.raise_alert("store", "error", "session", "record NOT saved", dismissible=False)
    center.raise_alert("fault:magnet_z", "warning", "magnet_z", "comm lost", dismissible=False)

    center.resolve("fault:magnet_z")

    assert [a.key for a in center.alerts()] == ["store"]


def test_alerts_are_ordered_by_severity_then_newest(center, clock):
    center.raise_alert("w1", "warning", "a", "old warning")
    clock.now += 1
    center.raise_alert("e1", "error", "b", "an error")
    clock.now += 1
    center.raise_alert("w2", "warning", "c", "new warning")
    clock.now += 1
    center.raise_alert("em", "emergency", "station", "EMERGENCY", dismissible=False)

    assert [a.key for a in center.alerts()] == ["em", "e1", "w2", "w1"]
    assert center.counts() == {"warning": 2, "error": 1, "emergency": 1}


def test_a_repeat_counts_instead_of_stacking(center, clock):
    center.raise_alert("blocked:x", "warning", "station", "refused: busy")
    clock.now += 5
    center.raise_alert("blocked:x", "warning", "station", "refused: busy")

    [alert] = center.alerts()
    assert alert.count == 2
    assert (alert.first_seen, alert.last_seen) == (1000.0, 1005.0)

    center.raise_alert("blocked:x", "warning", "station", "a changed description")
    assert center.get("blocked:x").count == 2, "a new message is not a repeat"


def test_a_condition_cannot_be_dismissed_only_resolved(center):
    center.raise_alert("hold:coolant", "error", "station", "coolant low", dismissible=False)
    center.raise_alert("blocked:x", "warning", "station", "refused")

    assert center.dismiss("hold:coolant") is False
    assert center.dismiss_all() == 1
    assert center.keys() == ["hold:coolant"]
    assert center.resolve("hold:coolant") is True
    assert len(center) == 0


def test_a_group_is_synced_to_exactly_its_current_members(center, clock):
    center.raise_alert("store", "error", "session", "not saved", dismissible=False)
    spec = {"severity": "warning", "source": "magnet_z", "message": "comm lost", "dismissible": False}
    center.sync_group("fault:", {"fault:magnet_z": spec, "fault:temp": {**spec, "source": "temp"}})
    first_seen = center.get("fault:magnet_z").first_seen
    clock.now += 10

    changes = []
    center.changed.connect(lambda: changes.append(1))
    center.sync_group("fault:", {"fault:magnet_z": spec})

    assert sorted(center.keys()) == ["fault:magnet_z", "store"]
    assert center.get("fault:magnet_z").first_seen == first_seen, "unchanged is not re-raised"
    assert changes == [1], "one redraw for the whole group"
    center.sync_group("fault:", {})
    assert center.keys() == ["store"]
    with pytest.raises(ValueError):
        center.sync_group("fault:", {"hold:x": spec})


def test_expiring_alerts_go_after_they_stop_happening(center, clock):
    center.raise_alert("blocked:x", "warning", "station", "refused", expires=True)
    center.raise_alert("error:y", "error", "station", "failed")
    clock.now += EXPIRY_S - 1
    center.raise_alert("blocked:x", "warning", "station", "refused", expires=True)
    clock.now += EXPIRY_S - 1
    center.expire()
    assert "blocked:x" in center.keys(), "a repeat restarts its clock"
    clock.now += 2
    center.expire()
    assert center.keys() == ["error:y"], "non-expiring events stay until dismissed"


def test_the_oldest_dismissible_alerts_make_room_but_conditions_never_do(center, clock):
    center.raise_alert("hold:a", "error", "station", "held", dismissible=False)
    for index in range(MAX_ALERTS + 5):
        clock.now += 1
        center.raise_alert(f"blocked:{index}", "warning", "station", f"refused {index}")

    assert len(center) == MAX_ALERTS
    assert "hold:a" in center.keys()
    assert "blocked:0" not in center.keys()
    assert f"blocked:{MAX_ALERTS + 4}" in center.keys()


def test_an_unknown_severity_is_refused(center):
    with pytest.raises(ValueError):
        center.raise_alert("x", "catastrophe", "s", "m")


# ── The band ──────────────────────────────────────────────────────────────


@pytest.fixture
def band(qtbot, center):
    """The band inside a shown parent, as in the window: its own visibility is its own."""
    from PyQt6.QtWidgets import QVBoxLayout, QWidget

    parent = QWidget()
    widget = AlertBand(center)
    QVBoxLayout(parent).addWidget(widget)
    qtbot.addWidget(parent)
    parent.show()
    widget.test_parent = parent  # keep the parent (and so the band) alive
    return widget


def test_the_band_is_hidden_until_there_is_an_alert(band, center):
    assert not band.isVisible()
    center.raise_alert("x", "warning", "s", "m")
    assert band.isVisible()
    center.resolve("x")
    assert not band.isVisible()


def test_each_row_offers_its_own_action_and_dismissal(band, center):
    pressed = []
    center.raise_alert(
        "emergency",
        "emergency",
        "station",
        "EMERGENCY",
        action_label="Acknowledge emergency",
        action=lambda: pressed.append("ack"),
        action_name="ack_emergency_btn",
        dismissible=False,
    )
    center.raise_alert("blocked:x", "warning", "station", "refused")

    emergency = band.row_for("emergency")
    assert emergency.property("severity") == "emergency"
    assert emergency.action_button.objectName() == "ack_emergency_btn"
    assert emergency.dismiss_button is None, "a condition has no ✕"
    emergency.action_button.click()
    assert pressed == ["ack"]

    band.row_for("blocked:x").dismiss_button.click()
    assert center.keys() == ["emergency"]


def test_extra_rows_fold_behind_show_more(band, center, clock):
    for index in range(VISIBLE_ROWS + 2):
        clock.now += 1
        center.raise_alert(f"blocked:{index}", "warning", "s", f"refused {index}")

    assert len(band.rows) == VISIBLE_ROWS
    more = band.findChild(type(band.rows[0].dismiss_button), "alert_more_btn")
    assert more.text() == "Show 2 more"
    more.click()
    QApplication.processEvents()
    assert len(band.rows) == VISIBLE_ROWS + 2
    band.findChild(type(more), "alert_dismiss_all_btn").click()
    assert not band.isVisible()
