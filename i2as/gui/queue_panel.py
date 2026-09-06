"""QueuePanel — the run-queue list and its per-item lifecycle status.

The panel renders the queue; it does not own it. Waiting runs live in the
session layer as immutable **run specs** (`session/run_queue.py`), the engine
pulls the next one when it is ready, and this widget holds nothing the engine
owns — no built procedure, no reach into a private engine list. What it adds
on top of the queue is the one thing the queue has no opinion about: the
per-item lifecycle status an operator watches (pending -> running -> done or
failed), which outlives the spec, since a run that has started is no longer
waiting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import qtawesome as qta
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from i2as.core.events import QueueChanged
from i2as.core.orchestrator_proxy import OrchestratorProxy
from i2as.core.plan import ProbeSpec
from i2as.core.procedure import BaseProcedure
from i2as.gui.form_autosave import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    QueueItemState,
)
from i2as.gui.theme import BTN_CLASS_PRIMARY, TEXT_ON_ACCENT, TEXT_PRIMARY
from i2as.session.run_queue import (
    KIND_PROCEDURE,
    RunQueueHost,
    RunSpec,
    RunValidation,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    # The GUI holds a Station only as a type (contract C19).
    from i2as.core.station import Station

logger = logging.getLogger(__name__)

#: The reduction a "Probe first" click asks for. The **Probe run** standard's
#: own defaults (first/middle/last, no averaging, waits capped) — a probe the
#: operator did not have to design is a probe they will actually run.
DEFAULT_PROBE_SPEC = ProbeSpec()

#: The filename prefix a probe queued from here carries, so its file is
#: recognisable on disk before anything is opened. The run itself also
#: declares ``run_kind = "probe"``, which is the authoritative marker.
PROBE_FILE_PREFIX = "probe"


@dataclass
class QueueEntry:
    """One row of the queue list: the queued spec plus its lifecycle status.

    The spec is the queue's own immutable record of what was asked for; the
    status is this panel's, because it describes what happened to the run
    afterwards and outlives the spec's time in the queue — a running or
    finished entry has already been pulled out of it.

    Attributes:
        spec: The queued ``RunSpec``.
        status: One of ``pending``, ``running``, ``done``, ``failed``.
    """

    spec: RunSpec
    status: str = STATUS_PENDING


class QueuePanel(QGroupBox):
    """The Queue group box: list, per-item actions, and Run Queue.

    Per-item actions act on the selected waiting row: reorder, remove, and
    **Probe first**, which queues a cheap **probe run** of that row ahead of
    it — the same procedure and the same instruments, reduced to minutes, so
    an hours-long run is committed to only after something has answered
    "would this actually work". A probe goes through the same
    ``RunQueueHost`` add every other run does, so the setup's limits and the
    open experiment's envelope judge it identically, and what validation said
    about it — its findings and its **Duration estimate** — is shown inline
    and hung on the row as a tooltip.

    ObjectNames (``queue_list``, ``queue_up_btn``, ``queue_down_btn``,
    ``queue_remove_btn``, ``queue_probe_btn``, ``queue_probe_label``,
    ``run_queue_btn``) are preserved API.

    Args:
        station: The active Station instance (needed to validate and build a
            queued run headlessly).
        orchestrator: The client's ``OrchestratorProxy``.
        parent: Optional Qt parent widget.
        get_experiment_info: Callable returning the session layer's experiment
            context, stamped into a queued run when it is built. ``None``
            means no session layer is wired — runs get ``{}``.
        queue_host: The session layer's run queue (normally
            ``ExperimentManager.run_queue_host``). ``None`` means no session
            layer is wired: the panel builds a queue of its own from
            *procedure_classes* so the window still works standalone, and
            adopts the engine's pull seam only if nobody else has claimed it.
        procedure_classes: The discovered procedure classes, used to render a
            spec's summary line and to build the standalone queue's catalog.
    """

    def __init__(
        self,
        station: Station,
        orchestrator: OrchestratorProxy,
        parent: QWidget | None = None,
        get_experiment_info: Callable[[], dict[str, str]] | None = None,
        queue_host: RunQueueHost | None = None,
        procedure_classes: Sequence[type[BaseProcedure]] = (),
    ) -> None:
        super().__init__("Queue", parent)
        self._station = station
        self._orchestrator = orchestrator
        self._get_experiment_info = get_experiment_info
        self._classes: dict[str, type[BaseProcedure]] = {
            cls.__name__: cls for cls in procedure_classes
        }
        self._host = queue_host or self._standalone_host()

        # The rendered rows: every pending spec the queue holds, plus the
        # running/finished ones it no longer does.
        self._queue: list[QueueEntry] = []
        # The entry a reorder wants to stay selected. Held across the
        # rebuild rather than applied once, because the QueueChanged that
        # rebuilds the list can arrive after the reorder call has
        # returned — which is what happens with the engine on the
        # instrument thread.
        self._keep_selected: str | None = None
        # True while a queued run is executing, so notify_finished advances
        # the queue's per-item status.
        self._queue_running = False
        # ``{spec_id: what validation said about it}`` — the findings and the
        # duration estimate a "Probe first" click produced, hung on the row
        # as its tooltip so the caveat stays attached to the run it is about
        # across every rebuild of the list.
        self._spec_notes: dict[str, str] = {}

        # The engine's one event stream, under whichever name this client
        # offers it. A client CONSUMES the stream rather than relaying it, so
        # the proxy renames ``event_emitted`` to ``event``; a window handed a
        # bare Orchestrator (the inline construction path the GUI suites take)
        # still finds the engine's own signal. The same deliberate rename
        # ``StatusMirror.of()`` accommodates for the mirror.
        stream = getattr(orchestrator, "event_emitted", None)
        if stream is None:
            stream = orchestrator.event
        stream.connect(self._on_event)

        vlay = QVBoxLayout(self)

        self._queue_list = QListWidget()
        self._queue_list.setObjectName("queue_list")
        vlay.addWidget(self._queue_list)

        btn_row = QHBoxLayout()
        up_btn = QPushButton()
        up_btn.setObjectName("queue_up_btn")
        up_btn.setIcon(qta.icon("fa5s.arrow-up", color=TEXT_PRIMARY))
        up_btn.setToolTip("Move the selected queue item up")
        up_btn.setMaximumWidth(40)
        up_btn.clicked.connect(self._queue_move_up)
        down_btn = QPushButton()
        down_btn.setObjectName("queue_down_btn")
        down_btn.setIcon(qta.icon("fa5s.arrow-down", color=TEXT_PRIMARY))
        down_btn.setToolTip("Move the selected queue item down")
        down_btn.setMaximumWidth(40)
        down_btn.clicked.connect(self._queue_move_down)
        remove_btn = QPushButton("Remove")
        remove_btn.setObjectName("queue_remove_btn")
        remove_btn.setIcon(qta.icon("fa5s.trash", color=TEXT_PRIMARY))
        remove_btn.setToolTip("Remove the selected item from the queue")
        remove_btn.clicked.connect(self._queue_remove)
        probe_btn = QPushButton("Probe first")
        probe_btn.setObjectName("queue_probe_btn")
        probe_btn.setIcon(qta.icon("fa5s.vial", color=TEXT_PRIMARY))
        probe_btn.setToolTip(
            "Queue a cheap probe of the selected run — same procedure, same "
            "instruments, reduced to minutes — ahead of it, so you find out "
            "whether it works before committing the hours."
        )
        probe_btn.clicked.connect(self._queue_probe_first)
        run_queue_btn = QPushButton("Run Queue")
        run_queue_btn.setObjectName("run_queue_btn")
        run_queue_btn.setProperty("class", BTN_CLASS_PRIMARY)
        run_queue_btn.setIcon(qta.icon("fa5s.forward", color=TEXT_ON_ACCENT))
        run_queue_btn.setToolTip("Run all queued procedures in order")
        run_queue_btn.clicked.connect(self._on_run_queue)
        # Two rows, not one: the per-item actions (which act on the selected
        # waiting run) on the first, and Run Queue (which starts the whole
        # queue) on the second. Keeping all five on one row pushed this group
        # box's minimum width past what the window's 50/50 split can give it
        # at 1280 px, which took the width back out of the parameter form and
        # put a horizontal scrollbar under it.
        btn_row.addWidget(up_btn)
        btn_row.addWidget(down_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addWidget(probe_btn)
        btn_row.addStretch()
        vlay.addLayout(btn_row)

        # What the last probe was told about itself: the validation's
        # findings and its **Duration estimate**, shown inline rather than in
        # a modal, because a caveat is something to read beside the queue
        # rather than dismiss to get back to it.
        self._probe_label = QLabel("")
        self._probe_label.setObjectName("queue_probe_label")
        self._probe_label.setProperty("class", "secondary_label")
        self._probe_label.setWordWrap(True)
        self._probe_label.setVisible(False)
        vlay.addWidget(self._probe_label)

        run_row = QHBoxLayout()
        run_row.addStretch()
        run_row.addWidget(run_queue_btn)
        vlay.addLayout(run_row)

        self._sync_from_queue()

    def _standalone_host(self) -> RunQueueHost:
        """Build this window's own run queue, for a window with no session layer.

        Unit tests (and any launch without an ``ExperimentManager``) still get
        a working queue: the setup's own ``control_limits`` guard every add
        exactly as they do in production — there is simply no open experiment,
        so no envelope narrows them. The engine's pull seam is adopted only if
        nothing has claimed it, so this can never displace the wiring
        ``main.py`` installed.

        Returns:
            A ``RunQueueHost`` over this window's discovered procedures.
        """
        host = RunQueueHost(
            station=self._station,
            run_catalog=dict(self._classes),
            publish=lambda actor: self._orchestrator.publish_queue(actor=actor),
            experiment_info=self._get_experiment_info,
        )
        if self._orchestrator.next_procedure is None:
            self._orchestrator.next_procedure = host.next_run
            self._orchestrator.queue_snapshot = host.entries
        return host

    # ------------------------------------------------------------------
    # Queue mutation
    # ------------------------------------------------------------------

    def add_run(
        self,
        procedure_cls: type[BaseProcedure],
        params: dict[str, Any],
        sample_info: dict[str, str],
        data_directory: str,
        file_prefix: str = "",
    ) -> RunSpec | None:
        """Queue one run, refusing it with its findings if validation fails.

        Validation happens here, at add time: the run is built headlessly and
        thrown away, so a parameter outside a setup limit or the open
        experiment's envelope is refused while the operator is still looking
        at the form — not an hour later when the run would have started. The
        refusal is a modal because it answers a direct click, exactly like the
        Run-now path's build refusal.

        Args:
            procedure_cls: The procedure class to queue.
            params: The parameter values collected from the form.
            sample_info: Sample metadata captured with the item.
            data_directory: Data directory captured with the item.
            file_prefix: Optional filename prefix.

        Returns:
            The queued ``RunSpec``, or ``None`` when it was refused.
        """
        spec, validation = self._host.add(
            procedure_cls,
            params,
            sample_info=sample_info,
            data_directory=data_directory,
            file_prefix=file_prefix,
        )
        if spec is None:
            logger.warning(
                "Refused to queue %s: %s",
                procedure_cls.__name__,
                "; ".join(validation.messages()),
            )
            QMessageBox.warning(
                self, "Cannot Queue Run", "\n".join(validation.messages())
            )
            return None
        return spec

    def is_running(self) -> bool:
        """Return True while a queued run is executing."""
        return self._queue_running

    def _on_event(self, event: object) -> None:
        """Re-render the queue whenever the engine says it changed.

        The single subscription that makes this panel a VIEW: a run queued by
        an agent, popped by the engine, or reordered elsewhere shows up here
        for the same reason the operator's own click does.

        Args:
            event: One event from the engine's stream; anything but a
                ``QueueChanged`` is ignored.
        """
        if isinstance(event, QueueChanged):
            self._sync_from_queue()

    def _sync_from_queue(self) -> None:
        """Re-derive the pending rows from the queue, keeping the finished ones.

        The queue is the source of truth for what is still waiting and in what
        order; this panel is the source of truth for what already happened.
        Rows that are no longer pending stay where they are, in the order they
        ran, and the pending rows below them are rebuilt from the snapshot so
        an entry keeps its ``QueueEntry`` — and therefore its status — across a
        reorder.
        """
        by_id = {entry.spec.spec_id: entry for entry in self._queue}
        history = [entry for entry in self._queue if entry.status != STATUS_PENDING]
        pending = [
            by_id.get(spec.spec_id) or QueueEntry(spec=spec)
            for spec in self._host.snapshot()
        ]
        self._queue = history + pending
        self._refresh_queue_list()
        if self._keep_selected is not None:
            self._select_spec(self._keep_selected)
            self._keep_selected = None

    def _on_run_queue(self) -> None:
        """Start the queue run, marking the first pending item as running.

        Wraps ``Orchestrator.run_queue`` so the queue's per-item status
        reflects execution: the first pending entry becomes ``running`` and,
        from here on, ``notify_finished`` advances the status (see
        ``_advance_queue_status``). The engine still decides *when* the run
        starts — this asks, it never starts anything itself.
        """
        first_pending = next(
            (e for e in self._queue if e.status == STATUS_PENDING), None
        )
        if first_pending is None:
            return
        self._queue_running = True
        first_pending.status = STATUS_RUNNING
        self._refresh_queue_list()
        self._orchestrator.run_queue()

    def notify_finished(self) -> None:
        """Advance the queue after a clean procedure finish (if a run is active).

        The Orchestrator auto-chains ``run_queue()`` right after emitting
        ``procedure_finished``, so this only updates per-item status.
        """
        if self._queue_running:
            self._advance_queue_status(STATUS_DONE)

    def notify_aborted(self) -> None:
        """Record the aborted queue item as failed (if a run is active).

        ``abort_procedure`` does not emit ``procedure_finished``; it goes IDLE
        and auto-runs the next item, so the running entry is finalised here.
        """
        if self._queue_running:
            self._advance_queue_status(STATUS_FAILED)

    def _advance_queue_status(self, final_status: str) -> None:
        """Finalise the running entry and promote the next pending one."""
        for entry in self._queue:
            if entry.status == STATUS_RUNNING:
                entry.status = final_status
                break
        next_pending = next(
            (e for e in self._queue if e.status == STATUS_PENDING), None
        )
        if next_pending is not None:
            next_pending.status = STATUS_RUNNING
        else:
            self._queue_running = False
        self._refresh_queue_list()

    def _selected_pending(self) -> QueueEntry | None:
        """Return the selected row when it is still waiting, else ``None``.

        A row that is running or finished has already left the queue, so there
        is nothing to remove or reorder.
        """
        row = self._queue_list.currentRow()
        if row < 0 or row >= len(self._queue):
            return None
        entry = self._queue[row]
        return entry if entry.status == STATUS_PENDING else None

    def _queue_move_up(self) -> None:
        """Move the selected queue item up by one position."""
        self._move_selected(-1)

    def _queue_move_down(self) -> None:
        """Move the selected queue item down by one position."""
        self._move_selected(1)

    def _move_selected(self, offset: int) -> None:
        """Move the selected pending entry and keep the selection on it.

        The selection is restored twice on purpose. Once here, for a client
        whose ``QueueChanged`` came back inside ``move()`` and has already
        rebuilt the list; and once from ``_sync_from_queue()`` when it comes
        back later instead — which is what happens with the engine on the
        instrument thread, where the rebuild lands after this method has
        returned and would otherwise clear the selection the operator was
        working with.

        Args:
            offset: ``-1`` to move up, ``+1`` to move down.
        """
        entry = self._selected_pending()
        if entry is None:
            return
        self._keep_selected = entry.spec.spec_id
        if self._host.move(entry.spec.spec_id, offset):
            self._select_spec(entry.spec.spec_id)
        else:
            self._keep_selected = None

    def _queue_probe_first(self) -> None:
        """Queue a **probe run** of the selected item, ahead of the item.

        The whole point of a probe is that it comes FIRST: the same procedure
        class driving the same instruments through the same code path,
        reduced until it costs minutes, so an hours-long run is committed to
        only once something has already answered "would this actually work".
        So the probe is validated and queued through the ``RunQueueHost``
        exactly as any other run — the setup's limits and the open
        experiment's envelope judge it identically — and then moved to sit
        immediately before the run it probes.

        A refusal is a modal, like every other refusal that answers a direct
        click. What validation said about an accepted probe — its caveats and
        its **Duration estimate** — is shown inline and hung on the row as a
        tooltip instead, because it is something to read beside the queue
        rather than dismiss to get back to it.
        """
        entry = self._selected_pending()
        if entry is None:
            return
        spec = entry.spec
        run_class = self._classes.get(spec.run_class)
        if run_class is None:
            QMessageBox.warning(
                self,
                "Cannot Probe This Item",
                f"{spec.run_class} is not among the procedures this window "
                "discovered, so there is nothing to build a probe from.",
            )
            return
        probe, validation = self._host.add(
            run_class,
            spec.params,
            kind=spec.kind,
            sample_info=spec.sample_info,
            data_directory=spec.data_directory,
            file_prefix=spec.file_prefix or PROBE_FILE_PREFIX,
            probe_spec=DEFAULT_PROBE_SPEC.to_json(),
        )
        if probe is None:
            logger.warning(
                "Refused to queue a probe of %s: %s",
                spec.run_class,
                "; ".join(validation.messages()),
            )
            QMessageBox.warning(
                self, "Cannot Queue Probe", "\n".join(validation.messages())
            )
            return
        self._move_before(probe.spec_id, spec.spec_id)
        self._spec_notes[probe.spec_id] = self._probe_note(validation)
        self._keep_selected = probe.spec_id
        self._sync_from_queue()
        self._show_probe_note(self._spec_notes[probe.spec_id])

    def _move_before(self, spec_id: str, target_id: str) -> None:
        """Move one waiting run to sit immediately before another.

        The queue's own ``move()`` takes an offset within a bucket, which is
        what a reorder button needs; this is the position-based move a probe
        needs, expressed in terms of it.

        Args:
            spec_id: The entry to move (a probe, freshly appended).
            target_id: The entry it must end up in front of.
        """
        order = [waiting.spec_id for waiting in self._host.snapshot()]
        if spec_id not in order or target_id not in order:
            return
        offset = order.index(target_id) - order.index(spec_id)
        if offset:
            self._host.move(spec_id, offset)

    @staticmethod
    def _probe_note(validation: RunValidation) -> str:
        """Render what validation said about a probe, for the row's tooltip.

        Args:
            validation: The ``RunValidation`` the host answered with.

        Returns:
            One line naming the estimate (with the first of its assumptions,
            since an estimate a client cannot qualify is worse than none) and
            every finding, or a plain statement that there was nothing to
            report.
        """
        parts: list[str] = []
        seconds = validation.duration_estimate_s
        if seconds is not None:
            parts.append(f"probe ≈ {seconds / 60:.1f} min")
            assumptions = validation.estimate.assumptions if validation.estimate else ()
            if assumptions:
                parts.append(f"assuming {assumptions[0]}")
        parts.extend(validation.messages())
        return " · ".join(parts) if parts else "probe queued — nothing to report"

    def _show_probe_note(self, note: str) -> None:
        """Show one probe's findings inline under the queue.

        Args:
            note: The line ``_probe_note()`` produced.
        """
        self._probe_label.setText(note)
        self._probe_label.setVisible(bool(note))

    def _queue_remove(self) -> None:
        """Remove the selected item from the queue."""
        entry = self._selected_pending()
        if entry is None:
            return
        self._host.remove(entry.spec.spec_id)

    def _select_spec(self, spec_id: str) -> None:
        """Keep the selection on one entry after the list was rebuilt."""
        for row, entry in enumerate(self._queue):
            if entry.spec.spec_id == spec_id:
                self._queue_list.setCurrentRow(row)
                return

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _entry_summary(self, entry: QueueEntry) -> str:
        """Return the one-line queue summary for a queue entry (prefix-aware)."""
        spec = entry.spec
        cls = self._classes.get(spec.run_class)
        name = getattr(cls, "name", "") or spec.run_class
        label = f"[{spec.file_prefix}] {name}" if spec.file_prefix else name
        if spec.probe_spec and PROBE_FILE_PREFIX not in spec.file_prefix:
            # A probe is never science data, so it says so in the queue too —
            # once: a spec whose own file prefix already says "probe" is not
            # made clearer by saying it twice.
            label = f"{label} (probe)"
        if cls is None:
            return label
        summary_parts = self._queue_summary_parts(cls, spec.params)
        return f"{label} ({', '.join(summary_parts)})"

    def _refresh_queue_list(self) -> None:
        """Rebuild the QListWidget from self._queue, annotating non-pending status."""
        self._queue_list.clear()
        for idx, entry in enumerate(self._queue):
            label = f"{idx + 1}. {self._entry_summary(entry)}"
            if entry.status != STATUS_PENDING:
                label = f"{label}  — {entry.status}"
            item = QListWidgetItem(label)
            note = self._spec_notes.get(entry.spec.spec_id)
            if note:
                item.setToolTip(note)
            self._queue_list.addItem(item)

    @staticmethod
    def _queue_summary_parts(cls: type[BaseProcedure], params: dict) -> list[str]:
        """Build a short "key=value" summary of a procedure's sweep for the queue list.

        A sweep_axis-declaring procedure gets a mode-aware one-liner (e.g.
        ``field=-1.0->1.0`` or ``field=segments(3)``) instead of dumping the
        raw hidden parameter names.

        Args:
            cls: The procedure class.
            params: Its collected parameter values.

        Returns:
            A list of up to 3 "key=value" strings.
        """
        if cls.sweep_axis is not None:
            k = cls.sweep_axis.key
            mode = params.get(f"{k}_mode", "linear")
            if mode == "segments":
                n = len(params.get(f"{k}_segments", []))
                return [f"{k}=segments({n})"]
            if mode == "csv":
                return [f"{k}=csv"]
            return [f"{k}={params[f'{k}_start']}->{params[f'{k}_end']}"]

        sweep_keys = list(cls.sweep_parameters.keys()) or list(cls.parameters.keys())
        return [f"{k}={params[k]}" for k in sweep_keys[:3]]

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def restore_items(
        self,
        items: list[QueueItemState],
        procedure_lookup: Callable[[str], type[BaseProcedure] | None],
    ) -> None:
        """Rebuild the queue from persisted items, re-queueing the pending ones.

        A restored spec is put back exactly as it was saved rather than
        re-validated: the operator gets the queue they left behind, and a
        stored value that has since gone out of bounds is reported when the
        engine pulls the run, not silently dropped while reopening a session.

        Args:
            items: The persisted queue items.
            procedure_lookup: Maps a saved procedure name to its discovered
                class (``None`` for an unknown name, which is skipped).
        """
        self._host.clear()
        self._queue.clear()
        history: list[QueueEntry] = []
        for item in items:
            cls = procedure_lookup(item.procedure)
            if cls is None:
                logger.warning(
                    "session: unknown procedure %r in saved queue; skipping",
                    item.procedure,
                )
                continue
            # A "running" item never finished (app closed mid-run) — treat as pending.
            status = (
                STATUS_PENDING
                if item.status in (STATUS_PENDING, STATUS_RUNNING)
                else item.status
            )
            spec = RunSpec(
                kind=KIND_PROCEDURE,
                run_class=cls.__name__,
                params=dict(item.params),
                sample_info=dict(item.sample_info),
                data_directory=item.data_dir,
                file_prefix=item.file_prefix,
            )
            if status == STATUS_PENDING:
                self._host.add_spec(spec)
            else:
                history.append(QueueEntry(spec=spec, status=status))
        self._queue = history
        self._sync_from_queue()

    def export_items(self) -> list[QueueItemState]:
        """Return the queue as persistable QueueItemStates."""
        return [
            QueueItemState(
                procedure=self._saved_name(entry.spec.run_class),
                params=dict(entry.spec.params),
                sample_info=dict(entry.spec.sample_info),
                data_dir=entry.spec.data_directory,
                file_prefix=entry.spec.file_prefix,
                status=entry.status,
            )
            for entry in self._queue
        ]

    def _saved_name(self, run_class: str) -> str:
        """Return the display name a queue item is persisted under.

        Saved state names a procedure the way the selector does, so the file
        format is unchanged by the queue moving to class names internally.
        """
        cls = self._classes.get(run_class)
        return getattr(cls, "name", "") or run_class

    def reset(self) -> None:
        """Clear the queue (rows and specs) and stop status tracking."""
        self._host.clear()
        self._queue.clear()
        self._spec_notes.clear()
        self._queue_running = False
        self._probe_label.setVisible(False)
        self._refresh_queue_list()
