"""The **eLab tab** — analyse a finished run, then approve what it wrote.

The procedure window's top-right quadrant carries two tabs: "Queue", the run
queue over the status log, and "eLab", this panel. It is the human half of
the analysis track: a recipe turns one finished run into an analysed entry,
the entry is parked on that run as a **pending entry**, and nothing reaches
the notebook until the person reading it here presses Publish.

Three rules shape it:

- **Nothing is published without approval.** Publish calls
  ``ExperimentManager.approve_eln_draft()`` and Discard
  ``discard_pending_eln_draft()`` — the manager is the single writer of
  experiment state, exactly as the Orchestrator is the single writer to
  hardware. This panel writes no record and sends nothing itself.
- **The preview is for the operator's eyes only.** The figures a recipe saved
  are shown here from their local files, above the body; the body that
  actually reaches the notebook never embeds an image (the entry's figures
  travel as attachments). Editing the preview changes nothing.
- **Every collaborator is optional.** With no session layer, no publisher and
  no runner — a unit test, or a launch without the session tier — the panel
  builds, says so in one line, and offers no action it cannot perform.

The recipe catalogue is read through ``i2as.analysis.discovery``, imported
lazily: a build without the analysis package degrades to an empty recipe list
and a status line saying so, rather than a window that will not open.
"""

from __future__ import annotations

import html
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from i2as.analysis.report import REPORT_FILENAME, AnalysisReport
from i2as.gui.eln_settings_dialog import persist_eln_settings
from i2as.gui.theme import BTN_CLASS_PRIMARY, BTN_CLASS_SECONDARY

logger = logging.getLogger(__name__)

#: Status line: no session layer is wired at all (unit tests, and any launch
#: without the experiment tier). Nothing on this panel can do anything.
NO_SESSION_TEXT = "Session layer not wired"

#: Status line: a session layer is wired, but no experiment is open.
NO_EXPERIMENT_TEXT = "No experiment is open"

#: Status line: the open experiment has no finished run to analyse yet.
NO_RUNS_TEXT = "No finished runs in this experiment yet"

#: Status line: a recipe is running for the selected run.
ANALYSING_TEXT = "Analysing…"

#: Status line: an analysed entry is parked on the selected run.
READY_TEXT = "Entry ready for review"

#: Status line: the selected run has no entry waiting.
NOTHING_PENDING_TEXT = "Nothing pending for this run"

#: Status line: this build carries no analysis recipes to choose from.
NO_RECIPES_TEXT = "Analysis recipes are not available in this build"

#: Suffix marking a recipe that lives in the open experiment's own folder
#: rather than in the package — the reader must be able to tell which code
#: produced an entry.
EXPERIMENT_SUFFIX = " (experiment)"

#: Any-procedure marker of the recipe contract (``analysis/report.py``'s
#: ``ANY_PROCEDURE``), repeated here so the filter needs no analysis import.
_ANY_PROCEDURE = "*"

#: Room left beside a preview figure for the browser's own frame and
#: scrollbar, so clamping a figure to the viewport does not itself push one.
_PREVIEW_MARGIN_PX = 28

#: Below this the preview has no meaningful width yet (it is not on screen),
#: and a recipe's declared figure width is used unclamped.
_MIN_CLAMP_WIDTH_PX = 200

#: Publish states the chip renders, from ``publish_state_changed``.
_CHIP_TEXT = {
    "synced": "eLab · synced",
    "pending": "eLab · pending",
    "offline": "eLab · offline",
    "disabled": "eLab · off",
}


def _run_is_finished(run: Any) -> bool:
    """Return whether one run record is over (whatever its outcome).

    Args:
        run: A ``RunRecord``.

    Returns:
        ``True`` unless the run is still running.
    """
    return str(getattr(run, "status", "")) != "running"


class AnalysisPanel(QWidget):
    """The **eLab tab**: analyse a finished run and approve the entry.

    Named widgets (``findChild`` objectNames are API): the panel itself
    ``analysis_panel``, the publish-state chip ``analysis_publish_chip``, the
    analysis toggle ``analysis_enabled_checkbox``, the setup button
    ``eln_setup_btn``, the run selector ``analysis_run_combo``, the recipe
    selector ``analysis_recipe_combo``, ``analysis_new_recipe_btn``,
    ``analysis_run_btn``, the status line ``analysis_status_label``, the
    preview ``analysis_preview``, the warnings box ``analysis_warnings``, and
    ``analysis_publish_btn`` / ``analysis_discard_btn``.

    Args:
        session_manager: The L6 ``ExperimentManager``. Used for the open
            experiment's runs, its store paths, the **pending entry** on a
            run, and the two approval calls. ``None`` leaves the panel in its
            not-wired state.
        eln_publisher: The ``ElnPublisher``, for the publish-state chip and
            the analysis on/off setting. ``None`` hides neither control but
            leaves both inert.
        analysis_runner: The ``AnalysisRunner``, for "Run analysis" and the
            three progress signals. ``None`` disables the button.
        open_settings: Called when "eLab setup…" is pressed; whoever built
            the panel owns the dialog. ``None`` disables the button.
        parent: Optional Qt parent widget.
    """

    def __init__(
        self,
        *,
        session_manager: Any | None = None,
        eln_publisher: Any | None = None,
        analysis_runner: Any | None = None,
        open_settings: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("analysis_panel")
        self._manager = session_manager
        self._publisher = eln_publisher
        self._runner = analysis_runner
        self._open_settings = open_settings
        #: Recipes offered for the selected run, in combo order.
        self._recipes: tuple[Any, ...] = ()
        #: Whether ``i2as.analysis.discovery`` could be imported at all.
        self._recipes_available = True
        #: ``{run_id: failure text}`` from the runner, so a failure stays on
        #: screen until that run is analysed again.
        self._failures: dict[str, str] = {}
        #: Guard against the settings write the checkbox itself triggers
        #: being re-applied while the panel is refreshing the checkbox.
        self._loading = False
        #: The pending entry currently on screen, with the report and figure
        #: directory it was rendered from — re-used by the resize path.
        self._entry: dict[str, Any] = {}
        self._report_shown: AnalysisReport | None = None
        self._report_dir: Path | None = None

        self._build_ui()
        self._connect_collaborators()
        self.reload()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the panel: header, run row, status, preview, approval row."""
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
        root.addLayout(self._build_header_row())
        root.addLayout(self._build_run_row())

        self._status_label = QLabel(NO_SESSION_TEXT)
        self._status_label.setObjectName("analysis_status_label")
        self._status_label.setProperty("class", "secondary_label")
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        self._preview = QTextBrowser()
        self._preview.setObjectName("analysis_preview")
        self._preview.setOpenExternalLinks(True)
        self._preview.setMinimumHeight(160)
        root.addWidget(self._preview, stretch=1)

        self._warnings = QTextEdit()
        self._warnings.setObjectName("analysis_warnings")
        self._warnings.setReadOnly(True)
        self._warnings.setMaximumHeight(64)
        self._warnings.setToolTip("What the recipe could not do, in its own words")
        self._warnings.hide()
        root.addWidget(self._warnings)

        root.addLayout(self._build_approval_row())

    def _build_header_row(self) -> QHBoxLayout:
        """Build the chip / analysis toggle / setup-button header.

        Returns:
            The header row's layout.
        """
        row = QHBoxLayout()
        self._chip = QLabel(_CHIP_TEXT["disabled"])
        self._chip.setObjectName("analysis_publish_chip")
        self._chip.setProperty("class", "publish_chip")
        self._chip.setProperty("state", "disabled")
        self._chip.setToolTip("Whether everything queued has reached the notebook")
        row.addWidget(self._chip)

        self._enabled_checkbox = QCheckBox("Analysis on")
        self._enabled_checkbox.setObjectName("analysis_enabled_checkbox")
        self._enabled_checkbox.setToolTip(
            "Analyse a finished run before its entry is written. The entry "
            "still waits here for your approval."
        )
        self._enabled_checkbox.toggled.connect(self._on_analysis_toggled)
        row.addWidget(self._enabled_checkbox)

        row.addStretch()

        self._setup_btn = QPushButton("eLab setup…")
        self._setup_btn.setObjectName("eln_setup_btn")
        self._setup_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._setup_btn.setToolTip("Notebook address, credentials and analysis options")
        self._setup_btn.setEnabled(self._open_settings is not None)
        self._setup_btn.clicked.connect(self._on_setup_clicked)
        row.addWidget(self._setup_btn)
        return row

    def _build_run_row(self) -> QGridLayout:
        """Build the run selector, recipe selector and the two recipe buttons.

        Two rows rather than one: this quadrant shares its width with the
        parameter form, and a single row of two combos plus two buttons sets
        a minimum width that would squeeze the form into a horizontal
        scrollbar.

        Returns:
            The run rows' layout.
        """
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setColumnStretch(1, 1)

        grid.addWidget(QLabel("Run:"), 0, 0)
        self._run_combo = QComboBox()
        self._run_combo.setObjectName("analysis_run_combo")
        self._run_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._run_combo.setToolTip("A finished run of the open experiment")
        self._run_combo.currentIndexChanged.connect(self._on_run_selected)
        grid.addWidget(self._run_combo, 0, 1)

        self._run_btn = QPushButton("Run analysis")
        self._run_btn.setObjectName("analysis_run_btn")
        self._run_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._run_btn.setToolTip("Analyse the selected run with the selected recipe")
        self._run_btn.clicked.connect(self._on_run_analysis_clicked)
        grid.addWidget(self._run_btn, 0, 2)

        grid.addWidget(QLabel("Recipe:"), 1, 0)
        self._recipe_combo = QComboBox()
        self._recipe_combo.setObjectName("analysis_recipe_combo")
        self._recipe_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._recipe_combo.setToolTip(
            "Which analysis recipe runs. Recipes marked (experiment) live in "
            "this experiment's own folder."
        )
        grid.addWidget(self._recipe_combo, 1, 1)

        self._new_recipe_btn = QPushButton("New recipe…")
        self._new_recipe_btn.setObjectName("analysis_new_recipe_btn")
        self._new_recipe_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._new_recipe_btn.setToolTip(
            "Write a starting recipe into this experiment's analysis folder "
            "and open it"
        )
        self._new_recipe_btn.clicked.connect(self._on_new_recipe_clicked)
        grid.addWidget(self._new_recipe_btn, 1, 2)
        return grid

    def _build_approval_row(self) -> QHBoxLayout:
        """Build the Publish / Discard row.

        Returns:
            The approval row's layout.
        """
        row = QHBoxLayout()
        row.addStretch()
        self._publish_btn = QPushButton("Publish")
        self._publish_btn.setObjectName("analysis_publish_btn")
        self._publish_btn.setProperty("class", BTN_CLASS_PRIMARY)
        self._publish_btn.setToolTip(
            "Queue this entry for the notebook. Nothing is sent until you do."
        )
        self._publish_btn.clicked.connect(self._on_publish_clicked)
        row.addWidget(self._publish_btn)

        self._discard_btn = QPushButton("Discard")
        self._discard_btn.setObjectName("analysis_discard_btn")
        self._discard_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._discard_btn.setToolTip("Drop this entry; the run stays unpublished")
        self._discard_btn.clicked.connect(self._on_discard_clicked)
        row.addWidget(self._discard_btn)
        return row

    def _connect_collaborators(self) -> None:
        """Connect the manager, publisher and runner signals, when present."""
        self._connect(self._manager, "experiment_changed", self._on_experiment_changed)
        self._connect(self._manager, "run_recorded", self._on_run_recorded)
        self._connect(self._publisher, "publish_state_changed", self.on_publish_state)
        self._connect(self._runner, "analysis_started", self.on_analysis_started)
        self._connect(self._runner, "analysis_finished", self.on_analysis_finished)
        self._connect(self._runner, "analysis_failed", self.on_analysis_failed)

    @staticmethod
    def _connect(source: Any | None, name: str, slot: Callable[..., None]) -> None:
        """Connect one optional collaborator's signal, ignoring what is absent.

        Args:
            source: The collaborator, or ``None``.
            name: The signal's attribute name.
            slot: The slot to connect it to.
        """
        signal = getattr(source, name, None)
        connect = getattr(signal, "connect", None)
        if callable(connect):
            connect(slot)

    # ------------------------------------------------------------------
    # The open experiment
    # ------------------------------------------------------------------

    def _experiment(self) -> Any | None:
        """Return the open ``ExperimentRecord``, or ``None``.

        Returns:
            The record, or ``None`` when no manager is wired, none is open,
            or the manager refused the read (logged, never raised).
        """
        if self._manager is None:
            return None
        try:
            return self._manager.current_experiment()
        except Exception:  # noqa: BLE001 - a view never raises into Qt
            logger.exception("eLab tab: could not read the open experiment")
            return None

    def current_run_id(self) -> str:
        """Return the run the panel is showing, or ``""`` when none is selected."""
        return str(self._run_combo.currentData() or "")

    def selected_recipe(self) -> str:
        """Return the selected recipe's name, or ``""`` for "let the runner pick"."""
        return str(self._recipe_combo.currentData() or "")

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Re-read runs, recipes and the pending entry, and repaint.

        The one refresh path: every signal this panel listens to, and every
        action it performs, ends here, so the panel always shows what the
        manager currently holds rather than what it last drew.
        """
        self._refresh_chip()
        self._refresh_analysis_toggle()
        self._reload_runs()
        self._reload_recipes()
        self._refresh_pending()

    def set_run(self, run_id: str) -> None:
        """Select one run and refresh everything that follows from it.

        Args:
            run_id: The run to show. Unknown ids leave the selection alone.
        """
        index = self._run_combo.findData(str(run_id))
        if index < 0:
            self._reload_runs()
            index = self._run_combo.findData(str(run_id))
        if index >= 0:
            self._run_combo.setCurrentIndex(index)
        self._reload_recipes()
        self._refresh_pending()

    def on_run_finished(self, manifest: Mapping[str, Any] | None) -> None:
        """Select the run that just finished (the Orchestrator's run boundary).

        Args:
            manifest: The run manifest the engine emitted; anything without a
                ``run_id`` only refreshes the panel.
        """
        self.reload()
        run_id = str((manifest or {}).get("run_id", ""))
        if run_id:
            self.set_run(run_id)

    def _reload_runs(self) -> None:
        """Repopulate the run combo with the open experiment's finished runs."""
        selected = self.current_run_id()
        record = self._experiment()
        runs = [run for run in getattr(record, "runs", ()) if _run_is_finished(run)]
        runs.reverse()  # records are stored oldest first; newest belongs on top

        self._run_combo.blockSignals(True)
        self._run_combo.clear()
        for run in runs:
            run_id = str(getattr(run, "run_id", ""))
            label = (
                f"{run_id} · {getattr(run, 'procedure', '')} · "
                f"{getattr(run, 'status', '')}"
            )
            self._run_combo.addItem(label, run_id)
        index = self._run_combo.findData(selected)
        if index >= 0:
            self._run_combo.setCurrentIndex(index)
        self._run_combo.blockSignals(False)

    def _run_record(self, run_id: str = "") -> Any | None:
        """Return one run record of the open experiment.

        Args:
            run_id: The run to find; ``""`` uses the current selection.

        Returns:
            The ``RunRecord``, or ``None`` when it is not in the open
            experiment.
        """
        wanted = run_id or self.current_run_id()
        record = self._experiment()
        for run in getattr(record, "runs", ()):
            if str(getattr(run, "run_id", "")) == wanted:
                return run
        return None

    def _procedure_of(self, run_id: str = "") -> str:
        """Return the procedure class name a run executed, or ``""``."""
        return str(getattr(self._run_record(run_id), "procedure", "") or "")

    # ------------------------------------------------------------------
    # Recipes
    # ------------------------------------------------------------------

    def _store_dir(self, method: str) -> Path | None:
        """Return one of the store's analysis paths for the open experiment.

        Args:
            method: ``"recipes_dir"`` or ``"report_dir"``.

        Returns:
            The path, or ``None`` when there is no experiment, no store, or
            this build's store does not offer that path yet.
        """
        record = self._experiment()
        store = getattr(self._manager, "store", None)
        resolve = getattr(store, method, None)
        if record is None or not callable(resolve):
            return None
        try:
            if method == "report_dir":
                return Path(resolve(record.experiment_id, self.current_run_id()))
            return Path(resolve(record.experiment_id))
        except Exception:  # noqa: BLE001 - a view never raises into Qt
            logger.exception("eLab tab: could not resolve the %s path", method)
            return None

    def _discover_recipes(self) -> tuple[Any, ...]:
        """Return every recipe available for the open experiment.

        Returns:
            The ``RecipeInfo`` records, package ones first; empty when the
            analysis package is unavailable (recorded in
            ``_recipes_available``, which the status line reports).
        """
        try:
            from i2as.analysis.discovery import discover_recipes
        except ImportError:
            self._recipes_available = False
            logger.warning("No analysis package in this build — no recipes to offer")
            return ()
        self._recipes_available = True
        recipes_dir = self._store_dir("recipes_dir")
        extra = [recipes_dir] if recipes_dir is not None else []
        try:
            return tuple(discover_recipes(extra))
        except Exception:  # noqa: BLE001 - discovery never breaks the panel
            logger.exception("eLab tab: recipe discovery failed")
            return ()

    def _preferred_recipe(self, procedure: str) -> str:
        """Return the recipe name the settings pin to one procedure, or ``""``."""
        analysis = getattr(getattr(self._publisher, "settings", None), "analysis", None)
        recipes = getattr(analysis, "recipes", None)
        if isinstance(recipes, Mapping):
            return str(recipes.get(procedure, "") or "")
        return ""

    def _reload_recipes(self) -> None:
        """Repopulate the recipe combo for the selected run's procedure."""
        procedure = self._procedure_of()
        self._recipes = self._discover_recipes()
        serving = tuple(
            info
            for info in self._recipes
            if procedure in tuple(getattr(info, "procedures", ()))
            or _ANY_PROCEDURE in tuple(getattr(info, "procedures", ()))
        )

        chosen = ""
        try:
            from i2as.analysis.discovery import recipe_for
        except ImportError:
            pass
        else:
            try:
                picked = recipe_for(
                    procedure, serving, self._preferred_recipe(procedure)
                )
            except Exception:  # noqa: BLE001 - a bad pick is not a broken panel
                logger.exception("eLab tab: could not pick a default recipe")
                picked = None
            chosen = str(getattr(picked, "name", "") or "")

        self._recipe_combo.clear()
        for info in serving:
            name = str(getattr(info, "name", ""))
            experiment_own = str(getattr(info, "origin", "")) == "experiment"
            label = name + (EXPERIMENT_SUFFIX if experiment_own else "")
            self._recipe_combo.addItem(label, name)
            self._recipe_combo.setItemData(
                self._recipe_combo.count() - 1,
                str(getattr(info, "description", "")),
                Qt.ItemDataRole.ToolTipRole,
            )
        index = self._recipe_combo.findData(chosen)
        if index >= 0:
            self._recipe_combo.setCurrentIndex(index)
        self._recipe_combo.setEnabled(self._recipe_combo.count() > 0)

    # ------------------------------------------------------------------
    # The pending entry and the report
    # ------------------------------------------------------------------

    def pending_entry(self, run_id: str = "") -> dict[str, Any]:
        """Return the **pending entry** parked on one run, or ``{}``.

        Args:
            run_id: The run to read; ``""`` uses the current selection.

        Returns:
            The entry's JSON dict, or ``{}`` when none is waiting.
        """
        wanted = run_id or self.current_run_id()
        read = getattr(self._manager, "pending_eln_draft", None)
        if not wanted or not callable(read):
            return {}
        try:
            return dict(read(wanted) or {})
        except Exception:  # noqa: BLE001 - a view never raises into Qt
            logger.exception("eLab tab: could not read the pending entry")
            return {}

    def _report(self) -> AnalysisReport | None:
        """Return the selected run's analysis report, when one was written.

        Returns:
            The parsed ``AnalysisReport``, or ``None`` when there is no
            report file (or it is unreadable — logged, never raised).
        """
        report_dir = self._store_dir("report_dir")
        if report_dir is None:
            return None
        path = report_dir / REPORT_FILENAME
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return AnalysisReport.from_dict(payload)

    def _figure_width(self, declared: int) -> int:
        """Return the width one preview figure is rendered at, in pixels.

        Clamped to the preview's own viewport whenever that viewport has a
        real width, so a wide figure fits the pane instead of pushing a
        horizontal scrollbar under it. Before the panel is on screen there is
        no meaningful width to clamp to, and the recipe's declared one is
        used as it stands.

        Args:
            declared: The figure's ``width_px``, or ``0`` for "unspecified".

        Returns:
            The width to render at, or ``0`` to leave it to the image itself.
        """
        available = self._preview.viewport().width() - _PREVIEW_MARGIN_PX
        if available < _MIN_CLAMP_WIDTH_PX:
            return declared
        return min(declared, available) if declared else available

    def _preview_html(
        self,
        entry: Mapping[str, Any],
        report: AnalysisReport | None,
        report_dir: Path | None,
    ) -> str:
        """Render the entry as the operator sees it: figures, then the body.

        The figures are prepended here and ONLY here. The published body
        carries no image — a notebook entry's figures travel as attachments —
        so this is a local view of local files, not a second renderer.

        Args:
            entry: The pending entry's dict (``title``/``body_html``).
            report: The run's analysis report, when one was written.
            report_dir: The directory that report's figures live in.

        Returns:
            The HTML for the preview browser.
        """
        title = html.escape(str(entry.get("title", "")))
        parts = [f"<h3>{title}</h3>"] if title else []

        if report is not None and report_dir is not None:
            for figure in report.figures:
                url = QUrl.fromLocalFile(str(report_dir / figure.file)).toString()
                width = self._figure_width(figure.width_px)
                attribute = f' width="{width}"' if width else ""
                parts.append(f'<p><img src="{html.escape(url)}"{attribute}></p>')
                if figure.caption:
                    parts.append(
                        f"<p><i>{html.escape(figure.caption)}</i></p>"
                    )
        parts.append(str(entry.get("body_html", "")))
        return "\n".join(parts)

    def resizeEvent(self, event: Any) -> None:
        """Re-render the preview so a figure keeps fitting the new width.

        Args:
            event: The Qt resize event.
        """
        super().resizeEvent(event)
        if self._entry:
            self._preview.setHtml(
                self._preview_html(self._entry, self._report_shown, self._report_dir)
            )

    def _refresh_pending(self) -> None:
        """Repaint the preview, the warnings box, the buttons and the status."""
        run_id = self.current_run_id()
        entry = self.pending_entry(run_id)
        report = self._report() if entry else None
        report_dir = self._store_dir("report_dir") if entry else None
        # Held for the resize path, which re-renders the preview without
        # going back to disk for a report it already read.
        self._entry = dict(entry)
        self._report_shown = report
        self._report_dir = report_dir
        self._preview.setHtml(
            self._preview_html(entry, report, report_dir) if entry else ""
        )

        notes: list[str] = []
        if report is not None and str(entry.get("source", "")) != "model":
            notes.extend(report.warnings)
            if report.error:
                notes.append(report.error.splitlines()[0])
        self._warnings.setPlainText("\n".join(notes))
        self._warnings.setVisible(bool(notes))

        pending = bool(entry)
        self._publish_btn.setEnabled(pending)
        self._discard_btn.setEnabled(pending)
        self._run_btn.setEnabled(
            self._runner is not None and bool(run_id) and not self._is_running(run_id)
        )
        self._new_recipe_btn.setEnabled(
            self._recipes_available and self._store_dir("recipes_dir") is not None
        )
        self._status_label.setText(self._status_text(run_id, pending))

    def _status_text(self, run_id: str, pending: bool) -> str:
        """Return the one line the status label shows.

        Args:
            run_id: The selected run, or ``""``.
            pending: Whether an entry is waiting on it.

        Returns:
            The status line, in priority order: not wired, no experiment, no
            runs, analysing, the last failure, ready, no recipes, nothing
            pending.
        """
        if self._manager is None:
            return NO_SESSION_TEXT
        if self._experiment() is None:
            return NO_EXPERIMENT_TEXT
        if not run_id:
            return NO_RUNS_TEXT
        if self._is_running(run_id):
            return ANALYSING_TEXT
        failure = self._failures.get(run_id, "")
        if failure:
            return f"Analysis failed: {failure}"
        if pending:
            return READY_TEXT
        if not self._recipes_available:
            return NO_RECIPES_TEXT
        return NOTHING_PENDING_TEXT

    def _is_running(self, run_id: str) -> bool:
        """Return whether the runner is analysing one run right now.

        Args:
            run_id: The run to ask about.

        Returns:
            ``False`` when no runner is wired or it refused the question.
        """
        is_running = getattr(self._runner, "is_running", None)
        if not callable(is_running):
            return False
        try:
            return bool(is_running(run_id))
        except Exception:  # noqa: BLE001 - a view never raises into Qt
            logger.exception("eLab tab: could not read the runner's state")
            return False

    # ------------------------------------------------------------------
    # The publish-state chip and the analysis toggle
    # ------------------------------------------------------------------

    def _refresh_chip(self) -> None:
        """Repaint the chip from the publisher's current status."""
        status = getattr(self._publisher, "status", None)
        if not callable(status):
            return
        try:
            self.on_publish_state(dict(status() or {}))
        except Exception:  # noqa: BLE001 - a view never raises into Qt
            logger.exception("eLab tab: could not read the publish state")

    def on_publish_state(self, status: Mapping[str, Any]) -> None:
        """Render one publish-state update on the chip.

        Args:
            status: ``{"state", "pending", "detail"}`` as the publisher's
                ``publish_state_changed`` carries it.
        """
        state = str(status.get("state", "disabled")) or "disabled"
        text = _CHIP_TEXT.get(state, f"eLab · {state}")
        queued = status.get("pending", 0)
        if state == "pending" and queued:
            text = f"{text} · {queued}"
        self._chip.setText(text)
        self._chip.setProperty("state", state)
        detail = str(status.get("detail", ""))
        self._chip.setToolTip(detail or "Whether everything queued reached the notebook")
        style = self._chip.style()
        style.unpolish(self._chip)
        style.polish(self._chip)

    def _refresh_analysis_toggle(self) -> None:
        """Reflect ``settings.analysis.enabled`` without writing it back."""
        analysis = getattr(getattr(self._publisher, "settings", None), "analysis", None)
        self._loading = True
        try:
            self._enabled_checkbox.setChecked(bool(getattr(analysis, "enabled", False)))
            self._enabled_checkbox.setEnabled(analysis is not None)
        finally:
            self._loading = False

    def _on_analysis_toggled(self, checked: bool) -> None:
        """Persist the analysis on/off switch through the settings file.

        Args:
            checked: The checkbox's new state.
        """
        if self._loading or self._publisher is None:
            return
        settings = getattr(self._publisher, "settings", None)
        analysis = getattr(settings, "analysis", None)
        if settings is None or analysis is None:
            logger.warning("This build stores no analysis settings — nothing saved")
            return
        persist_eln_settings(
            replace(settings, analysis=replace(analysis, enabled=bool(checked))),
            self._publisher,
        )
        self.reload()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_experiment_changed(self, _record: Mapping[str, Any]) -> None:
        """Rebuild everything when the open experiment changes.

        Args:
            _record: The experiment as a dict; the panel re-reads the
                manager, which is the single writer of that record.
        """
        self.reload()

    def _on_run_recorded(self, _record: Mapping[str, Any]) -> None:
        """Re-read runs and the pending entry when a run record changes.

        Args:
            _record: The ``RunRecord`` as a dict; the panel re-reads the
                manager rather than the payload.
        """
        self.reload()

    def _on_run_selected(self, _index: int) -> None:
        """Follow the run combo: new run, new recipes, new pending entry.

        Args:
            _index: The combo's new index; the panel reads the selection.
        """
        self._reload_recipes()
        self._refresh_pending()

    def on_analysis_started(self, run_id: str) -> None:
        """Show that a recipe is running.

        Args:
            run_id: The run being analysed.
        """
        self._failures.pop(run_id, None)
        if run_id == self.current_run_id():
            self._status_label.setText(ANALYSING_TEXT)
            self._run_btn.setEnabled(False)

    def on_analysis_finished(self, run_id: str, _report: Mapping[str, Any]) -> None:
        """Refresh once a recipe has finished.

        Args:
            run_id: The run that was analysed.
            _report: The report as a dict; the panel re-reads the report file
                and the manager, so one slot serves every path.
        """
        self._failures.pop(run_id, None)
        self.reload()
        if run_id:
            self.set_run(run_id)

    def on_analysis_failed(self, run_id: str, message: str) -> None:
        """Record and show a failed analysis.

        Args:
            run_id: The run whose analysis failed.
            message: The failure, one line.
        """
        self._failures[run_id] = str(message).splitlines()[0] if message else "unknown"
        self.reload()
        if run_id:
            self.set_run(run_id)

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def _on_setup_clicked(self) -> None:
        """Open the **eLab setup dialog**, when an opener was supplied."""
        if self._open_settings is None:
            return
        self._open_settings()
        self.reload()

    def _on_new_recipe_clicked(self) -> None:
        """Scaffold a recipe into the experiment's folder and open it."""
        recipes_dir = self._store_dir("recipes_dir")
        if recipes_dir is None:
            return
        name, accepted = QInputDialog.getText(self, "New recipe", "Recipe name:")
        if not accepted or not name.strip():
            return
        try:
            from i2as.analysis.discovery import scaffold_recipe
        except ImportError:
            self._status_label.setText(NO_RECIPES_TEXT)
            return
        try:
            path = Path(scaffold_recipe(name.strip(), recipes_dir, self._procedure_of()))
        except Exception as exc:  # noqa: BLE001 - a refusal is a status line
            logger.warning("Could not scaffold recipe %r: %s", name, exc)
            self._status_label.setText(f"Could not create the recipe: {exc}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        self._reload_recipes()
        index = self._recipe_combo.findData(name.strip())
        if index >= 0:
            self._recipe_combo.setCurrentIndex(index)

    def _on_run_analysis_clicked(self) -> None:
        """Start the runner for the selected run and recipe."""
        run_id = self.current_run_id()
        start = getattr(self._runner, "start", None)
        if not run_id or not callable(start):
            return
        try:
            started = start(run_id, recipe=self.selected_recipe())
        except Exception as exc:  # noqa: BLE001 - a refusal is a status line
            logger.exception("eLab tab: starting the analysis failed")
            self._status_label.setText(f"Analysis failed: {exc}")
            return
        if not started:
            self._status_label.setText(
                "Analysis could not start — the run has no data file, or no "
                "experiment is open"
            )
            return
        self._status_label.setText(ANALYSING_TEXT)
        self._run_btn.setEnabled(False)

    def _on_publish_clicked(self) -> None:
        """Approve the pending entry — the human half of the approval gate."""
        run_id = self.current_run_id()
        approve = getattr(self._manager, "approve_eln_draft", None)
        if not run_id or not callable(approve):
            return
        job_id = ""
        try:
            job_id = str(approve(run_id) or "")
        except Exception:  # noqa: BLE001 - approval must not raise into Qt
            logger.exception("eLab tab: approving the entry failed")
        if not job_id:
            logger.warning("eLab tab: nothing was queued for run %s", run_id)
        self.reload()
        if not job_id:
            self._status_label.setText(
                "Nothing was queued — check the notebook settings"
            )

    def _on_discard_clicked(self) -> None:
        """Drop the pending entry; the run simply stays unpublished."""
        run_id = self.current_run_id()
        discard = getattr(self._manager, "discard_pending_eln_draft", None)
        if not run_id or not callable(discard):
            logger.warning("This build cannot discard a pending entry")
            return
        try:
            discard(run_id)
        except Exception:  # noqa: BLE001 - a discard must not raise into Qt
            logger.exception("eLab tab: discarding the entry failed")
        self.reload()
