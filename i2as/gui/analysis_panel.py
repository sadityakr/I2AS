"""The **Analysis tab** — analyse finished runs, choose what represents them, publish.

The procedure window's top-right quadrant carries two tabs: "Queue", the run
queue over the status log, and "Analysis", this panel. It shows the two
independent layers side by side and keeps them apart:

- **Analysis.** Pick a finished run; run a recipe over it (in its container);
  see every **analysis bundle** the run has — a recipe's, a script's, a
  draft's — and choose the one that represents the run
  (``ExperimentManager.select_bundle``). The preview shows the chosen bundle
  from its local, sealed files. Nothing here involves a notebook.
- **Notebook.** The strip at the bottom is the experiment's ONE notebook
  page: link it (or create it), read fields back into the sample metadata,
  approve publishing once for the whole experiment, and publish every
  finished run not yet on the page as one appended, timestamped section.
  Every notebook call goes through the ``ElnService`` on its worker thread.

The panel writes no record itself — the manager is the single writer of
experiment state — and every collaborator is optional: with none wired (a
unit test) the panel builds and says so.
"""

from __future__ import annotations

import html
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

from i2as.analysis.bundle import Bundle, verify_artifact
from i2as.gui import app_settings
from i2as.gui.theme import BTN_CLASS_PRIMARY, BTN_CLASS_SECONDARY

logger = logging.getLogger(__name__)

#: Status line: no session layer is wired at all.
NO_SESSION_TEXT = "Session layer not wired"

#: Status line: a session layer is wired, but no experiment is open.
NO_EXPERIMENT_TEXT = "No experiment is open"

#: Status line: the open experiment has no finished run yet.
NO_RUNS_TEXT = "No finished runs in this experiment yet"

#: Status line: a recipe is running for the selected run.
ANALYSING_TEXT = "Analysing…"

#: Status line: the selected run has no bundle yet.
NO_BUNDLES_TEXT = "This run has not been analysed yet"

#: Status line: this build carries no analysis recipes to choose from.
NO_RECIPES_TEXT = "Analysis recipes are not available in this build"

#: Marker of the bundle that represents the run.
SELECTED_MARK = "★ "

#: Suffix marking a recipe that lives in the open experiment's own folder.
EXPERIMENT_SUFFIX = " (experiment)"

_ANY_PROCEDURE = "*"
_PREVIEW_MARGIN_PX = 28
_MIN_CLAMP_WIDTH_PX = 200

#: Notebook states the chip renders, from ``ElnService.status_changed``.
_CHIP_TEXT = {
    "synced": "Notebook · synced",
    "pending": "Notebook · sending",
    "offline": "Notebook · offline",
    "attention": "Notebook · needs attention",
    "disabled": "Notebook · off",
}


def _run_is_finished(run: Any) -> bool:
    """Whether one run record is over (whatever its outcome)."""
    return str(getattr(run, "status", "")) != "running"


def _bundle_label(bundle: Bundle, selected: str) -> str:
    """One line naming a bundle in the combo."""
    mark = SELECTED_MARK if bundle.bundle_id == selected else ""
    when = bundle.created_utc.replace("T", " ")[:16]
    state = "" if bundle.ok else " · FAILED"
    return f"{mark}{bundle.producer.kind} {bundle.producer.name or ''} · {when}{state}".replace("  ", " ")


class AnalysisPanel(QWidget):
    """The **Analysis tab**.

    Named widgets (``findChild`` objectNames are API): ``analysis_panel``,
    ``analysis_publish_chip``, ``analysis_enabled_checkbox``,
    ``notebook_settings_btn``, ``analysis_run_combo``, ``analysis_run_btn``,
    ``analysis_recipe_combo``, ``analysis_new_recipe_btn``,
    ``analysis_bundle_combo``, ``analysis_select_bundle_btn``,
    ``analysis_status_label``, ``analysis_preview``, ``analysis_warnings``,
    ``notebook_status_label``, ``notebook_link_btn``,
    ``notebook_read_fields_btn``, ``notebook_approve_btn``,
    ``notebook_publish_btn``, ``notebook_retry_btn``.

    Args:
        session_manager: The ``ExperimentManager``.
        eln_service: The ``ElnService``, or ``None`` (the notebook strip is
            then inert).
        analysis_runner: The ``AnalysisRunner``, or ``None``.
        open_settings: Opens the Settings dialog on its notebook page.
        config_store: The ``AppConfigStore``; ``None`` for the process-wide one.
        engine_checker: Checks the container engine before analysis goes on.
        dialog_factory: Opens the link / read-fields dialogs
            (``(kind, service, experiment, manager, parent) -> None``); tests
            replace it. ``None`` uses ``notebook_dialogs``.
        parent: Optional Qt parent.
    """

    def __init__(
        self,
        *,
        session_manager: Any | None = None,
        eln_service: Any | None = None,
        analysis_runner: Any | None = None,
        open_settings: Callable[[], None] | None = None,
        config_store: Any | None = None,
        engine_checker: Callable[[Any], Any] | None = None,
        dialog_factory: Callable[..., Any] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("analysis_panel")
        self._config_store = config_store
        self._engine_checker = engine_checker
        self._manager = session_manager
        self._service = eln_service
        self._runner = analysis_runner
        self._open_settings = open_settings
        self._dialog_factory = dialog_factory
        self._recipes: tuple[Any, ...] = ()
        self._recipes_available = True
        self._failures: dict[str, str] = {}
        self._loading = False
        self._bundle_shown: Bundle | None = None
        self._bundle_dir: Path | None = None
        self._notice = ""

        self._build_ui()
        self._connect_collaborators()
        self.reload()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
        root.addLayout(self._build_header_row())
        root.addLayout(self._build_run_rows())
        self._status_label = QLabel(NO_SESSION_TEXT)
        self._status_label.setObjectName("analysis_status_label")
        self._status_label.setProperty("class", "secondary_label")
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)
        self._preview = QTextBrowser()
        self._preview.setObjectName("analysis_preview")
        self._preview.setOpenExternalLinks(True)
        self._preview.setMinimumHeight(140)
        root.addWidget(self._preview, stretch=1)
        self._warnings = QTextEdit()
        self._warnings.setObjectName("analysis_warnings")
        self._warnings.setReadOnly(True)
        self._warnings.setMaximumHeight(64)
        self._warnings.hide()
        root.addWidget(self._warnings)
        root.addLayout(self._build_notebook_rows())

    def _build_header_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._chip = QLabel(_CHIP_TEXT["disabled"])
        self._chip.setObjectName("analysis_publish_chip")
        self._chip.setProperty("class", "publish_chip")
        self._chip.setProperty("state", "disabled")
        row.addWidget(self._chip)
        self._enabled_checkbox = QCheckBox("Analyse finished runs")
        self._enabled_checkbox.setObjectName("analysis_enabled_checkbox")
        self._enabled_checkbox.setToolTip("Run the preferred recipe over every finished run, in its container.")
        self._enabled_checkbox.toggled.connect(self._on_analysis_toggled)
        row.addWidget(self._enabled_checkbox)
        row.addStretch()
        self._settings_btn = QPushButton("Notebook settings…")
        self._settings_btn.setObjectName("notebook_settings_btn")
        self._settings_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._settings_btn.setEnabled(self._open_settings is not None)
        self._settings_btn.clicked.connect(self._on_settings_clicked)
        row.addWidget(self._settings_btn)
        return row

    def _build_run_rows(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setColumnStretch(1, 1)
        grid.addWidget(QLabel("Run:"), 0, 0)
        self._run_combo = QComboBox()
        self._run_combo.setObjectName("analysis_run_combo")
        self._run_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._run_combo.currentIndexChanged.connect(self._on_run_selected)
        grid.addWidget(self._run_combo, 0, 1)
        self._run_btn = QPushButton("Run analysis")
        self._run_btn.setObjectName("analysis_run_btn")
        self._run_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._run_btn.clicked.connect(self._on_run_analysis_clicked)
        grid.addWidget(self._run_btn, 0, 2)

        grid.addWidget(QLabel("Recipe:"), 1, 0)
        self._recipe_combo = QComboBox()
        self._recipe_combo.setObjectName("analysis_recipe_combo")
        self._recipe_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        grid.addWidget(self._recipe_combo, 1, 1)
        self._new_recipe_btn = QPushButton("New recipe…")
        self._new_recipe_btn.setObjectName("analysis_new_recipe_btn")
        self._new_recipe_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._new_recipe_btn.clicked.connect(self._on_new_recipe_clicked)
        grid.addWidget(self._new_recipe_btn, 1, 2)

        grid.addWidget(QLabel("Result:"), 2, 0)
        self._bundle_combo = QComboBox()
        self._bundle_combo.setObjectName("analysis_bundle_combo")
        self._bundle_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._bundle_combo.setToolTip(f"Every analysis of this run. {SELECTED_MARK}marks the one that represents it.")
        self._bundle_combo.currentIndexChanged.connect(self._on_bundle_shown)
        grid.addWidget(self._bundle_combo, 2, 1)
        self._select_btn = QPushButton("Use for this run")
        self._select_btn.setObjectName("analysis_select_bundle_btn")
        self._select_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._select_btn.setToolTip("Make the shown result the one that represents the run (and is published for it).")
        self._select_btn.clicked.connect(self._on_select_clicked)
        grid.addWidget(self._select_btn, 2, 2)
        return grid

    def _build_notebook_rows(self) -> QVBoxLayout:
        box = QVBoxLayout()
        self._notebook_label = QLabel("")
        self._notebook_label.setObjectName("notebook_status_label")
        self._notebook_label.setWordWrap(True)
        self._notebook_label.setOpenExternalLinks(True)
        self._notebook_label.setTextFormat(Qt.TextFormat.RichText)
        box.addWidget(self._notebook_label)
        row = QHBoxLayout()
        self._link_btn = QPushButton("Link page…")
        self._link_btn.setObjectName("notebook_link_btn")
        self._link_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._link_btn.clicked.connect(self._on_link_clicked)
        row.addWidget(self._link_btn)
        self._fields_btn = QPushButton("Read fields…")
        self._fields_btn.setObjectName("notebook_read_fields_btn")
        self._fields_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._fields_btn.setToolTip("Read the page's and linked samples' fields into the sample metadata")
        self._fields_btn.clicked.connect(self._on_read_fields_clicked)
        row.addWidget(self._fields_btn)
        self._retry_btn = QPushButton("Retry")
        self._retry_btn.setObjectName("notebook_retry_btn")
        self._retry_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._retry_btn.clicked.connect(self._on_retry_clicked)
        row.addWidget(self._retry_btn)
        row.addStretch()
        self._approve_btn = QPushButton("Approve publishing")
        self._approve_btn.setObjectName("notebook_approve_btn")
        self._approve_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._approve_btn.setToolTip("Allow this experiment's finished runs to be appended to its page from now on")
        self._approve_btn.clicked.connect(self._on_approve_clicked)
        row.addWidget(self._approve_btn)
        self._publish_btn = QPushButton("Publish new runs")
        self._publish_btn.setObjectName("notebook_publish_btn")
        self._publish_btn.setProperty("class", BTN_CLASS_PRIMARY)
        self._publish_btn.setToolTip("Append every finished run not yet on the page as one timestamped section")
        self._publish_btn.clicked.connect(self._on_publish_clicked)
        row.addWidget(self._publish_btn)
        box.addLayout(row)
        return box

    def _connect_collaborators(self) -> None:
        self._connect(self._manager, "experiment_changed", self._on_changed)
        self._connect(self._manager, "run_recorded", self._on_changed)
        self._connect(self._service, "status_changed", self.on_publish_state)
        self._connect(self._service, "publish_finished", self._on_publish_finished)
        self._connect(self._service, "publish_failed", self._on_publish_failed)
        self._connect(self._service, "page_ready", self._on_changed)
        self._connect(self._runner, "analysis_started", self.on_analysis_started)
        self._connect(self._runner, "analysis_finished", self.on_analysis_finished)
        self._connect(self._runner, "analysis_failed", self.on_analysis_failed)
        self._connect(self._runner, "bundle_ready", self._on_bundle_ready)

    @staticmethod
    def _connect(source: Any | None, name: str, slot: Callable[..., None]) -> None:
        signal = getattr(source, name, None)
        connect = getattr(signal, "connect", None)
        if callable(connect):
            connect(slot)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def _experiment(self) -> Any | None:
        if self._manager is None:
            return None
        try:
            return self._manager.current_experiment()
        except Exception:  # noqa: BLE001 - a view never raises into Qt
            logger.exception("Analysis tab: could not read the open experiment")
            return None

    def current_run_id(self) -> str:
        """The run the panel is showing, or ``""``."""
        return str(self._run_combo.currentData() or "")

    def selected_recipe(self) -> str:
        """The selected recipe's name, or ``""`` for "let the runner pick"."""
        return str(self._recipe_combo.currentData() or "")

    def shown_bundle_id(self) -> str:
        """The bundle the preview shows, or ``""``."""
        return str(self._bundle_combo.currentData() or "")

    def _run_record(self, run_id: str = "") -> Any | None:
        wanted = run_id or self.current_run_id()
        for run in getattr(self._experiment(), "runs", ()):
            if str(getattr(run, "run_id", "")) == wanted:
                return run
        return None

    def _procedure_of(self, run_id: str = "") -> str:
        return str(getattr(self._run_record(run_id), "procedure", "") or "")

    def _store(self) -> Any:
        return getattr(self._manager, "store", None)

    def _recipes_dir(self) -> Path | None:
        record = self._experiment()
        store = self._store()
        if record is None or store is None or not hasattr(store, "recipes_dir"):
            return None
        return Path(store.recipes_dir(record.experiment_id))

    def bundles(self, run_id: str = "") -> list[Bundle]:
        """Every bundle of one run (the selected run by default), oldest first."""
        record = self._experiment()
        store = self._store()
        wanted = run_id or self.current_run_id()
        if record is None or store is None or not wanted or not hasattr(store, "list_bundles"):
            return []
        try:
            return list(store.list_bundles(record.experiment_id, wanted))
        except Exception:  # noqa: BLE001
            logger.exception("Analysis tab: could not list the bundles of %s", wanted)
            return []

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Re-read everything and repaint (the one refresh path)."""
        self._refresh_chip()
        self._refresh_analysis_toggle()
        self._reload_runs()
        self._reload_recipes()
        self._reload_bundles()
        self._refresh_notebook()

    def set_run(self, run_id: str) -> None:
        """Select one run and refresh what follows from it."""
        index = self._run_combo.findData(str(run_id))
        if index < 0:
            self._reload_runs()
            index = self._run_combo.findData(str(run_id))
        if index >= 0:
            self._run_combo.setCurrentIndex(index)
        self._reload_recipes()
        self._reload_bundles()

    def on_run_finished(self, manifest: Mapping[str, Any] | None) -> None:
        """Select the run that just finished."""
        self.reload()
        run_id = str((manifest or {}).get("run_id", ""))
        if run_id:
            self.set_run(run_id)

    def _reload_runs(self) -> None:
        selected = self.current_run_id()
        runs = [run for run in getattr(self._experiment(), "runs", ()) if _run_is_finished(run)]
        runs.reverse()
        self._run_combo.blockSignals(True)
        self._run_combo.clear()
        for run in runs:
            published = " · on page" if getattr(run, "published", False) else ""
            label = f"{run.run_id} · {getattr(run, 'procedure', '')} · {getattr(run, 'status', '')}{published}"
            self._run_combo.addItem(label, run.run_id)
        index = self._run_combo.findData(selected)
        if index >= 0:
            self._run_combo.setCurrentIndex(index)
        self._run_combo.blockSignals(False)

    def _reload_recipes(self) -> None:
        procedure = self._procedure_of()
        try:
            from i2as.analysis.discovery import discover_recipes, recipe_for
        except ImportError:
            self._recipes_available = False
            self._recipe_combo.clear()
            self._recipe_combo.setEnabled(False)
            return
        self._recipes_available = True
        extra = [self._recipes_dir()] if self._recipes_dir() is not None else []
        try:
            self._recipes = tuple(discover_recipes(extra))
        except Exception:  # noqa: BLE001
            logger.exception("Analysis tab: recipe discovery failed")
            self._recipes = ()
        serving = tuple(
            info
            for info in self._recipes
            if procedure in tuple(getattr(info, "procedures", ())) or _ANY_PROCEDURE in tuple(getattr(info, "procedures", ()))
        )
        preferred = ""
        recipes_pref = getattr(self._config().analysis(), "recipes", None)
        if isinstance(recipes_pref, Mapping):
            preferred = str(recipes_pref.get(procedure, "") or "")
        try:
            picked = recipe_for(procedure, serving, preferred)
        except Exception:  # noqa: BLE001
            picked = None
        chosen = str(getattr(picked, "name", "") or "")
        self._recipe_combo.clear()
        for info in serving:
            name = str(getattr(info, "name", ""))
            own = str(getattr(info, "origin", "")) == "experiment"
            self._recipe_combo.addItem(name + (EXPERIMENT_SUFFIX if own else ""), name)
            self._recipe_combo.setItemData(self._recipe_combo.count() - 1, str(getattr(info, "description", "")), Qt.ItemDataRole.ToolTipRole)
        index = self._recipe_combo.findData(chosen)
        if index >= 0:
            self._recipe_combo.setCurrentIndex(index)
        self._recipe_combo.setEnabled(self._recipe_combo.count() > 0)

    def _reload_bundles(self) -> None:
        run = self._run_record()
        selected = str(getattr(run, "selected_bundle", "") or "")
        shown = self.shown_bundle_id()
        bundles = self.bundles()
        self._bundle_combo.blockSignals(True)
        self._bundle_combo.clear()
        for bundle in reversed(bundles):
            self._bundle_combo.addItem(_bundle_label(bundle, selected), bundle.bundle_id)
        target = shown if self._bundle_combo.findData(shown) >= 0 else selected
        index = self._bundle_combo.findData(target)
        self._bundle_combo.setCurrentIndex(index if index >= 0 else 0)
        self._bundle_combo.blockSignals(False)
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        run_id = self.current_run_id()
        record = self._experiment()
        store = self._store()
        bundle_id = self.shown_bundle_id()
        bundle = None
        folder = None
        if record is not None and store is not None and bundle_id:
            bundle = store.read_bundle(record.experiment_id, run_id, bundle_id)
            folder = Path(store.bundle_dir(record.experiment_id, run_id, bundle_id)) if bundle is not None else None
        self._bundle_shown = bundle
        self._bundle_dir = folder
        self._preview.setHtml(self._preview_html(bundle, folder) if bundle is not None else "")
        notes = list(bundle.warnings) if bundle is not None else []
        if bundle is not None and bundle.error:
            notes.append(bundle.error.strip().splitlines()[0])
        self._warnings.setPlainText("\n".join(notes))
        self._warnings.setVisible(bool(notes))
        selected = str(getattr(self._run_record(), "selected_bundle", "") or "")
        self._select_btn.setEnabled(bool(bundle is not None and bundle.ok and bundle.sealed and bundle_id != selected))
        self._run_btn.setEnabled(self._runner is not None and bool(run_id) and not self._is_running(run_id))
        self._new_recipe_btn.setEnabled(self._recipes_available and self._recipes_dir() is not None)
        self._status_label.setText(self._notice or self._status_text(run_id, bundle))

    def _status_text(self, run_id: str, bundle: Bundle | None) -> str:
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
        if bundle is None:
            return NO_BUNDLES_TEXT if self._recipes_available else NO_RECIPES_TEXT
        run = self._run_record(run_id)
        if bundle.bundle_id == getattr(run, "selected_bundle", ""):
            return "This result represents the run."
        return "Not the result that represents the run — press “Use for this run” to choose it."

    def _figure_width(self, declared: int) -> int:
        available = self._preview.viewport().width() - _PREVIEW_MARGIN_PX
        if available < _MIN_CLAMP_WIDTH_PX:
            return declared
        return min(declared, available) if declared else available

    def _preview_html(self, bundle: Bundle, folder: Path | None) -> str:
        """Render one bundle for the operator, from its local sealed files."""
        parts: list[str] = [
            f"<h3>{html.escape(bundle.producer.kind)} {html.escape(bundle.producer.name)}</h3>"
        ]
        if not bundle.ok:
            parts.append(f"<p><b>Failed:</b> {html.escape(bundle.error.strip().splitlines()[0] if bundle.error.strip() else '')}</p>")
        for paragraph in bundle.summary:
            parts.append(f"<p>{html.escape(paragraph)}</p>")
        if bundle.results:
            rows = "".join(
                f"<tr><td>{html.escape(str(r.get('name', '')))}</td><td>{html.escape(str(r.get('value', '')))}"
                f"{' ± ' + html.escape(str(r.get('uncertainty'))) if r.get('uncertainty') not in (None, '') else ''} "
                f"{html.escape(str(r.get('unit') or ''))}</td></tr>"
                for r in bundle.results
            )
            parts.append(f"<table border='1' cellpadding='3'>{rows}</table>")
        for artifact in bundle.artifacts:
            if artifact.kind != "figure" or folder is None:
                continue
            path = verify_artifact(folder, artifact)
            if path is None:
                continue
            url = QUrl.fromLocalFile(str(path)).toString()
            width = self._figure_width(0)
            attribute = f' width="{width}"' if width else ""
            parts.append(f'<p><img src="{html.escape(url)}"{attribute}></p>')
            if artifact.caption:
                parts.append(f"<p><i>{html.escape(artifact.caption)}</i></p>")
        for spec in bundle.tables:
            head = "".join(f"<th>{html.escape(str(c))}</th>" for c in spec.get("columns") or [])
            body = "".join(
                "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in spec.get("rows") or []
            )
            parts.append(f"<p><b>{html.escape(str(spec.get('caption', '')))}</b></p><table border='1' cellpadding='3'><tr>{head}</tr>{body}</table>")
        return "\n".join(parts)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if self._bundle_shown is not None:
            self._preview.setHtml(self._preview_html(self._bundle_shown, self._bundle_dir))

    def _is_running(self, run_id: str) -> bool:
        is_running = getattr(self._runner, "is_running", None)
        if not callable(is_running):
            return False
        try:
            return bool(is_running(run_id))
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # The notebook strip
    # ------------------------------------------------------------------

    def _refresh_notebook(self) -> None:
        record = self._experiment()
        binding = getattr(record, "eln", None)
        service = self._service
        enabled = False
        if service is not None and record is not None:
            try:
                enabled = bool(service.enabled(record.user_id or "guest"))
            except Exception:  # noqa: BLE001
                enabled = False
        if record is None:
            text = ""
        elif service is None:
            text = "No notebook service in this session."
        elif not enabled:
            text = "Publishing is off for this user (Notebook settings…)."
        elif binding is None:
            text = "Not linked to a notebook page yet."
        elif binding.entry is None:
            text = "The notebook page is being created…"
        else:
            url = html.escape(binding.entry.url or binding.entry.entry_id)
            text = f'Page: <a href="{url}">{url}</a>'
            if not binding.publish_approved:
                text += " · publishing not approved yet"
            else:
                pending = [r for r in getattr(record, "runs", ()) if _run_is_finished(r) and not getattr(r, "published", False)]
                text += f" · {len(pending)} run(s) not on the page yet" if pending else " · every finished run is on the page"
        self._notebook_label.setText(text)
        linked = binding is not None
        usable = service is not None and record is not None and enabled
        self._link_btn.setEnabled(usable)
        self._link_btn.setText("Change page…" if linked else "Link page…")
        self._fields_btn.setEnabled(usable and linked)
        self._approve_btn.setVisible(linked and not getattr(binding, "publish_approved", False))
        self._approve_btn.setEnabled(usable and linked)
        self._publish_btn.setEnabled(usable and linked and bool(getattr(binding, "publish_approved", False)))
        status = {}
        if service is not None:
            try:
                status = dict(service.status(getattr(record, "experiment_id", "")))
            except Exception:  # noqa: BLE001
                status = {}
        self._retry_btn.setVisible(bool(status.get("attention")))

    def _refresh_chip(self) -> None:
        status = getattr(self._service, "status", None)
        if not callable(status):
            return
        try:
            self.on_publish_state(dict(status() or {}))
        except Exception:  # noqa: BLE001
            logger.exception("Analysis tab: could not read the notebook status")

    def on_publish_state(self, status: Mapping[str, Any]) -> None:
        """Render one notebook-status update on the chip."""
        state = str(status.get("state", "disabled")) or "disabled"
        text = _CHIP_TEXT.get(state, f"Notebook · {state}")
        if state == "pending" and status.get("pending"):
            text = f"{text} · {status['pending']}"
        self._chip.setText(text)
        self._chip.setProperty("state", state)
        attention = status.get("attention") or []
        detail = "; ".join(str(a.get("error", "")) for a in attention) or str(status.get("detail", ""))
        self._chip.setToolTip(detail or "Whether everything queued has reached the notebook")
        style = self._chip.style()
        style.unpolish(self._chip)
        style.polish(self._chip)
        self._retry_btn.setVisible(bool(attention))

    # ------------------------------------------------------------------
    # The analysis switch
    # ------------------------------------------------------------------

    def _config(self) -> Any:
        return self._config_store if self._config_store is not None else app_settings.config_store()

    def _refresh_analysis_toggle(self) -> None:
        self._loading = True
        try:
            self._enabled_checkbox.setChecked(bool(self._config().analysis().enabled))
        finally:
            self._loading = False

    def _on_analysis_toggled(self, checked: bool) -> None:
        if self._loading:
            return
        store = self._config()
        config = store.current
        if checked:
            checker = self._engine_checker
            if checker is None:
                from i2as.session.analysis_sandbox import check_engine

                checker = check_engine
            status = checker(config.analysis.sandbox)
            if not status.ready:
                self._refresh_analysis_toggle()
                self._status_label.setText(f"Analysis stays off — {status.detail} (Settings → Analysis)")
                return
        try:
            store.save(replace(config, analysis=replace(config.analysis, enabled=bool(checked))))
        except OSError as exc:
            logger.error("Could not save the analysis switch: %s", exc)
            self._refresh_analysis_toggle()
            return
        self.reload()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_changed(self, *_args: Any) -> None:
        self.reload()

    def _on_run_selected(self, _index: int) -> None:
        self._notice = ""
        self._reload_recipes()
        self._reload_bundles()

    def _on_bundle_shown(self, _index: int) -> None:
        self._notice = ""
        self._refresh_preview()

    def _on_bundle_ready(self, run_id: str, _bundle_id: str, _bundle: Mapping[str, Any]) -> None:
        if run_id == self.current_run_id():
            self._reload_bundles()

    def on_analysis_started(self, run_id: str) -> None:
        self._failures.pop(run_id, None)
        if run_id == self.current_run_id():
            self._status_label.setText(ANALYSING_TEXT)
            self._run_btn.setEnabled(False)

    def on_analysis_finished(self, run_id: str, _report: Mapping[str, Any]) -> None:
        self._failures.pop(run_id, None)
        self.reload()
        if run_id:
            self.set_run(run_id)

    def on_analysis_failed(self, run_id: str, message: str) -> None:
        self._failures[run_id] = str(message).splitlines()[0] if message else "unknown"
        self.reload()
        if run_id:
            self.set_run(run_id)

    def _on_publish_finished(self, info: Mapping[str, Any]) -> None:
        self._notice = f"Published {', '.join(sorted(info.get('run_bundles') or {}))} to the notebook."
        self.reload()

    def _on_publish_failed(self, info: Mapping[str, Any]) -> None:
        self._notice = f"Publishing failed: {info.get('reason', '')}"
        self.reload()

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def _on_settings_clicked(self) -> None:
        if self._open_settings is not None:
            self._open_settings()
            self.reload()

    def _on_new_recipe_clicked(self) -> None:
        recipes_dir = self._recipes_dir()
        if recipes_dir is None:
            return
        name, accepted = QInputDialog.getText(self, "New recipe", "Recipe name:")
        if not accepted or not name.strip():
            return
        try:
            from i2as.analysis.discovery import scaffold_recipe

            path = Path(scaffold_recipe(name.strip(), recipes_dir, self._procedure_of()))
        except Exception as exc:  # noqa: BLE001
            self._status_label.setText(f"Could not create the recipe: {exc}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        self._reload_recipes()
        index = self._recipe_combo.findData(name.strip())
        if index >= 0:
            self._recipe_combo.setCurrentIndex(index)

    def _on_run_analysis_clicked(self) -> None:
        run_id = self.current_run_id()
        start = getattr(self._runner, "start", None)
        if not run_id or not callable(start):
            return
        try:
            started = start(run_id, recipe=self.selected_recipe(), actor="operator")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Analysis tab: starting the analysis failed")
            self._status_label.setText(f"Analysis failed: {exc}")
            return
        if not started:
            self._status_label.setText("Analysis could not start — the run has no data file, or no experiment is open")
            return
        self._status_label.setText(ANALYSING_TEXT)
        self._run_btn.setEnabled(False)

    def _on_select_clicked(self) -> None:
        select = getattr(self._manager, "select_bundle", None)
        run_id = self.current_run_id()
        bundle_id = self.shown_bundle_id()
        if not callable(select) or not run_id or not bundle_id:
            return
        if not select(run_id, bundle_id):
            self._notice = "That result cannot represent the run (it failed, or is not sealed)."
        else:
            self._notice = ""
        self.reload()

    def _open_dialog(self, kind: str) -> None:
        record = self._experiment()
        if record is None or self._service is None:
            return
        if self._dialog_factory is not None:
            self._dialog_factory(kind, self._service, record, self._manager, self)
        else:
            from i2as.gui.notebook_dialogs import LinkNotebookDialog, ReadFieldsDialog

            dialog = LinkNotebookDialog(self._service, record, self) if kind == "link" else ReadFieldsDialog(self._service, self._manager, self)
            dialog.exec()
        self.reload()

    def _on_link_clicked(self) -> None:
        self._open_dialog("link")

    def _on_read_fields_clicked(self) -> None:
        self._open_dialog("fields")

    def _on_approve_clicked(self) -> None:
        record = self._experiment()
        approve = getattr(self._manager, "approve_eln_publishing", None)
        if record is None or not callable(approve):
            return
        user = app_settings.current_user_id() or record.user_id or "guest"
        approve(user)
        self.reload()

    def _on_publish_clicked(self) -> None:
        if self._service is None:
            return
        from i2as.session.eln.publishing import PublishError

        try:
            publish_id = self._service.publish()
        except PublishError as exc:
            self._notice = str(exc)
        else:
            self._notice = f"Publishing ({publish_id})…"
        self.reload()

    def _on_retry_clicked(self) -> None:
        record = self._experiment()
        if self._service is not None:
            self._service.retry(getattr(record, "experiment_id", ""))
        self.reload()
