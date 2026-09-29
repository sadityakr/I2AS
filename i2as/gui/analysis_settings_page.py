"""The **Analysis page** — whether and where finished runs are analysed.

One form over the ``analysis`` section of the general settings file
(``i2as.session.app_config``), shown as the Analysis page of the Settings
dialog (``i2as/gui/settings_dialog.py``). It edits the stage's switches —
on/off, the worker's timeout, the two report defaults — and the container
every analysis runs in: the engine, the image and the resource caps.

**Analysis needs a container engine.** Every recipe and script runs in a
container (``i2as.session.analysis_sandbox``), so Save refuses to switch
analysis on until ``check_engine()`` finds the engine running and the image
built, and says what is missing. "Check" asks the same question on demand;
"Build image" stages the image's build context and runs ``<engine> build``
in the background, streaming its output into the log box, then checks again.

**Unshown fields survive.** Save builds its answer with
``dataclasses.replace`` on the section it loaded, so the per-procedure
recipe preferences (edited nowhere in the GUI yet) are carried through.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QProcess, QProcessEnvironment
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from i2as.gui.theme import BTN_CLASS_PRIMARY, BTN_CLASS_SECONDARY
from i2as.session.analysis_sandbox import (
    SandboxError,
    build_image_command,
    check_engine,
    engine_environment,
    stage_image_context,
)
from i2as.session.app_config import CONTAINER_ENGINES, AnalysisSettings, SandboxSettings

logger = logging.getLogger(__name__)

#: What the engine line says before anyone has pressed Check.
NOT_CHECKED_TEXT = "Not checked yet — press Check."

#: Longest build log kept in the box, so a long build does not grow without bound.
_BUILD_LOG_MAX_BLOCKS = 2000


class AnalysisSettingsPage(QWidget):
    """The Analysis page: the stage's switches and its container.

    Named widgets (``findChild`` objectNames are API):
    ``settings_analysis_enabled_checkbox``, ``settings_analysis_timeout_input``,
    ``settings_analysis_facts_checkbox``, ``settings_analysis_attach_checkbox``,
    ``settings_analysis_engine_combo``, ``settings_analysis_image_edit``,
    ``settings_analysis_memory_edit``, ``settings_analysis_cpus_input``,
    ``settings_analysis_engine_label``, ``settings_analysis_check_btn``,
    ``settings_analysis_build_btn``, ``settings_analysis_build_log``,
    ``settings_analysis_save_btn`` and ``settings_analysis_status_label``.

    Args:
        store: The ``AppConfigStore`` this page reads and saves.
        on_saved: Called with the saved ``AnalysisSettings`` after every
            successful Save, so an open eLab tab can re-read its switch.
        engine_checker: ``check_engine`` unless a test injects a stand-in.
        parent: Optional Qt parent widget.
    """

    def __init__(
        self,
        store: Any,
        *,
        on_saved: Callable[[AnalysisSettings], None] | None = None,
        engine_checker: Callable[[SandboxSettings], Any] = check_engine,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("analysis_settings_page")
        self._store = store
        self._on_saved = on_saved
        self._engine_checker = engine_checker
        self._build_process: QProcess | None = None
        self._build_context: tempfile.TemporaryDirectory[str] | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        root.addWidget(self._build_stage_group())
        root.addWidget(self._build_container_group(), stretch=1)

        self._status_label = QLabel("")
        self._status_label.setObjectName("settings_analysis_status_label")
        self._status_label.setProperty("class", "secondary_label")
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        row = QHBoxLayout()
        row.addStretch(1)
        save_btn = QPushButton("Save")
        save_btn.setObjectName("settings_analysis_save_btn")
        save_btn.setProperty("class", BTN_CLASS_PRIMARY)
        save_btn.clicked.connect(self.save)
        row.addWidget(save_btn)
        root.addLayout(row)

        self._load(store.analysis())

    # ── Layout ────────────────────────────────────────────────────────

    def _build_stage_group(self) -> QGroupBox:
        group = QGroupBox("Analysis stage")
        form = QFormLayout(group)

        self._enabled_checkbox = QCheckBox("Analyse a finished run before its notebook entry")
        self._enabled_checkbox.setObjectName("settings_analysis_enabled_checkbox")
        self._enabled_checkbox.setToolTip(
            "The analysed entry still waits in the eLab tab for your approval. "
            "Needs the container engine running and the image built."
        )
        form.addRow(self._enabled_checkbox)

        self._timeout_input = QDoubleSpinBox()
        self._timeout_input.setObjectName("settings_analysis_timeout_input")
        self._timeout_input.setRange(1.0, 3600.0)
        self._timeout_input.setDecimals(0)
        self._timeout_input.setSuffix(" s")
        self._timeout_input.setToolTip("How long one analysis may run before its container is killed")
        form.addRow("Timeout:", self._timeout_input)

        self._facts_checkbox = QCheckBox("Append the run's own fact tables below the analysis")
        self._facts_checkbox.setObjectName("settings_analysis_facts_checkbox")
        form.addRow("Fact tables:", self._facts_checkbox)

        self._attach_checkbox = QCheckBox("Attach the raw data file")
        self._attach_checkbox.setObjectName("settings_analysis_attach_checkbox")
        form.addRow("Data file:", self._attach_checkbox)
        return group

    def _build_container_group(self) -> QGroupBox:
        group = QGroupBox("Container")
        layout = QVBoxLayout(group)
        form = QFormLayout()
        layout.addLayout(form)

        self._runtime_combo = QComboBox()
        self._runtime_combo.setObjectName("settings_analysis_engine_combo")
        self._runtime_combo.setEditable(True)
        self._runtime_combo.addItems(CONTAINER_ENGINES)
        self._runtime_combo.setToolTip(
            "The container engine's command, or a full path to it. Docker Desktop "
            "on Windows and macOS; Docker or Podman on Linux."
        )
        form.addRow("Engine:", self._runtime_combo)

        self._image_edit = QLineEdit()
        self._image_edit.setObjectName("settings_analysis_image_edit")
        self._image_edit.setToolTip(
            "The image every analysis runs in. Never pulled: build it here, or "
            "build your own FROM it with the extra libraries your lab needs."
        )
        form.addRow("Image:", self._image_edit)

        self._memory_edit = QLineEdit()
        self._memory_edit.setObjectName("settings_analysis_memory_edit")
        self._memory_edit.setPlaceholderText("e.g. 4g — blank for no cap")
        form.addRow("Memory cap:", self._memory_edit)

        self._cpus_input = QDoubleSpinBox()
        self._cpus_input.setObjectName("settings_analysis_cpus_input")
        self._cpus_input.setRange(0.0, 256.0)
        self._cpus_input.setDecimals(1)
        self._cpus_input.setSpecialValueText("no cap")
        form.addRow("CPU cap:", self._cpus_input)

        self._runtime_label = QLabel(NOT_CHECKED_TEXT)
        self._runtime_label.setObjectName("settings_analysis_engine_label")
        self._runtime_label.setWordWrap(True)
        form.addRow("Status:", self._runtime_label)

        button_row = QHBoxLayout()
        self._check_btn = QPushButton("Check")
        self._check_btn.setObjectName("settings_analysis_check_btn")
        self._check_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._check_btn.setToolTip("Ask the engine whether it is running and the image is built")
        self._check_btn.clicked.connect(self.check)
        button_row.addWidget(self._check_btn)
        self._build_btn = QPushButton("Build image")
        self._build_btn.setObjectName("settings_analysis_build_btn")
        self._build_btn.setProperty("class", BTN_CLASS_SECONDARY)
        self._build_btn.setToolTip(
            "Build the image from the Dockerfile shipped with I2AS. Needs the "
            "network once, to fetch the Python base image and libraries."
        )
        self._build_btn.clicked.connect(self.build_image)
        button_row.addWidget(self._build_btn)
        button_row.addStretch(1)
        form.addRow("", button_row)

        self._build_log = QPlainTextEdit()
        self._build_log.setObjectName("settings_analysis_build_log")
        self._build_log.setReadOnly(True)
        self._build_log.setMaximumBlockCount(_BUILD_LOG_MAX_BLOCKS)
        self._build_log.setPlaceholderText("Build output appears here.")
        layout.addWidget(self._build_log, stretch=1)
        return group

    # ── State ─────────────────────────────────────────────────────────

    def _load(self, analysis: AnalysisSettings) -> None:
        """Show one ``AnalysisSettings`` record in the form."""
        self._loaded = analysis
        self._enabled_checkbox.setChecked(analysis.enabled)
        self._timeout_input.setValue(float(analysis.timeout_s))
        self._facts_checkbox.setChecked(analysis.include_fact_tables)
        self._attach_checkbox.setChecked(analysis.attach_data_file)
        sandbox = analysis.sandbox
        self._runtime_combo.setCurrentText(sandbox.engine)
        self._image_edit.setText(sandbox.image)
        self._memory_edit.setText(sandbox.memory)
        self._cpus_input.setValue(float(sandbox.cpus))

    def sandbox_from_form(self) -> SandboxSettings:
        """Return the container settings the form shows."""
        loaded = self._loaded.sandbox
        return replace(
            loaded,
            engine=self._runtime_combo.currentText().strip() or loaded.engine,
            image=self._image_edit.text().strip() or loaded.image,
            memory=self._memory_edit.text().strip(),
            cpus=float(self._cpus_input.value()),
        )

    def settings_from_form(self) -> AnalysisSettings:
        """Return the ``AnalysisSettings`` the form shows, unshown fields carried through."""
        return replace(
            self._loaded,
            enabled=self._enabled_checkbox.isChecked(),
            timeout_s=float(self._timeout_input.value()),
            include_fact_tables=self._facts_checkbox.isChecked(),
            attach_data_file=self._attach_checkbox.isChecked(),
            sandbox=self.sandbox_from_form(),
        )

    # ── Actions ───────────────────────────────────────────────────────

    def check(self) -> Any:
        """Ask the engine about itself and the image, and show the answer.

        Returns:
            The ``EngineStatus``.
        """
        status = self._engine_checker(self.sandbox_from_form())
        self._runtime_label.setText(status.detail)
        self._runtime_label.setProperty("state", "ok" if status.ready else "warn")
        return status

    def save(self) -> bool:
        """Write the form to the settings file; refuse to switch analysis on without an engine.

        Returns:
            ``True`` when the settings were written.
        """
        edited = self.settings_from_form()
        if edited.enabled:
            status = self.check()
            if not status.ready:
                self._status_label.setText(
                    f"Not saved: analysis cannot be switched on yet. {status.detail}"
                )
                return False
        config = self._store.current
        try:
            self._store.save(replace(config, analysis=edited))
        except OSError as exc:
            logger.exception("Could not write the analysis settings")
            self._status_label.setText(f"Not saved: {exc}")
            return False
        self._loaded = edited
        self._status_label.setText(
            "Saved. Analysis is on." if edited.enabled else "Saved. Analysis is off."
        )
        if self._on_saved is not None:
            self._on_saved(edited)
        return True

    def build_image(self) -> None:
        """Stage the build context and run ``<engine> build`` in the background."""
        if self._build_process is not None:
            return
        sandbox = self.sandbox_from_form()
        context = tempfile.TemporaryDirectory(prefix="i2as-analysis-image-")
        try:
            stage_image_context(Path(context.name))
            command = build_image_command(sandbox, Path(context.name))
        except (OSError, SandboxError) as exc:
            context.cleanup()
            self._runtime_label.setText(f"Cannot build: {exc}")
            return
        self._build_context = context
        self._build_log.clear()
        self._build_log.appendPlainText("$ " + " ".join(command))
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment()
        for name, value in engine_environment(command[0]).items():
            environment.insert(name, value)
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._on_build_output)
        process.finished.connect(self._on_build_finished)
        process.errorOccurred.connect(self._on_build_error)
        self._build_process = process
        self._build_btn.setEnabled(False)
        self._runtime_label.setText(f"Building {sandbox.image}…")
        process.start(command[0], command[1:])

    def is_building(self) -> bool:
        """Whether an image build is running."""
        return self._build_process is not None

    def _on_build_output(self) -> None:
        if self._build_process is None:
            return
        text = bytes(self._build_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in text.rstrip().splitlines():
            self._build_log.appendPlainText(line)

    def _on_build_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._build_log.appendPlainText(f"The engine could not be started ({error.name}).")
            self._on_build_finished(-1, QProcess.ExitStatus.CrashExit)

    def _on_build_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        process, self._build_process = self._build_process, None
        if process is None:
            return
        self._on_build_output_from(process)
        process.deleteLater()
        if self._build_context is not None:
            self._build_context.cleanup()
            self._build_context = None
        self._build_btn.setEnabled(True)
        if exit_code == 0:
            self._build_log.appendPlainText("Build finished.")
            self.check()
        else:
            self._build_log.appendPlainText(f"Build failed (exit code {exit_code}).")
            self._runtime_label.setText("The image build failed — see the log below.")

    def _on_build_output_from(self, process: QProcess) -> None:
        text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in text.rstrip().splitlines():
            self._build_log.appendPlainText(line)

    def stop_build(self) -> None:
        """Kill a running build (the dialog is closing)."""
        if self._build_process is not None:
            self._build_process.kill()
            self._build_process.waitForFinished(2000)
