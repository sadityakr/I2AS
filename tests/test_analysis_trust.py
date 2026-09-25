"""What the application trusts about the analysis stage's output, and what it does not.

The analysis worker runs code the application does not trust, so every file
name a report CLAIMS is checked before the application follows it
(``report.output_file``), every report file is read within a size limit
(``report.read_report_file``), and every experiment id that reaches the
filesystem is a single plain folder name (``store.is_plain_name``). And the
gateway every agent connects to is handed the collaborators that make the
analysis tools work at all (``main.gateway_tool_context``).
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from i2as.analysis.report import (
    MAX_REPORT_BYTES,
    AnalysisReport,
    FigureRef,
    output_file,
    read_report_file,
)
from i2as.session.store import ExperimentStore, is_plain_name


@pytest.fixture
def report_dir(tmp_path) -> Path:
    folder = tmp_path / "analysis" / "run-0001"
    folder.mkdir(parents=True)
    (folder / "fit.png").write_bytes(b"\x89PNG")
    return folder


@pytest.mark.parametrize(
    "name",
    [
        "fit.png",
        "FIT.PNG",
    ],
)
def test_a_plain_png_in_the_report_folder_is_honoured(report_dir, name):
    if name != "fit.png":
        (report_dir / name).write_bytes(b"\x89PNG")
    assert output_file(report_dir, name) == (report_dir / name).resolve()


def test_every_other_figure_claim_is_refused(report_dir, tmp_path):
    secret = tmp_path / "gateway.json"
    secret.write_text("{}", encoding="utf-8")
    (report_dir / "notes.txt").write_text("x", encoding="utf-8")
    (report_dir / ".hidden.png").write_bytes(b"\x89PNG")
    (report_dir / "big.png").write_bytes(b"0" * 64)

    for claim in (
        str(secret),
        "../../gateway.json",
        "sub/fit.png",
        "notes.txt",
        ".hidden.png",
        "missing.png",
        "",
    ):
        assert output_file(report_dir, claim) is None, claim
    assert output_file(report_dir, "big.png", max_bytes=10) is None


def test_a_report_file_is_read_within_its_limit(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(AnalysisReport(run_id="r").to_dict()), encoding="utf-8")
    assert read_report_file(path).run_id == "r"

    path.write_text(" " * (MAX_REPORT_BYTES + 1), encoding="utf-8")
    assert read_report_file(path) is None
    path.write_text("{not json", encoding="utf-8")
    assert read_report_file(path) is None
    assert read_report_file(tmp_path / "absent.json") is None


def test_the_publisher_attaches_only_safe_figures(report_dir, tmp_path):
    from i2as.session.eln.publisher import ElnPublisher

    secret = tmp_path / "eln-settings.json"
    secret.write_text("{}", encoding="utf-8")
    report = AnalysisReport(
        figures=(FigureRef(file="fit.png", caption="fit"), FigureRef(file=str(secret))),
    )

    attachments = ElnPublisher._figure_attachments(report, report_dir)

    assert attachments == [{"path": str((report_dir / "fit.png").resolve()), "comment": "fit"}]


@pytest.mark.parametrize(
    ("name", "plain"),
    [
        ("001_mnsi_field_sweeps", True),
        ("20260101_sample A", True),
        ("..", False),
        (".hidden", False),
        ("a/b", False),
        ("a\\b", False),
        ("C:evil", False),
        ("/etc", False),
        ("", False),
        (None, False),
    ],
)
def test_only_plain_names_are_experiment_ids(name, plain):
    assert is_plain_name(name) is plain


def test_the_store_never_leaves_its_root(tmp_path):
    store = ExperimentStore(tmp_path / "session")

    assert store.load("../other") is None
    with pytest.raises(ValueError):
        store.data_dir("../other")
    with pytest.raises(ValueError):
        store.analysis_dir("/etc")
    assert store.data_dir("001_x") == tmp_path / "session" / "001_x" / "data"


def test_the_gateway_is_handed_the_analysis_collaborators():
    """Every agent connection gets the runner and publisher the GUI uses."""
    from i2as.main import gateway_tool_context

    manager, runner, publisher = object(), types.SimpleNamespace(), types.SimpleNamespace()

    context = gateway_tool_context(manager, {"X": object}, publisher, runner)

    assert context.experiments is manager
    assert context.analysis_runner is runner
    assert context.publisher is publisher
    assert set(context.run_catalog) == {"X"}
