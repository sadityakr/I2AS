import pytest
import logging

@pytest.fixture(autouse=True)
def configure_logging():
    """Ensure logging is configured for all tests."""
    logging.basicConfig(level=logging.DEBUG)


@pytest.fixture(autouse=True)
def isolated_measurement_root(tmp_path, monkeypatch):
    """Point i2as.core.paths.measurement_root() at a throwaway directory.

    ExperimentInfoPanel falls back to measurement_root() whenever no
    experiment is open and the Data Dir field is empty — including at
    MonitorWindow construction (apply_session()) — so this must be isolated
    globally, the same way the per-file isolated_settings fixtures isolate
    QSettings, or a pytest run would either raise (no
    I2AS_MEASUREMENT_ROOT/App-config.yaml configured on the test
    machine) or read the real machine-level settings file.
    """
    monkeypatch.setenv("I2AS_MEASUREMENT_ROOT", str(tmp_path / "measurement_root"))
