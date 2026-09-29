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


@pytest.fixture(autouse=True)
def isolated_app_config(tmp_path, monkeypatch):
    """Point the general settings file (and the legacy ELN file) at a throwaway path.

    ``i2as.gui.app_settings.config_store()`` is a process-wide singleton over
    ``settings.json``, and it migrates from ``eln-settings.json`` on first
    use; both must be isolated for every test, or a pytest run would read —
    and on migration WRITE — the user's real settings. The singleton is reset
    so each test builds its own store over its own file.
    """
    monkeypatch.setenv("I2AS_SETTINGS", str(tmp_path / "app_config" / "settings.json"))
    monkeypatch.setenv("I2AS_ELN_SETTINGS", str(tmp_path / "app_config" / "eln-settings.json"))
    monkeypatch.setattr("i2as.gui.app_settings._CONFIG_STORE", None)
