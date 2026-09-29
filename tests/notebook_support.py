"""Shared test support: a notebook service over the simulated notebook.

Builds what ``i2as.main`` builds for the notebook — a user profile with one
account, its key in a credential store, the service — but over the shipped
``sim`` connector (its pages in a JSON file the test can read back), with the
service running inline and blocks called in-process, so an end-to-end test is
deterministic and needs no network and no helper process.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from i2as.session.app_config import PublishingSettings
from i2as.session.credentials import CredentialStore, credential_key
from i2as.session.eln import BlockCatalog, ElnService, InProcessBlockRunner
from i2as.session.user_profile import ElnAccount, ElnUserSettings, UserProfile, UserProfileStore


def notebook_service(manager: Any, tmp_path: Path, user_id: str = "jdoe") -> tuple[ElnService, Callable[[], dict]]:
    """Return ``(service, read_notebook)`` for ``user_id`` publishing to the sim notebook.

    Args:
        manager: The ``ExperimentManager``.
        tmp_path: Where the profile, the key and the notebook's pages live.
        user_id: The experimenter.

    Returns:
        The service, and a callable returning the sim notebook's whole state
        (``{"entries": {id: {"title", "body", "uploads", "links", "fields"}}, ...}``).
    """
    creds = CredentialStore(fallback_path=tmp_path / "credentials.json", environ={}, detect=False)
    profiles = UserProfileStore(tmp_path / "profiles", creds, tmp_path / "no-legacy.json")
    state = tmp_path / "sim-notebook.json"
    profiles.save(
        user_id,
        UserProfile(
            eln=ElnUserSettings(
                enabled=True,
                accounts=(ElnAccount(account_id="lab", connector="sim", settings={"state_file": str(state)}),),
                default_account="lab",
            )
        ),
    )
    creds.set(credential_key("eln", "lab", user_id), "sim-key")
    service = ElnService(
        manager,
        profiles,
        creds,
        publishing=PublishingSettings,
        catalog=BlockCatalog(tmp_path / "blocks"),
        runner_factory=InProcessBlockRunner,
        synchronous=True,
    )
    return service, (lambda: json.loads(state.read_text(encoding="utf-8")))
