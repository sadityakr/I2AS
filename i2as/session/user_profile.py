"""Per-user settings — each person's notebook accounts, assistant and sessions.

**What belongs to a person, not to the machine.** The machine-wide
``settings.json`` (``i2as.session.app_config``) says how THIS installation
behaves: the gateway, the analysis container, the publish retry timings. What
a PERSON chooses lives in their own profile, one file per roster user::

    <user config dir>/users/<user_id>/profile.yaml

* ``eln``: whether they publish, their notebook accounts (which connector,
  and its non-secret settings — server URL, TLS, timeout), which account and
  profile a new experiment uses, and the template a new page is made from;
* ``assistant``: the drafting model, its token cap and price table;
* ``sessions``: their active and recent session folders.

Switching the logged-in user switches all of it. No profile ever holds a
secret: API keys are in the credential store (``i2as.session.credentials``),
keyed by account and user.

**Tolerant, like every settings file here.** A missing profile is the normal
first-launch case; a malformed one logs a WARNING and yields the defaults.
``UserProfileStore.load()`` also migrates, once, the older machine-wide
``eln-settings.json`` (its notebook account, its keys, its assistant block)
into the profile of the user who launches first.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from i2as.core.paths import user_config_dir
from i2as.session.app_config import legacy_eln_settings_path
from i2as.session.credentials import SCOPE_ASSISTANT, SCOPE_ELN, CredentialStore, credential_key
from i2as.session.drafting import AssistantSettings

logger = logging.getLogger(__name__)

#: Environment variable pointing at an explicit users folder (tests).
USERS_DIR_ENV_VAR = "I2AS_USERS_DIR"

_PROFILE_FILENAME = "profile.yaml"
_PLAIN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")


def users_dir() -> Path:
    """Return the folder holding one sub-folder per user."""
    override = os.environ.get(USERS_DIR_ENV_VAR)
    return Path(override) if override else user_config_dir() / "users"


def _str(value: object, default: str = "") -> str:
    return default if value is None else str(value)


def _dict(value: object) -> dict[str, Any]:
    return {str(k): v for k, v in value.items()} if isinstance(value, dict) else {}


@dataclass(frozen=True)
class ElnAccount:
    """One notebook account of one user.

    Attributes:
        account_id: A short id, unique for the user (``"lab"``).
        connector: The connector block's id (``"elabftw"``).
        settings: The connector's NON-secret settings (its
            ``settings_schema``): server URL, TLS policy, timeout.
        label: What the GUI shows.
    """

    account_id: str = "default"
    connector: str = "elabftw"
    settings: dict[str, Any] = field(default_factory=dict)
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.account_id, "connector": self.connector, "settings": dict(self.settings), "label": self.label}

    @classmethod
    def from_dict(cls, data: object) -> ElnAccount:
        payload = _dict(data)
        account_id = _str(payload.get("id"), "default")
        return cls(
            account_id=account_id if _PLAIN_ID.match(account_id) else "default",
            connector=_str(payload.get("connector"), "elabftw"),
            settings=_dict(payload.get("settings")),
            label=_str(payload.get("label")),
        )


@dataclass(frozen=True)
class ElnUserSettings:
    """One user's notebook preferences.

    Attributes:
        enabled: Publishing is switched on for this user.
        accounts: Their notebook accounts.
        default_account: The account a new experiment is linked through.
        default_profile: The profile a new experiment uses.
        default_template: The template a new page is created from when the
            profile names none.
    """

    enabled: bool = False
    accounts: tuple[ElnAccount, ...] = ()
    default_account: str = ""
    default_profile: str = "default"
    default_template: str = ""

    def account(self, account_id: str = "") -> ElnAccount | None:
        """Return one account (the default one for ``""``), or ``None``."""
        wanted = account_id or self.default_account
        for account in self.accounts:
            if account.account_id == wanted:
                return account
        return self.accounts[0] if self.accounts and not account_id else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "accounts": [a.to_dict() for a in self.accounts],
            "default_account": self.default_account,
            "default_profile": self.default_profile,
            "default_template": self.default_template,
        }

    @classmethod
    def from_dict(cls, data: object) -> ElnUserSettings:
        payload = _dict(data)
        raw = payload.get("accounts")
        accounts: list[ElnAccount] = []
        for item in raw if isinstance(raw, list) else []:
            account = ElnAccount.from_dict(item)
            if all(a.account_id != account.account_id for a in accounts):
                accounts.append(account)
        return cls(
            enabled=payload.get("enabled") is True,
            accounts=tuple(accounts),
            default_account=_str(payload.get("default_account")),
            default_profile=_str(payload.get("default_profile"), "default") or "default",
            default_template=_str(payload.get("default_template")),
        )


@dataclass(frozen=True)
class UserProfile:
    """Everything one person chooses.

    Attributes:
        eln: Their notebook preferences.
        assistant: Their drafting assistant (never with its key).
        sessions: ``{"active": folder, "recent": [folders]}``.
    """

    eln: ElnUserSettings = field(default_factory=ElnUserSettings)
    assistant: AssistantSettings = field(default_factory=AssistantSettings)
    sessions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        assistant = self.assistant.to_dict()
        assistant.pop("api_key", None)
        return {"eln": self.eln.to_dict(), "assistant": assistant, "sessions": dict(self.sessions)}

    @classmethod
    def from_dict(cls, data: object) -> UserProfile:
        payload = _dict(data)
        assistant = _dict(payload.get("assistant"))
        assistant.pop("api_key", None)
        return cls(
            eln=ElnUserSettings.from_dict(payload.get("eln")),
            assistant=AssistantSettings.from_dict(assistant),
            sessions=_dict(payload.get("sessions")),
        )


def _plain(value: Any) -> Any:
    """Convert ruamel containers to plain Python."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


class UserProfileStore:
    """Reads and writes each user's ``profile.yaml``; tells listeners of a save.

    Args:
        root: The users folder; ``None`` for ``users_dir()``.
        credentials: The credential store a legacy migration moves keys into.
        legacy_eln_path: The old ``eln-settings.json``; ``None`` for its usual
            place in the user config directory.
    """

    def __init__(
        self,
        root: Path | None = None,
        credentials: CredentialStore | None = None,
        legacy_eln_path: Path | None = None,
    ) -> None:
        self._root = Path(root) if root is not None else users_dir()
        self._credentials = credentials
        self._legacy = legacy_eln_path if legacy_eln_path is not None else legacy_eln_settings_path()
        self._listeners: list[Callable[[str, UserProfile], None]] = []

    @property
    def root(self) -> Path:
        return self._root

    def path(self, user_id: str) -> Path:
        """Return one user's profile file.

        Raises:
            ValueError: ``user_id`` is not a plain roster id.
        """
        if not _PLAIN_ID.match(user_id or ""):
            raise ValueError(f"not a user id: {user_id!r}")
        return self._root / user_id / _PROFILE_FILENAME

    def load(self, user_id: str) -> UserProfile:
        """Return one user's profile (defaults when absent or unreadable). Never raises."""
        try:
            path = self.path(user_id)
        except ValueError:
            return UserProfile()
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            migrated = self._migrate_legacy(user_id)
            return migrated if migrated is not None else UserProfile()
        except OSError as exc:
            logger.warning("Could not read the profile of %s: %s", user_id, exc)
            return UserProfile()
        try:
            data = _plain(YAML(typ="safe").load(io.StringIO(text)) or {})
        except YAMLError as exc:
            logger.warning("Malformed profile %s: %s", path, exc)
            return UserProfile()
        return UserProfile.from_dict(data)

    def save(self, user_id: str, profile: UserProfile) -> Path:
        """Write one user's profile atomically and tell every listener.

        Raises:
            ValueError: ``user_id`` is not a plain id.
            OSError: The file could not be written.
        """
        path = self.path(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.StringIO()
        yaml = YAML(typ="safe")
        yaml.default_flow_style = False
        yaml.dump(profile.to_dict(), buffer)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text("# I2AS user profile — no secrets here (keys are in the system keyring).\n" + buffer.getvalue(), encoding="utf-8")
        os.replace(temporary, path)
        for listener in list(self._listeners):
            try:
                listener(user_id, profile)
            except Exception:  # noqa: BLE001 - a listener must not break a save
                logger.exception("A profile listener failed")
        return path

    def update(self, user_id: str, change: Callable[[UserProfile], UserProfile]) -> UserProfile:
        """Load, change and save one profile in one step.

        Returns:
            The saved profile.
        """
        profile = change(self.load(user_id))
        self.save(user_id, profile)
        return profile

    def subscribe(self, listener: Callable[[str, UserProfile], None]) -> None:
        """Call ``listener(user_id, profile)`` after every save."""
        self._listeners.append(listener)

    def session_registry(self, user_id: str) -> tuple[Callable[[], dict[str, object]], Callable[[dict[str, object]], None]]:
        """Return ``(read, write)`` for ``SessionStore``: this user's session list.

        A user with no list yet starts from the machine's ``sessions.json``
        (the list before sessions became per user).
        """

        def _read() -> dict[str, object]:
            sessions = self.load(user_id).sessions
            return dict(sessions) if sessions else {}

        def _write(data: dict[str, object]) -> None:
            self.update(user_id, lambda p: replace(p, sessions={"active": data.get("active", ""), "recent": list(data.get("recent") or [])}))

        return _read, _write

    def _migrate_legacy(self, user_id: str) -> UserProfile | None:
        """Move the old machine-wide ``eln-settings.json`` into this user's profile, once."""
        try:
            data = json.loads(self._legacy.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        eln = ElnUserSettings()
        base_url = _str(data.get("base_url"))
        if base_url:
            account = ElnAccount(
                account_id="lab",
                connector=_str(data.get("backend"), "elabftw") or "elabftw",
                settings={
                    "base_url": base_url,
                    "verify_tls": data.get("verify_tls") is not False,
                    "timeout_s": data.get("timeout_s", 15.0),
                },
                label="Lab notebook",
            )
            eln = ElnUserSettings(
                enabled=data.get("enabled") is True,
                accounts=(account,),
                default_account="lab",
                default_template=_str(data.get("template_id")),
            )
        assistant_block = _dict(data.get("assistant"))
        profile = UserProfile(eln=eln, assistant=AssistantSettings.from_dict({k: v for k, v in assistant_block.items() if k != "api_key"}))
        if self._credentials is not None:
            try:
                if _str(data.get("api_key")) and base_url:
                    self._credentials.set(credential_key(SCOPE_ELN, "lab", user_id), _str(data.get("api_key")))
                if _str(assistant_block.get("api_key")):
                    self._credentials.set(credential_key(SCOPE_ASSISTANT, "default", user_id), _str(assistant_block.get("api_key")))
            except Exception as exc:  # noqa: BLE001 - a key that did not move is re-entered, not fatal
                logger.warning("Could not move the old notebook keys into the credential store: %s", exc)
        try:
            self.save(user_id, profile)
            migrated = self._legacy.with_name(self._legacy.name + ".migrated")
            os.replace(self._legacy, migrated)
            logger.info("Moved %s into the profile of %s (the old file is kept as %s)", self._legacy, user_id, migrated.name)
        except OSError as exc:
            logger.warning("Could not save the migrated profile of %s: %s", user_id, exc)
        return profile
