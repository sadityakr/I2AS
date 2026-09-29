"""Credentials — where a user's API keys live, and who may read them.

**The OS keyring first.** Each secret is stored in the platform's credential
store through ``keyring`` (Windows Credential Manager, macOS Keychain, the
Secret Service on Linux), under the service ``I2AS`` and a key naming its
scope, the account and the user: ``eln/<account_id>/<user_id>`` or
``assistant/default/<user_id>``. The same person's key never reaches another
user of the machine, and no settings file ever holds a secret.

**A file only when there is no keyring.** Without a usable keyring backend
the secret goes into ``credentials.json`` in the per-user config directory,
written atomically with its mode tightened, and a WARNING says so once. An
environment variable overrides both — the way to run without the key ever
touching a disk (``I2AS_ELN_APIKEY`` for every notebook account, the older
``I2AS_ELAB_APIKEY`` too, ``I2AS_ASSISTANT_APIKEY`` for the drafting model).

**Who may read one.** Only the code that needs it at the moment it needs it:
the notebook service hands a connector its key when it starts the connector's
helper process, and the drafting client reads the model key. A secret never
goes to an agent, a tool result, MCP, a log line, a renderer or the analysis
container; the Settings dialog only ever learns whether one is stored.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from i2as.core.paths import user_config_dir

logger = logging.getLogger(__name__)

#: The keyring service every I2AS secret is stored under.
KEYRING_SERVICE = "I2AS"

#: Scopes.
SCOPE_ELN = "eln"
SCOPE_ASSISTANT = "assistant"

#: Environment overrides, by scope; the first set one wins.
ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    SCOPE_ELN: ("I2AS_ELN_APIKEY", "I2AS_ELAB_APIKEY"),
    SCOPE_ASSISTANT: ("I2AS_ASSISTANT_APIKEY",),
}

_FALLBACK_FILENAME = "credentials.json"


def credential_key(scope: str, account_id: str, user_id: str) -> str:
    """Return the key one secret is stored under.

    Args:
        scope: ``eln`` or ``assistant``.
        account_id: The account (``"default"`` for the assistant).
        user_id: Whose secret it is.

    Returns:
        ``"<scope>/<account_id>/<user_id>"``.
    """
    return f"{scope}/{account_id or 'default'}/{user_id or 'guest'}"


def _default_keyring() -> Any | None:
    """Return the ``keyring`` module when it has a real backend, else ``None``."""
    try:
        import keyring
        from keyring.backends import fail
    except ImportError:
        return None
    try:
        backend = keyring.get_keyring()
    except Exception:  # noqa: BLE001 - a broken keyring is "no keyring"
        return None
    if isinstance(backend, fail.Keyring) or type(backend).__module__.endswith(".null"):
        return None
    return keyring


class CredentialStore:
    """Reads and writes secrets; never logs one.

    Args:
        keyring_module: The ``keyring`` module (or a test stand-in with
            ``get_password``/``set_password``/``delete_password``); ``None``
            detects the real one, and falls back to the file.
        fallback_path: The file used without a keyring; ``None`` for
            ``<user config>/credentials.json``.
        environ: The environment consulted for overrides (tests pass a dict).
    """

    def __init__(
        self,
        keyring_module: Any | None = None,
        fallback_path: Path | None = None,
        environ: dict[str, str] | None = None,
        *,
        detect: bool = True,
    ) -> None:
        self._keyring = keyring_module if keyring_module is not None else (_default_keyring() if detect else None)
        self._fallback = fallback_path or (user_config_dir() / _FALLBACK_FILENAME)
        self._environ = os.environ if environ is None else environ
        self._warned = False

    @property
    def backend_name(self) -> str:
        """Where secrets are kept, in words for the Settings dialog."""
        if self._keyring is None:
            return f"file ({self._fallback})"
        try:
            return type(self._keyring.get_keyring()).__name__
        except Exception:  # noqa: BLE001
            return "system keyring"

    def _scope_of(self, key: str) -> str:
        return key.split("/", 1)[0]

    def get(self, key: str) -> str:
        """Return one secret, or ``""`` when none is stored.

        Args:
            key: From ``credential_key()``.

        Returns:
            The environment override, else the stored secret, else ``""``.
        """
        for name in ENV_OVERRIDES.get(self._scope_of(key), ()):
            value = self._environ.get(name, "")
            if value:
                return value
        if self._keyring is not None:
            try:
                return self._keyring.get_password(KEYRING_SERVICE, key) or ""
            except Exception as exc:  # noqa: BLE001 - never raise into a caller for a read
                logger.warning("The keyring could not be read (%s)", type(exc).__name__)
                return ""
        return str(self._read_file().get(key, ""))

    def has(self, key: str) -> bool:
        """Whether a secret is available for ``key`` (stored or overridden)."""
        return bool(self.get(key))

    def set(self, key: str, secret: str) -> None:
        """Store one secret (``""`` deletes it).

        Raises:
            OSError: The fallback file could not be written.
        """
        if not secret:
            self.delete(key)
            return
        if self._keyring is not None:
            self._keyring.set_password(KEYRING_SERVICE, key, secret)
            logger.info("Stored a credential for %s in the system keyring", key)
            return
        self._warn_file()
        data = self._read_file()
        data[key] = secret
        self._write_file(data)
        logger.info("Stored a credential for %s in the fallback file", key)

    def delete(self, key: str) -> None:
        """Remove one secret; absent is fine."""
        if self._keyring is not None:
            try:
                self._keyring.delete_password(KEYRING_SERVICE, key)
            except Exception:  # noqa: BLE001 - "not there" is the goal
                pass
            return
        data = self._read_file()
        if data.pop(key, None) is not None:
            self._write_file(data)

    def _warn_file(self) -> None:
        if not self._warned:
            logger.warning(
                "No system keyring is available: API keys are kept in %s (owner-only)",
                self._fallback,
            )
            self._warned = True

    def _read_file(self) -> dict[str, str]:
        try:
            data = json.loads(self._fallback.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def _write_file(self, data: dict[str, str]) -> None:
        self._fallback.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._fallback.with_name(self._fallback.name + ".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, self._fallback)
