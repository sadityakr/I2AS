"""**Access keys** for the HTTP MCP endpoint: a named identity per key.

The local-socket **Gateway server** admits a client with a per-launch token
that the descriptor file publishes: whoever can read that file is on this
machine and logged in as this user, and that is the whole test. An HTTP
endpoint reached through a tunnel has no such test — the request comes from
a service on the internet — so the credential itself has to say who is
calling. A key here is therefore not a password for the endpoint; it IS the
agent: it carries the actor id stamped on every verdict, run record and
feed entry the connection causes, and the **Role** the connection is granted
at its ``hello``.

**Only the digest is kept.** A key is shown once, at creation, and the store
keeps its SHA-256. A copy of the store therefore admits nobody, and a leaked
key is revoked by name rather than by rotating every other one.

**Stdlib only.** This module runs inside the app but is imported by the HTTP
transport in ``i2as.mcp``, which import contract C21 keeps free of the
session layer and the engine. It knows nothing about a role ceiling: the
controller that creates keys checks that (``role_within_ceiling``), and the
socket server checks it again at the handshake, so a key that outranks the
setup is refused twice rather than trusted once.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["KEY_PREFIX", "AccessKey", "KeyStore", "digest_key"]

#: Every key starts with this so a pasted secret is recognisable as one of
#: ours in a client's config, the way ``sk-`` marks an API key.
KEY_PREFIX = "i2as_"

#: The number of random bytes behind a key; 32 bytes is 256 bits.
_SECRET_BYTES = 32

#: The file's schema, stamped in so a later layout is told apart from a
#: corrupt one.
_SCHEMA = 1


def digest_key(plaintext: str) -> str:
    """Return the hex SHA-256 the store keeps for one key.

    Args:
        plaintext: The key as handed to the user.

    Returns:
        The lowercase hex digest.
    """
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AccessKey:
    """One key's identity — everything but the secret.

    Attributes:
        name: The label the operator chose; unique within the store and the
            handle a revoke uses.
        actor_id: The identity every connection under this key declares.
        role: The **Role** value string every connection under this key
            asks for.
        digest: The SHA-256 hex of the secret.
        created: When the key was made, ISO-8601 UTC.
        hint: The first characters of the secret, enough to match a key in
            a client's config against the list and no more.
    """

    name: str
    actor_id: str
    role: str
    digest: str
    created: str
    hint: str

    def to_json(self) -> dict[str, Any]:
        """Return the JSON-safe record the store writes."""
        return {
            "name": self.name,
            "actor_id": self.actor_id,
            "role": self.role,
            "digest": self.digest,
            "created": self.created,
            "hint": self.hint,
        }

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> AccessKey:
        """Rebuild one record.

        Args:
            record: What ``to_json()`` wrote.

        Returns:
            The key.

        Raises:
            KeyError: If a field is missing.
        """
        return cls(
            name=str(record["name"]),
            actor_id=str(record["actor_id"]),
            role=str(record["role"]),
            digest=str(record["digest"]),
            created=str(record["created"]),
            hint=str(record.get("hint", "")),
        )


class KeyStore:
    """The keys this installation has issued, in one JSON file.

    Every method is safe to call from any thread: the HTTP transport
    verifies keys on its own thread while the GUI creates and revokes them
    on its own.

    Attributes:
        path: The file the keys live in.
    """

    def __init__(self, path: Path | str) -> None:
        """Open (or, on first use, prepare to create) the store.

        Args:
            path: The JSON file. It need not exist yet; the first ``create``
                writes it with owner-only permissions.
        """
        self.path = Path(path)
        self._lock = threading.Lock()
        self._keys: dict[str, AccessKey] = {}
        self._load()

    # ── Reading ───────────────────────────────────────────────────────

    def keys(self) -> list[AccessKey]:
        """Return every key, oldest first.

        Returns:
            The keys, without their secrets.
        """
        with self._lock:
            return sorted(self._keys.values(), key=lambda key: key.created)

    def get(self, name: str) -> AccessKey | None:
        """Return the key called *name*, if there is one.

        Args:
            name: The label.

        Returns:
            The key, or ``None``.
        """
        with self._lock:
            return self._keys.get(name)

    def verify(self, plaintext: str) -> AccessKey | None:
        """Find the key a presented secret belongs to.

        Args:
            plaintext: The secret a request carried.

        Returns:
            The matching key, or ``None`` when no key matches. Digests are
            compared in constant time.
        """
        if not isinstance(plaintext, str) or not plaintext:
            return None
        presented = digest_key(plaintext)
        with self._lock:
            for key in self._keys.values():
                if hmac.compare_digest(key.digest, presented):
                    return key
        return None

    # ── Writing ───────────────────────────────────────────────────────

    def create(self, name: str, *, actor_id: str | None = None, role: str) -> tuple[AccessKey, str]:
        """Issue a new key.

        Args:
            name: The label; must be non-empty and not already in use.
            actor_id: The identity the key declares; defaults to *name*.
            role: The **Role** value string the key asks for. Not checked
                here — the caller that knows the ceiling checks it.

        Returns:
            The key record and the secret itself. The secret is returned
            exactly once and is not kept.

        Raises:
            ValueError: If *name* is blank or already taken.
        """
        label = (name or "").strip()
        if not label:
            raise ValueError("a key needs a name")
        secret = KEY_PREFIX + secrets.token_urlsafe(_SECRET_BYTES)
        key = AccessKey(
            name=label,
            actor_id=(actor_id or label).strip() or label,
            role=str(role),
            digest=digest_key(secret),
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            hint=secret[: len(KEY_PREFIX) + 4],
        )
        with self._lock:
            if label in self._keys:
                raise ValueError(f"a key called {label!r} already exists")
            self._keys[label] = key
            self._save()
        logger.info("Access key %r created for %r as %r", label, key.actor_id, key.role)
        return key, secret

    def revoke(self, name: str) -> bool:
        """Delete the key called *name*.

        Args:
            name: The label.

        Returns:
            ``True`` when a key was removed; ``False`` when there was none.
        """
        with self._lock:
            removed = self._keys.pop(name, None)
            if removed is not None:
                self._save()
        if removed is not None:
            logger.info("Access key %r revoked", name)
        return removed is not None

    # ── The file ──────────────────────────────────────────────────────

    def _load(self) -> None:
        """Read the file if it exists; an unreadable one is logged and treated as empty."""
        if not self.path.exists():
            return
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
            records = document.get("keys") if isinstance(document, dict) else None
            keys = [AccessKey.from_json(record) for record in (records or [])]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            logger.exception("Access keys in %s could not be read; treating as none", self.path)
            return
        self._keys = {key.name: key for key in keys}

    def _save(self) -> None:
        """Write the file with owner-only permissions. Caller holds the lock."""
        document = {"schema": _SCHEMA, "keys": [key.to_json() for key in self._keys.values()]}
        payload = json.dumps(document, indent=2).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
        os.replace(tmp, self.path)
