"""The eLabFTW backend — the first concrete ``ElnAdapter``.

Speaks eLabFTW's REST API v2 over the endpoints I2AS actually needs:

===========================================  ==================================
``GET  /api/v2/users/me``                    health check + account identity
``GET  /api/v2/experiments_templates``       the lab's own entry templates
``POST /api/v2/experiments``                 create an entry (from a template)
``GET/PATCH /api/v2/experiments/{id}``       read and rewrite title/body/metadata
``POST /api/v2/experiments/{id}/uploads``    attach a file (multipart)
===========================================  ==================================

**Token auth**: every request carries the user's API key in an
``Authorization`` header. The key comes from the user-level settings file or
``I2AS_ELAB_APIKEY`` and is never logged — not in a debug line, not in an
error message; a failure reports the method, the path, and the status code.

**TLS is verified by default.** ``verify_tls: false`` in the settings file
turns verification off for a lab instance with a self-signed certificate, and
says so with a WARNING every time an adapter is built that way.

**One injectable transport.** Every byte in and out goes through a single
``ElnHttpTransport.request()`` call, so the whole backend is testable against
canned responses — no live server in CI, no network in a test — while the
adapter surface stays the plain contract. This is the same split as the driver
layer's: the adapter is the instrument, the transport is the bus.

There is no eLabFTW endpoint for "attach a bare URL", so ``attach_link``
appends a paragraph to the entry body (read, append, rewrite) — the entry
still says exactly where the data lives when the bytes are too large to
upload or never left the measurement machine.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import ssl
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from html import escape
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from i2as.session.eln.adapter import (
    ElnAdapter,
    ElnCapabilities,
    ElnEntryRef,
    ElnError,
    ElnTemplate,
)
from i2as.session.eln.settings import ElnSettings

logger = logging.getLogger(__name__)

_API_ROOT = "/api/v2"

_MAX_ERROR_BODY_CHARS = 300


@dataclass(frozen=True)
class HttpResponse:
    """One HTTP response, reduced to what an adapter needs.

    Attributes:
        status: The HTTP status code.
        headers: Response headers, lowercased keys.
        body: The raw response body.
    """

    status: int = 0
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""

    def json(self) -> Any:
        """Parse the body as JSON.

        Returns:
            The parsed value, or ``None`` for an empty body.

        Raises:
            ElnError: If the body is not valid JSON.
        """
        if not self.body:
            return None
        try:
            return json.loads(self.body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ElnError(f"backend returned a non-JSON body: {exc}") from exc


@runtime_checkable
class ElnHttpTransport(Protocol):
    """The one seam between an HTTP backend adapter and the network.

    A test supplies a fake implementing this single method and drives every
    branch of a backend — success, 4xx, 5xx, timeout, garbage body — without a
    server. A production adapter is handed ``UrllibTransport``.
    """

    def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
    ) -> HttpResponse:
        """Perform one HTTP request.

        Args:
            method: The HTTP verb.
            url: The absolute URL.
            headers: Request headers, including authorization.
            body: The request body, or ``None``.
            timeout_s: Per-request timeout in seconds.

        Returns:
            The response, whatever its status — a non-2xx status is data, not
            an exception; the adapter decides what it means.

        Raises:
            ElnError: Only when no response could be obtained at all
                (unreachable host, TLS failure, timeout).
        """
        ...


class UrllibTransport:
    """The production transport: stdlib ``urllib`` with an explicit TLS policy.

    Deliberately dependency-free — the application ships no HTTP client
    library, and one publish every few seconds does not need one.
    """

    def __init__(self, verify_tls: bool = True) -> None:
        """Build a transport with a fixed TLS policy.

        Args:
            verify_tls: Verify the server certificate. ``True`` is the only
                value that should ever reach production; ``False`` exists for
                a lab instance with a self-signed certificate and logs a
                WARNING.
        """
        self._verify_tls = verify_tls
        if verify_tls:
            self._context: ssl.SSLContext | None = ssl.create_default_context()
        else:
            logger.warning(
                "ELN TLS certificate verification is DISABLED — the connection "
                "to the notebook is encrypted but unauthenticated"
            )
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            self._context = context

    def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
    ) -> HttpResponse:
        """Perform one request, mapping every transport failure to ``ElnError``.

        Args:
            method: The HTTP verb.
            url: The absolute URL.
            headers: Request headers.
            body: The request body, or ``None``.
            timeout_s: Per-request timeout in seconds.

        Returns:
            The response, including non-2xx statuses.

        Raises:
            ElnError: The host is unreachable, TLS failed, or the request
                timed out.
        """
        request = urllib.request.Request(url, data=body, method=method)
        for name, value in headers.items():
            request.add_header(name, value)
        try:
            with urllib.request.urlopen(  # noqa: S310 - scheme is fixed by settings
                request, timeout=timeout_s, context=self._context
            ) as response:
                return HttpResponse(
                    status=int(response.status),
                    headers={k.lower(): v for k, v in response.headers.items()},
                    body=response.read(),
                )
        except urllib.error.HTTPError as exc:
            # A 4xx/5xx is a real response; the adapter turns it into a
            # message the operator can act on (wrong key, missing template).
            return HttpResponse(
                status=int(exc.code),
                headers={k.lower(): v for k, v in (exc.headers or {}).items()},
                body=exc.read(),
            )
        except (urllib.error.URLError, ssl.SSLError, OSError) as exc:
            raise ElnError(f"{method} {url} failed: {exc}") from exc


class ElabFtwAdapter(ElnAdapter):
    """An electronic lab notebook served by eLabFTW.

    Class attributes:
        backend: ``"elabftw"``.
        capabilities: eLabFTW offers templates, uploads, body updates, and —
            via an appended body paragraph — links. It declares no upload cap
            of its own; the server's own PHP limit surfaces as a 4xx, and the
            caller's cap in settings is what normally decides.
    """

    backend: ClassVar[str] = "elabftw"
    capabilities: ClassVar[ElnCapabilities] = ElnCapabilities(
        templates=True,
        attachments=True,
        links=True,
        update=True,
        max_attachment_bytes=0,
    )

    def __init__(
        self,
        settings: Mapping[str, Any] | ElnSettings,
        transport: ElnHttpTransport | None = None,
    ) -> None:
        """Build the adapter from a settings mapping.

        Args:
            settings: The ELN settings — a plain mapping (as read from the
                user-level settings file) or an ``ElnSettings``. Supplies the
                base URL, the API key, the TLS policy, and the timeout.
            transport: The HTTP seam. ``None`` builds a ``UrllibTransport``
                honouring ``verify_tls``; tests inject a fake.
        """
        self._settings = (
            settings
            if isinstance(settings, ElnSettings)
            else ElnSettings.from_dict(dict(settings))
        )
        self._transport = (
            transport
            if transport is not None
            else UrllibTransport(verify_tls=self._settings.verify_tls)
        )

    # ------------------------------------------------------------------
    # Internals — the one place a request is built and a failure is read
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        """Return the absolute URL for one API path.

        Args:
            path: Path below the API root, e.g. ``"/experiments"``.

        Returns:
            The absolute URL.

        Raises:
            ElnError: If no base URL is configured.
        """
        base = self._settings.base_url.rstrip("/")
        if not base:
            raise ElnError("no ELN base_url is configured")
        return f"{base}{_API_ROOT}{path}"

    def _headers(self, content_type: str = "") -> dict[str, str]:
        """Return the request headers, including the never-logged API key.

        Args:
            content_type: Value for the ``Content-Type`` header, or ``""`` to
                omit it.

        Returns:
            The header mapping.

        Raises:
            ElnError: If no API key is configured.
        """
        if not self._settings.api_key:
            raise ElnError("no ELN api_key is configured")
        headers = {"Authorization": self._settings.api_key, "Accept": "application/json"}
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def _call(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        content_type: str = "",
    ) -> HttpResponse:
        """Issue one request and turn any non-2xx status into an ``ElnError``.

        Args:
            method: The HTTP verb.
            path: Path below the API root.
            body: The encoded request body, or ``None``.
            content_type: The body's content type, or ``""``.

        Returns:
            The successful response.

        Raises:
            ElnError: On any transport failure or non-2xx status. The message
                names the method, the path, and the status — never the key.
        """
        url = self._url(path)
        logger.debug("ELN %s %s", method, path)
        response = self._transport.request(
            method, url, self._headers(content_type), body, self._settings.timeout_s
        )
        if not 200 <= response.status < 300:
            raise ElnError(
                f"{method} {path} refused with HTTP {response.status}: "
                f"{self._detail(response)}"
            )
        return response

    @staticmethod
    def _detail(response: HttpResponse) -> str:
        """Return a short, safe rendering of an error response body.

        Args:
            response: The refused response.

        Returns:
            The body text, truncated — enough to tell a wrong key from a
            missing template without pasting a page of HTML into a log.
        """
        text = response.body.decode("utf-8", errors="replace").strip()
        return text[:_MAX_ERROR_BODY_CHARS] if text else "(no body)"

    @staticmethod
    def _entry_id_from(response: HttpResponse) -> str:
        """Extract the new entry's id from a create response.

        eLabFTW answers a create with ``201`` and a ``Location`` header naming
        the new resource; some deployments also echo the id in the body.

        Args:
            response: The create response.

        Returns:
            The entry id.

        Raises:
            ElnError: If neither the header nor the body names an id.
        """
        location = response.headers.get("location", "")
        candidate = location.rstrip("/").rsplit("/", 1)[-1] if location else ""
        if candidate.isdigit():
            return candidate
        payload = response.json()
        if isinstance(payload, dict) and payload.get("id") is not None:
            return str(payload["id"])
        raise ElnError(
            "backend created an entry but named no id (no usable Location header)"
        )

    def _entry_url(self, entry_id: str) -> str:
        """Return the human-facing URL of one entry.

        Args:
            entry_id: The entry's backend id.

        Returns:
            The URL a person opens in a browser.
        """
        return f"{self._settings.base_url.rstrip('/')}/experiments.php?mode=view&id={entry_id}"

    def _read_body(self, ref: ElnEntryRef) -> str:
        """Return an entry's current body HTML.

        Args:
            ref: The entry to read.

        Returns:
            The body, or ``""`` when the backend reports none.

        Raises:
            ElnError: The entry could not be read.
        """
        payload = self._call("GET", f"/experiments/{ref.entry_id}").json()
        if isinstance(payload, dict) and payload.get("body") is not None:
            return str(payload["body"])
        return ""

    @staticmethod
    def _multipart(path: Path, comment: str) -> tuple[bytes, str]:
        """Encode one file upload as ``multipart/form-data``.

        Hand-rolled because the application ships no HTTP client library and
        one upload per run does not justify adding one.

        Args:
            path: The file to upload.
            comment: Optional caption stored with the upload.

        Returns:
            ``(body, content_type)``.

        Raises:
            ElnError: The file cannot be read.
        """
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise ElnError(f"cannot read {path}: {exc}") from exc
        boundary = f"----I2AS{uuid.uuid4().hex}"
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        parts: list[bytes] = []
        if comment:
            parts += [
                f"--{boundary}\r\n".encode(),
                b'Content-Disposition: form-data; name="comment"\r\n\r\n',
                comment.encode("utf-8") + b"\r\n",
            ]
        parts += [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode(),
            f"Content-Type: {mime}\r\n\r\n".encode(),
            content,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
        return b"".join(parts), f"multipart/form-data; boundary={boundary}"

    # ------------------------------------------------------------------
    # The contract
    # ------------------------------------------------------------------

    def verify(self) -> str:
        """Check reachability and credentials against ``/users/me``.

        Returns:
            The authenticated account's name (and team, when the backend
            reports one).

        Raises:
            ElnError: The backend is unreachable or the key is rejected.
        """
        payload = self._call("GET", "/users/me").json()
        if not isinstance(payload, dict):
            raise ElnError("backend did not describe the authenticated account")
        name = str(payload.get("fullname") or payload.get("email") or "authenticated")
        team = payload.get("team_name") or payload.get("team") or ""
        return f"{name} ({team})" if team else name

    def list_templates(self) -> list[ElnTemplate]:
        """List the lab's own eLabFTW experiment templates.

        Returns:
            The available templates; empty when the backend has none.

        Raises:
            ElnError: The backend is unreachable or refuses the request.
        """
        payload = self._call("GET", "/experiments_templates").json()
        if not isinstance(payload, list):
            return []
        return [
            ElnTemplate(
                template_id=str(item.get("id", "")),
                name=str(item.get("title") or item.get("name") or ""),
            )
            for item in payload
            if isinstance(item, dict)
        ]

    def create_entry(
        self,
        title: str,
        template_id: str | None,
        body_html: str,
        tags: list[str],
        metadata: dict[str, Any],
    ) -> ElnEntryRef:
        """Create one experiment entry, then fill in its title, body, and tags.

        eLabFTW creates an entry (optionally from a template) and then accepts
        its content on a second call, so this is deliberately two requests: a
        create that can only fail before anything exists, and an update whose
        failure leaves an empty entry the outbox's retry can fill in without
        creating a second one.

        Args:
            title: The entry title.
            template_id: eLabFTW experiment-template id (from
                ``list_templates()``), or ``None`` for the backend default.
            body_html: The rendered entry body.
            tags: Tags to set on the entry.
            metadata: JSON-safe extra fields, stored as eLabFTW metadata.

        Returns:
            The new entry's reference.

        Raises:
            ElnError: The entry could not be created or filled in.
        """
        chosen = template_id or self._settings.template_id
        payload: dict[str, Any] = {"template": chosen} if chosen else {}
        created = self._call(
            "POST", "/experiments", json.dumps(payload).encode("utf-8"), "application/json"
        )
        entry_id = self._entry_id_from(created)
        ref = ElnEntryRef(
            backend=self.backend,
            entry_id=entry_id,
            url=self._entry_url(entry_id),
            template_id=str(chosen or ""),
        )
        update: dict[str, Any] = {
            "title": title,
            "body": body_html,
            "metadata": json.dumps(metadata),
        }
        if tags:
            update["tags"] = list(tags)
        self._call(
            "PATCH",
            f"/experiments/{entry_id}",
            json.dumps(update).encode("utf-8"),
            "application/json",
        )
        logger.info("Created eLabFTW entry %s", ref.url)
        return ref

    def update_entry(
        self, ref: ElnEntryRef, body_html: str, metadata: dict[str, Any]
    ) -> None:
        """Rewrite an existing entry's body and metadata.

        Args:
            ref: The entry to update.
            body_html: The new rendered body.
            metadata: JSON-safe extra fields.

        Raises:
            ElnError: The entry could not be updated.
        """
        self._call(
            "PATCH",
            f"/experiments/{ref.entry_id}",
            json.dumps({"body": body_html, "metadata": json.dumps(metadata)}).encode("utf-8"),
            "application/json",
        )

    def attach_file(self, ref: ElnEntryRef, path: Path, comment: str = "") -> None:
        """Upload one file to an existing entry.

        Args:
            ref: The entry to attach to.
            path: Local file to upload; read into memory in one piece, which
                is why the caller's ``max_attachment_bytes`` cap matters.
            comment: Optional caption stored with the upload.

        Raises:
            ElnError: The file is unreadable, or the upload was refused.
        """
        body, content_type = self._multipart(Path(path), comment)
        self._call("POST", f"/experiments/{ref.entry_id}/uploads", body, content_type)
        logger.info("Attached %s to eLabFTW entry %s", Path(path).name, ref.entry_id)

    def attach_link(self, ref: ElnEntryRef, url: str, comment: str = "") -> None:
        """Record a URL or file path by appending a paragraph to the entry body.

        eLabFTW has no "attach a bare URL" endpoint, so this reads the current
        body, appends one escaped paragraph, and writes it back — the entry
        still says exactly where the data lives when the bytes were too large
        to upload or never left the measurement machine.

        Args:
            ref: The entry to attach to.
            url: The URL or absolute path to record.
            comment: Optional caption shown with the link.

        Raises:
            ElnError: The entry could not be read or rewritten.
        """
        caption = f"{escape(comment)}: " if comment else ""
        paragraph = f"<p>{caption}<code>{escape(url)}</code></p>"
        self._call(
            "PATCH",
            f"/experiments/{ref.entry_id}",
            json.dumps({"body": self._read_body(ref) + paragraph}).encode("utf-8"),
            "application/json",
        )
