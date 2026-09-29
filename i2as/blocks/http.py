"""The HTTP seam a connector talks through — and the one place failures are classified.

Every byte a connector sends goes through one ``ElnHttpTransport.request()``
call, so a connector is tested against canned responses (no server, no
network), and the production ``UrllibTransport`` is the standard library
alone. ``raise_for_status()`` turns an HTTP status into the ``ElnError``
subclass the framework acts on, so every connector classifies failures the
same way: 401/403 stop and ask the user to sign in again, 404 and 4xx stop and
show the reason, 429/5xx and transport failures are retried.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from i2as.blocks.connector import (
    ElnAuthError,
    ElnError,
    ElnNotFound,
    ElnTransientError,
    ElnValidationError,
)

logger = logging.getLogger(__name__)

#: How much of an error body is quoted in a message.
MAX_ERROR_BODY_CHARS = 300


@dataclass(frozen=True)
class HttpResponse:
    """One HTTP response, reduced to what a connector needs.

    Attributes:
        status: The HTTP status code.
        headers: Response headers, lower-cased keys.
        body: The raw body.
    """

    status: int = 0
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""

    def json(self) -> Any:
        """Parse the body as JSON (``None`` for an empty body).

        Raises:
            ElnValidationError: The body is not JSON.
        """
        if not self.body:
            return None
        try:
            return json.loads(self.body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ElnValidationError(f"the notebook returned a non-JSON body: {exc}") from exc

    def detail(self) -> str:
        """Return a short, safe rendering of the body for an error message."""
        text = self.body.decode("utf-8", errors="replace").strip()
        return text[:MAX_ERROR_BODY_CHARS] if text else "(no body)"


@runtime_checkable
class ElnHttpTransport(Protocol):
    """The one seam between a connector and the network."""

    def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
    ) -> HttpResponse:
        """Perform one request; a non-2xx status is data, not an exception.

        Raises:
            ElnTransientError: Only when no response could be obtained at all.
        """
        ...


class UrllibTransport:
    """The production transport: ``urllib`` with an explicit TLS policy."""

    def __init__(self, verify_tls: bool = True) -> None:
        """Build a transport.

        Args:
            verify_tls: Verify the server certificate. ``False`` exists only
                for a lab instance with a self-signed certificate and logs a
                WARNING.
        """
        context = ssl.create_default_context()
        if not verify_tls:
            logger.warning(
                "Notebook TLS verification is DISABLED: the connection is encrypted "
                "but the server is not authenticated"
            )
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
        """Perform one request, mapping every transport failure to ``ElnTransientError``."""
        request = urllib.request.Request(url, data=body, method=method)
        for name, value in headers.items():
            request.add_header(name, value)
        try:
            with urllib.request.urlopen(  # noqa: S310 - the scheme comes from the user's settings
                request, timeout=timeout_s, context=self._context
            ) as response:
                return HttpResponse(
                    status=int(response.status),
                    headers={k.lower(): v for k, v in response.headers.items()},
                    body=response.read(),
                )
        except urllib.error.HTTPError as exc:
            return HttpResponse(
                status=int(exc.code),
                headers={k.lower(): v for k, v in (exc.headers or {}).items()},
                body=exc.read(),
            )
        except (urllib.error.URLError, ssl.SSLError, OSError) as exc:
            raise ElnTransientError(f"{method} {url} failed: {exc}") from exc


def raise_for_status(method: str, path: str, response: HttpResponse) -> HttpResponse:
    """Return a 2xx response; raise the matching ``ElnError`` subclass otherwise.

    The message names the method, the path and the status — never a header,
    so never the credential.

    Args:
        method: The HTTP verb.
        path: The request path (without the host).
        response: The response.

    Returns:
        The response, when its status is 2xx.

    Raises:
        ElnAuthError: 401 or 403.
        ElnNotFound: 404.
        ElnTransientError: 408, 429 or 5xx.
        ElnValidationError: Any other 4xx.
        ElnError: Anything else.
    """
    status = response.status
    if 200 <= status < 300:
        return response
    message = f"{method} {path} refused with HTTP {status}: {response.detail()}"
    if status in (401, 403):
        raise ElnAuthError(message)
    if status == 404:
        raise ElnNotFound(message)
    if status in (408, 429) or status >= 500:
        raise ElnTransientError(message)
    if 400 <= status < 500:
        raise ElnValidationError(message)
    raise ElnError(message)


def multipart_file(path: Path, fields: Mapping[str, str] | None = None) -> tuple[bytes, str]:
    """Encode one file (plus text fields) as ``multipart/form-data``.

    Args:
        path: The file.
        fields: Extra text fields (``{"comment": "..."}``).

    Returns:
        ``(body, content_type)``.

    Raises:
        ElnValidationError: The file cannot be read.
    """
    try:
        content = Path(path).read_bytes()
    except OSError as exc:
        raise ElnValidationError(f"cannot read {Path(path).name}: {exc}") from exc
    boundary = f"----I2AS{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(Path(path).name)[0] or "application/octet-stream"
    parts: list[bytes] = []
    for name, value in (fields or {}).items():
        parts += [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            str(value).encode("utf-8") + b"\r\n",
        ]
    parts += [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{Path(path).name}"\r\n'.encode(),
        f"Content-Type: {mime}\r\n\r\n".encode(),
        content,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
