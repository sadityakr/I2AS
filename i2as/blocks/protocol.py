"""The block protocol — how the application calls a block across a process boundary.

One request per line of JSON on the helper's stdin, one answer per line on its
stdout::

    {"id": 1, "op": "load", "kind": "connector", "path": "...", "settings": {...}, "credential": "..."}
    {"id": 2, "op": "call", "method": "search", "args": {"query": {...}}}
    -> {"id": 2, "ok": true, "result": [...]}
    -> {"id": 2, "ok": false, "error": {"type": "ElnAuthError", "message": "..."}}

Lines are ASCII (``json.dumps(..., ensure_ascii=True)``), so no console or
pipe encoding can corrupt a character on the way. ``dispatch()`` is the ONE translation between JSON arguments and a block's
typed methods, used by the helper process and by the in-process runner the
tests use, so both speak exactly the same protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from i2as.blocks.connector import (
    ERROR_TYPES,
    ElnConnector,
    ElnEntryRef,
    ElnError,
    ElnQuery,
    ElnRef,
)
from i2as.blocks.renderer import RenderContext, Section

#: Methods a renderer block answers.
RENDERER_METHODS: tuple[str, ...] = ("render",)


class BlockProtocolError(RuntimeError):
    """A request the protocol cannot answer (unknown method, bad arguments)."""


def _str(value: object) -> str:
    """Coerce to ``str``."""
    return "" if value is None else str(value)


def _fields(value: object) -> dict[str, str]:
    """Coerce to ``{str: str}``."""
    return {str(k): _str(v) for k, v in value.items()} if isinstance(value, dict) else {}


def dispatch_connector(connector: ElnConnector, method: str, args: dict[str, Any]) -> Any:
    """Call one connector method from JSON arguments; return a JSON-safe result.

    Raises:
        BlockProtocolError: Unknown method or unusable arguments.
        ElnError: Whatever the connector raised.
    """
    entry = ElnEntryRef.from_dict(args.get("entry"))
    if method == "verify":
        return connector.verify().to_dict()
    if method == "list_templates":
        return [t.to_dict() for t in connector.list_templates()]
    if method == "search":
        return [h.to_dict() for h in connector.search(ElnQuery.from_dict(args.get("query")))]
    if method == "get_record":
        return connector.get_record(ElnRef.from_dict(args.get("ref"))).to_dict()
    if method == "create_entry":
        return connector.create_entry(
            _str(args.get("title")), _str(args.get("template_id")), _fields(args.get("fields"))
        ).to_dict()
    if method == "append_section":
        connector.append_section(entry, _str(args.get("publish_id")), _str(args.get("html")))
        return None
    if method == "has_section":
        return bool(connector.has_section(entry, _str(args.get("publish_id"))))
    if method == "set_fields":
        connector.set_fields(entry, _fields(args.get("fields")))
        return None
    if method == "upload":
        return _str(connector.upload(entry, Path(_str(args.get("path"))), _str(args.get("caption"))))
    if method == "link_item":
        connector.link_item(entry, ElnRef.from_dict(args.get("item")))
        return None
    raise BlockProtocolError(f"a connector has no method {method!r}")


def dispatch_renderer(render: Any, method: str, args: dict[str, Any]) -> Any:
    """Call a renderer from JSON arguments; return the section as JSON.

    Raises:
        BlockProtocolError: Unknown method.
    """
    if method != "render":
        raise BlockProtocolError(f"a renderer has no method {method!r}")
    produced = render(RenderContext.from_dict(args.get("context")))
    payload = produced.to_dict() if isinstance(produced, Section) else produced
    return Section.from_dict(payload).to_dict()


def error_payload(exc: BaseException) -> dict[str, str]:
    """Return the protocol's error object for one exception."""
    if isinstance(exc, ElnError):
        # The most specific contract class the exception is (a connector may
        # subclass ElnAuthError; the caller must still see an auth failure).
        kind = next(
            (c.__name__ for c in type(exc).__mro__ if ERROR_TYPES.get(c.__name__) is c),
            "ElnError",
        )
    else:
        kind = "BlockError"
    message = str(exc) or type(exc).__name__
    return {"type": kind, "message": message[:2000]}


class BlockError(RuntimeError):
    """A block failed for a reason that is not an ``ElnError`` (a bug, a crash, a timeout)."""


def raise_error(payload: object) -> None:
    """Re-raise a protocol error object in the caller's process.

    Raises:
        ElnError: (or a subclass) for a notebook failure.
        BlockError: For anything else.
    """
    data = payload if isinstance(payload, dict) else {}
    kind = _str(data.get("type"))
    message = _str(data.get("message")) or "the block failed"
    cls = ERROR_TYPES.get(kind)
    if cls is not None:
        raise cls(message)
    raise BlockError(message)
