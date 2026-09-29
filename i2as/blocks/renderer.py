"""The renderer contract — how a publish's section is laid out (tier 2).

**The renderer standard.** A renderer is one Python module with one function::

    NAME = "my_layout"                   # the id a profile names it by
    DESCRIPTION = "One line."
    CONTRACT_VERSION = 1

    def render(context: RenderContext) -> Section: ...

It receives everything a section may show — the experiment, and for each run
being published its facts and its selected **analysis bundle** — and returns a
``Section``: a title, an ordered list of blocks (headings, text, Markdown,
tables, results, figures, key/value lists, a little HTML), tags, and the
page fields to set. It never touches the network, a credential, a file or an
experiment record: the framework runs it in a helper process with no
credential and the network switched off, turns its ``Section`` into safe HTML
(``i2as.blocks.markup``: every text escaped, Markdown limited, HTML
sanitised), uploads the figures it names, and appends the result to the page.
A renderer that raises, hangs or returns junk costs a failed publish that
says why — never a published page with broken content, never the station.

The shipped layout (``i2as/blocks/shipped/default_renderer.py``) is itself an
ordinary renderer; copy it to start your own:
``python -m i2as.blocks new-renderer <name> --dir <folder>``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from i2as.analysis.bundle import Bundle

#: The version of this contract.
RENDERER_CONTRACT_VERSION = 1

#: Caps applied to what a renderer returns, whatever it returns.
MAX_BLOCKS = 200
MAX_TEXT_CHARS = 20_000
MAX_TABLE_ROWS = 200
MAX_TABLE_COLUMNS = 20
MAX_FIELDS = 100
MAX_TAGS = 50

#: Block types, and the keys each carries.
BLOCK_HEADING = "heading"
BLOCK_PARAGRAPH = "paragraph"
BLOCK_MARKDOWN = "markdown"
BLOCK_HTML = "html"
BLOCK_TABLE = "table"
BLOCK_RESULTS = "results"
BLOCK_FIGURE = "figure"
BLOCK_KEY_VALUES = "key_values"
BLOCK_LINK = "link"
BLOCK_TYPES: tuple[str, ...] = (
    BLOCK_HEADING,
    BLOCK_PARAGRAPH,
    BLOCK_MARKDOWN,
    BLOCK_HTML,
    BLOCK_TABLE,
    BLOCK_RESULTS,
    BLOCK_FIGURE,
    BLOCK_KEY_VALUES,
    BLOCK_LINK,
)


def _text(value: object, limit: int = MAX_TEXT_CHARS) -> str:
    """Coerce to a bounded string."""
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cell(value: object) -> Any:
    """Return a JSON scalar for a table cell."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value if not isinstance(value, str) else _text(value, 2000)
    return _text(value, 2000)


def _dict(value: object) -> dict[str, Any]:
    """Return ``value`` if it is a dict, else ``{}``."""
    return {str(k): v for k, v in value.items()} if isinstance(value, dict) else {}


def _list(value: object) -> list[Any]:
    """Return ``value`` as a list (tuples accepted), else ``[]``."""
    return list(value) if isinstance(value, (list, tuple)) else []


# ----------------------------------------------------------------------
# What a renderer receives
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class RunContext:
    """One run being published.

    Attributes:
        run_id: The run.
        procedure: Its procedure.
        params: The parameters it ran with.
        status: ``done`` / ``failed`` / ``aborted``.
        reason: The failure reason, if any.
        started_utc: When it started.
        finished_utc: When it ended.
        data_file: Its data file (as recorded).
        legacy_entry_url: The per-run page an older I2AS created for it, or
            ``""`` — link it, so nothing already in the notebook is orphaned.
        bundle: Its selected analysis bundle, or ``None`` (present it from
            its facts).
    """

    run_id: str
    procedure: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    status: str = ""
    reason: str = ""
    started_utc: str = ""
    finished_utc: str = ""
    data_file: str = ""
    legacy_entry_url: str = ""
    bundle: Bundle | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form."""
        return {
            "run_id": self.run_id,
            "procedure": self.procedure,
            "params": dict(self.params),
            "status": self.status,
            "reason": self.reason,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "data_file": self.data_file,
            "legacy_entry_url": self.legacy_entry_url,
            "bundle": self.bundle.to_dict() if self.bundle is not None else None,
        }

    @classmethod
    def from_dict(cls, data: object) -> RunContext:
        """Load tolerantly."""
        payload = _dict(data)
        bundle = payload.get("bundle")
        return cls(
            run_id=_text(payload.get("run_id"), 200),
            procedure=_text(payload.get("procedure"), 200),
            params=_dict(payload.get("params")),
            status=_text(payload.get("status"), 32),
            reason=_text(payload.get("reason")),
            started_utc=_text(payload.get("started_utc"), 64),
            finished_utc=_text(payload.get("finished_utc"), 64),
            data_file=_text(payload.get("data_file"), 4096),
            legacy_entry_url=_text(payload.get("legacy_entry_url"), 2000),
            bundle=Bundle.from_dict(bundle) if isinstance(bundle, dict) else None,
        )


@dataclass(frozen=True)
class RenderContext:
    """Everything one publish's section may show.

    Attributes:
        publish_id: This publish's id (also stamped into the section heading
            by the framework).
        published_utc: When this publish was made.
        experiment: ``experiment_id``, ``title``, ``user_name``,
            ``sample_info``, ``findings``, ``config_name``.
        runs: The runs being published, oldest first.
        options: The profile's ``render:`` options, passed through.
    """

    publish_id: str = ""
    published_utc: str = ""
    experiment: dict[str, Any] = field(default_factory=dict)
    runs: tuple[RunContext, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form."""
        return {
            "publish_id": self.publish_id,
            "published_utc": self.published_utc,
            "experiment": dict(self.experiment),
            "runs": [run.to_dict() for run in self.runs],
            "options": dict(self.options),
        }

    @classmethod
    def from_dict(cls, data: object) -> RenderContext:
        """Load tolerantly."""
        payload = _dict(data)
        return cls(
            publish_id=_text(payload.get("publish_id"), 200),
            published_utc=_text(payload.get("published_utc"), 64),
            experiment=_dict(payload.get("experiment")),
            runs=tuple(RunContext.from_dict(r) for r in _list(payload.get("runs"))),
            options=_dict(payload.get("options")),
        )


# ----------------------------------------------------------------------
# What a renderer returns
# ----------------------------------------------------------------------


def heading(text: str, level: int = 3) -> dict[str, Any]:
    """A heading (level 2–5)."""
    return {"type": BLOCK_HEADING, "text": text, "level": level}


def paragraph(text: str) -> dict[str, Any]:
    """A paragraph of plain text (escaped, never markup)."""
    return {"type": BLOCK_PARAGRAPH, "text": text}


def markdown(text: str) -> dict[str, Any]:
    """Restricted Markdown: paragraphs, lists, ``**bold**``, ``*italic*``, ``code``, links."""
    return {"type": BLOCK_MARKDOWN, "text": text}


def html(fragment: str) -> dict[str, Any]:
    """An HTML fragment; sanitised to a small allow-list of tags before publishing."""
    return {"type": BLOCK_HTML, "html": fragment}


def table(caption: str, columns: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    """A table."""
    return {"type": BLOCK_TABLE, "caption": caption, "columns": columns, "rows": rows}


def results(caption: str, values: list[dict[str, Any]]) -> dict[str, Any]:
    """Derived values: ``[{name, value, unit, uncertainty, note}]``."""
    return {"type": BLOCK_RESULTS, "caption": caption, "values": values}


def figure(bundle_id: str, artifact_id: str, caption: str = "") -> dict[str, Any]:
    """A figure from a bundle, uploaded to the page and named in the section."""
    return {"type": BLOCK_FIGURE, "bundle_id": bundle_id, "artifact_id": artifact_id, "caption": caption}


def key_values(caption: str, pairs: list[tuple[str, Any]] | dict[str, Any]) -> dict[str, Any]:
    """A two-column list of names and values."""
    items = list(pairs.items()) if isinstance(pairs, dict) else list(pairs)
    return {"type": BLOCK_KEY_VALUES, "caption": caption, "pairs": [[str(k), v] for k, v in items]}


def link(text: str, url: str) -> dict[str, Any]:
    """A link (``http``/``https`` only)."""
    return {"type": BLOCK_LINK, "text": text, "url": url}


def _clean_block(block: object) -> dict[str, Any] | None:
    """Return one block with its caps applied, or ``None`` for junk."""
    raw = _dict(block)
    kind = raw.get("type")
    if kind not in BLOCK_TYPES:
        return None
    if kind == BLOCK_HEADING:
        level = raw.get("level", 3)
        level = min(max(level, 2), 5) if isinstance(level, int) and not isinstance(level, bool) else 3
        return {"type": kind, "text": _text(raw.get("text"), 500), "level": level}
    if kind in (BLOCK_PARAGRAPH, BLOCK_MARKDOWN):
        return {"type": kind, "text": _text(raw.get("text"))}
    if kind == BLOCK_HTML:
        return {"type": kind, "html": _text(raw.get("html"), 100_000)}
    if kind == BLOCK_TABLE:
        columns = [_text(c, 200) for c in _list(raw.get("columns"))[:MAX_TABLE_COLUMNS]]
        width = len(columns)
        rows = [
            [_cell(c) for c in (_list(r)[:width] + [None] * (width - len(_list(r))))]
            for r in _list(raw.get("rows"))[:MAX_TABLE_ROWS]
        ]
        return {"type": kind, "caption": _text(raw.get("caption"), 500), "columns": columns, "rows": rows}
    if kind == BLOCK_RESULTS:
        values = []
        for item in _list(raw.get("values"))[:MAX_TABLE_ROWS]:
            entry = _dict(item)
            values.append(
                {
                    "name": _text(entry.get("name"), 200),
                    "value": _cell(entry.get("value")),
                    "unit": _text(entry.get("unit"), 50),
                    "uncertainty": _cell(entry.get("uncertainty")),
                    "note": _text(entry.get("note"), 500),
                }
            )
        return {"type": kind, "caption": _text(raw.get("caption"), 500), "values": values}
    if kind == BLOCK_FIGURE:
        return {
            "type": kind,
            "bundle_id": _text(raw.get("bundle_id"), 200),
            "artifact_id": _text(raw.get("artifact_id"), 255),
            "caption": _text(raw.get("caption"), 1000),
        }
    if kind == BLOCK_KEY_VALUES:
        pairs = [
            [_text(_list(p)[0] if _list(p) else "", 200), _cell(_list(p)[1] if len(_list(p)) > 1 else None)]
            for p in _list(raw.get("pairs"))[:MAX_TABLE_ROWS]
        ]
        return {"type": kind, "caption": _text(raw.get("caption"), 500), "pairs": pairs}
    return {"type": kind, "text": _text(raw.get("text"), 500), "url": _text(raw.get("url"), 2000)}


@dataclass(frozen=True)
class Section:
    """One publish's section, as a renderer lays it out.

    Attributes:
        title: The section's title (the framework prefixes the publish stamp).
        blocks: Ordered block dicts (build them with the helpers above).
        fields: Page fields to overwrite: ``{field name: text value}``.
        tags: Tags to add to the page.
    """

    title: str = ""
    blocks: tuple[dict[str, Any], ...] = ()
    fields: dict[str, str] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def figures(self) -> list[dict[str, Any]]:
        """Return the figure blocks, in order."""
        return [b for b in self.blocks if b.get("type") == BLOCK_FIGURE]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form."""
        return {
            "title": self.title,
            "blocks": [dict(b) for b in self.blocks],
            "fields": dict(self.fields),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: object) -> Section:
        """Load what a renderer returned, tolerantly and with every cap applied.

        This is the one door renderer output comes through: unknown block
        types are dropped, every string is bounded, and a non-dict answer is
        an empty section.
        """
        payload = _dict(data)
        blocks = [b for b in (_clean_block(x) for x in _list(payload.get("blocks"))[:MAX_BLOCKS]) if b]
        fields = {
            _text(k, 200): _text(v, 2000)
            for k, v in list(_dict(payload.get("fields")).items())[:MAX_FIELDS]
        }
        tags = tuple(_text(t, 100) for t in _list(payload.get("tags"))[:MAX_TAGS] if t)
        return cls(title=_text(payload.get("title"), 500), blocks=tuple(blocks), fields=fields, tags=tags)
