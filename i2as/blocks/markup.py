"""Safe markup — the one door a renderer's section comes through on its way to a page.

A renderer is user code (tier 2) and a bundle's text is producer output
(tier 3), so nothing either of them writes reaches a notebook page as markup
of its own choosing. ``section_to_html()`` builds the HTML itself from the
``Section``'s blocks: every text is escaped, Markdown is limited to
paragraphs, lists, emphasis, inline code and ``http(s)`` links, and an
explicit HTML block is reduced to a small allow-list of tags and attributes
(no scripts, no styles, no images, no event handlers, no ``javascript:``).
The section heading — carrying the publish id a retry looks for — is written
by the framework, never by the renderer.
"""

from __future__ import annotations

import html as _html
import re
from collections.abc import Mapping
from html.parser import HTMLParser
from typing import Any

from i2as.blocks.renderer import (
    BLOCK_FIGURE,
    BLOCK_HEADING,
    BLOCK_HTML,
    BLOCK_KEY_VALUES,
    BLOCK_LINK,
    BLOCK_MARKDOWN,
    BLOCK_PARAGRAPH,
    BLOCK_RESULTS,
    BLOCK_TABLE,
    Section,
)

#: Tags an HTML block may keep; everything else is dropped (its text kept).
ALLOWED_TAGS: frozenset[str] = frozenset(
    {
        "p", "br", "hr", "h2", "h3", "h4", "h5", "ul", "ol", "li", "strong", "b",
        "em", "i", "code", "pre", "blockquote", "table", "thead", "tbody", "tr",
        "th", "td", "caption", "a", "span", "div", "sub", "sup",
    }
)

#: Tags whose CONTENT is dropped too.
DROPPED_CONTENT_TAGS: frozenset[str] = frozenset({"script", "style", "iframe", "object", "embed", "template"})

#: Attributes kept, per tag.
ALLOWED_ATTRIBUTES: dict[str, frozenset[str]] = {
    "a": frozenset({"href", "title"}),
    "td": frozenset({"colspan", "rowspan"}),
    "th": frozenset({"colspan", "rowspan"}),
}

_VOID_TAGS = frozenset({"br", "hr"})
_SAFE_URL = re.compile(r"^https?://", re.IGNORECASE)


def escape(value: object) -> str:
    """Escape any value as HTML text (``None`` → ``""``)."""
    return _html.escape("" if value is None else str(value), quote=True)


def is_safe_url(url: str) -> bool:
    """Whether ``url`` is an absolute ``http``/``https`` URL."""
    return bool(_SAFE_URL.match(url.strip())) and not any(c in url for c in "\"'<>\n\r")


class _Sanitizer(HTMLParser):
    """Rebuilds an HTML fragment from allowed tags and escaped text only."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self._open: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in DROPPED_CONTENT_TAGS:
            self._skip += 1
            return
        if self._skip or tag not in ALLOWED_TAGS:
            return
        kept: list[str] = []
        for name, value in attrs:
            name = name.lower()
            if name not in ALLOWED_ATTRIBUTES.get(tag, frozenset()) or value is None:
                continue
            if name == "href" and not is_safe_url(value):
                continue
            if name in ("colspan", "rowspan") and not value.isdigit():
                continue
            kept.append(f' {name}="{escape(value)}"')
        self.out.append(f"<{tag}{''.join(kept)}>")
        if tag not in _VOID_TAGS:
            self._open.append(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in DROPPED_CONTENT_TAGS:
            self._skip = max(self._skip - 1, 0)
            return
        if self._skip or tag not in self._open:
            return
        while self._open:
            current = self._open.pop()
            self.out.append(f"</{current}>")
            if current == tag:
                break

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.out.append(escape(data))

    def close_all(self) -> str:
        self.close()
        while self._open:
            self.out.append(f"</{self._open.pop()}>")
        return "".join(self.out)


def sanitize_html(fragment: str) -> str:
    """Reduce an HTML fragment to the allow-list; the text always survives, escaped.

    Args:
        fragment: Untrusted HTML.

    Returns:
        Well-formed HTML using only ``ALLOWED_TAGS`` and ``ALLOWED_ATTRIBUTES``.
    """
    parser = _Sanitizer()
    parser.feed(fragment or "")
    return parser.close_all()


_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_BOLD = re.compile(r"\*\*([^*\n]+)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])")
_BULLET = re.compile(r"^\s*[-*]\s+")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+")
_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")


def _inline(text: str) -> str:
    """Render inline Markdown over ESCAPED text."""
    escaped = escape(text)
    codes: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        codes.append(f"<code>{match.group(1)}</code>")
        return f"\x00{len(codes) - 1}\x00"

    escaped = _INLINE_CODE.sub(_stash, escaped)
    escaped = _LINK.sub(lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', escaped)
    escaped = _BOLD.sub(r"<strong>\1</strong>", escaped)
    escaped = _ITALIC.sub(r"<em>\1</em>", escaped)
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], escaped)


def markdown_to_html(text: str) -> str:
    """Render restricted Markdown: paragraphs, headings, lists and inline marks.

    Args:
        text: Markdown (untrusted).

    Returns:
        HTML built only from ``p``, ``h4``, ``ul``/``ol``/``li``, ``strong``,
        ``em``, ``code`` and ``a`` with an ``http(s)`` link — nothing the
        input wrote as HTML survives as markup.
    """
    blocks = re.split(r"\n\s*\n", (text or "").strip())
    out: list[str] = []
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if all(_BULLET.match(line) for line in lines):
            items = "".join(f"<li>{_inline(_BULLET.sub('', line))}</li>" for line in lines)
            out.append(f"<ul>{items}</ul>")
        elif all(_NUMBERED.match(line) for line in lines):
            items = "".join(f"<li>{_inline(_NUMBERED.sub('', line))}</li>" for line in lines)
            out.append(f"<ol>{items}</ol>")
        elif len(lines) == 1 and lines[0].startswith("#"):
            out.append(f"<h4>{_inline(lines[0].lstrip('#').strip())}</h4>")
        else:
            out.append(f"<p>{'<br>'.join(_inline(line) for line in lines)}</p>")
    return "".join(out)


def _format_value(value: Any, unit: str = "") -> str:
    """Format one value (and unit) as text."""
    if isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = "" if value is None else str(value)
    return f"{text} {unit}".strip() if unit and text else text


def _table_html(caption: str, columns: list[str], rows: list[list[Any]], *, header: bool = True) -> str:
    """Render one escaped table (``header=False``: no heading row, for name/value lists)."""
    head = "".join(f"<th>{escape(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(_format_value(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    cap = f"<caption>{escape(caption)}</caption>" if caption else ""
    thead = f"<thead><tr>{head}</tr></thead>" if header else ""
    return f"<table>{cap}{thead}<tbody>{body}</tbody></table>"


def block_to_html(block: Mapping[str, Any], figure_names: Mapping[tuple[str, str], str]) -> str:
    """Render one (already capped) block as safe HTML.

    Args:
        block: One block from ``Section.from_dict()``.
        figure_names: ``{(bundle_id, artifact_id): file name as uploaded}``;
            a figure missing from it is named as not uploaded.

    Returns:
        The HTML, or ``""`` for a block with nothing to show.
    """
    kind = block.get("type")
    if kind == BLOCK_HEADING:
        level = int(block.get("level", 3))
        return f"<h{level}>{escape(block.get('text'))}</h{level}>"
    if kind == BLOCK_PARAGRAPH:
        text = str(block.get("text") or "")
        return f"<p>{escape(text).replace(chr(10), '<br>')}</p>" if text else ""
    if kind == BLOCK_MARKDOWN:
        return markdown_to_html(str(block.get("text") or ""))
    if kind == BLOCK_HTML:
        return sanitize_html(str(block.get("html") or ""))
    if kind == BLOCK_TABLE:
        return _table_html(str(block.get("caption") or ""), list(block.get("columns") or []), list(block.get("rows") or []))
    if kind == BLOCK_RESULTS:
        rows = []
        for value in block.get("values") or []:
            uncertainty = value.get("uncertainty")
            shown = _format_value(value.get("value"))
            if uncertainty not in (None, ""):
                shown = f"{shown} ± {_format_value(uncertainty)}"
            rows.append([value.get("name"), f"{shown} {value.get('unit') or ''}".strip(), value.get("note") or ""])
        return _table_html(str(block.get("caption") or "Results"), ["Quantity", "Value", "Note"], rows)
    if kind == BLOCK_FIGURE:
        key = (str(block.get("bundle_id") or ""), str(block.get("artifact_id") or ""))
        name = figure_names.get(key)
        caption = str(block.get("caption") or "")
        if name is None:
            return f"<p><em>Figure {escape(key[1])} was not uploaded.</em></p>"
        label = f"Figure <code>{escape(name)}</code> (attached)"
        return f"<p>{label}{': ' + escape(caption) if caption else ''}</p>"
    if kind == BLOCK_KEY_VALUES:
        pairs = [[k, v] for k, v in block.get("pairs") or []]
        return _table_html(str(block.get("caption") or ""), ["", ""], pairs, header=False) if pairs else ""
    if kind == BLOCK_LINK:
        url = str(block.get("url") or "")
        text = escape(block.get("text") or url)
        return f'<p><a href="{escape(url)}">{text}</a></p>' if is_safe_url(url) else f"<p>{text}</p>"
    return ""


def section_heading(publish_id: str, published_utc: str, title: str) -> str:
    """Return the framework-owned heading of one appended section.

    The publish id is written as plain text, so it survives any notebook's
    own sanitiser and a retry can find it (``ElnConnector.has_section``).

    Args:
        publish_id: The publish.
        published_utc: When it was made (ISO 8601).
        title: The renderer's title.

    Returns:
        ``<h2>I2AS · <time> · <title> · <publish_id></h2>``.
    """
    stamp = published_utc.replace("T", " ")[:16] + " UTC" if published_utc else ""
    parts = [p for p in ("I2AS", stamp, title, publish_id) if p]
    return f"<h2>{' · '.join(escape(p) for p in parts)}</h2>"


def section_to_html(
    section: Section,
    *,
    publish_id: str,
    published_utc: str,
    figure_names: Mapping[tuple[str, str], str] | None = None,
) -> str:
    """Render one section, headed by the framework, as safe HTML.

    Args:
        section: What the renderer returned (loaded through
            ``Section.from_dict``).
        publish_id: The publish.
        published_utc: When it was made.
        figure_names: Uploaded figure names, by ``(bundle_id, artifact_id)``.

    Returns:
        The section's HTML, ready to append.
    """
    names = figure_names or {}
    body = "".join(block_to_html(block, names) for block in section.blocks)
    return f"<div>{section_heading(publish_id, published_utc, section.title)}{body}</div>"
