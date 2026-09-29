"""Finding blocks — without ever running one in the application's process.

Blocks live in two places, with the same layout:

* shipped (maintained with I2AS): ``i2as/blocks/shipped/{connectors,renderers,profiles}/``;
* the user's own: ``<user config dir>/blocks/{connectors,renderers,profiles}/``.

A block is identified by what it declares, read STATICALLY from its source
with ``ast`` — a connector's ``backend``, ``display_name``,
``contract_version``, ``capabilities`` and ``settings_schema`` (all literals),
a renderer's ``NAME``, ``DESCRIPTION`` and ``CONTRACT_VERSION`` — so listing
the blocks, filling the Settings dialog and pinning an experiment never
execute user code in the application. Blocks are executed only in the helper
process (``i2as.blocks.host``). A user block with the same id as a shipped
one takes its place.

Every block is identified by the SHA-256 of its source (``digest``): an
experiment pins the digest it was linked with, so an edited block never
silently changes how an ongoing experiment is published.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from i2as.blocks.connector import CONNECTOR_CONTRACT_VERSION, ElnCapabilities, ElnConnector
from i2as.blocks.renderer import RENDERER_CONTRACT_VERSION

KIND_CONNECTOR = "connector"
KIND_RENDERER = "renderer"
KIND_PROFILE = "profile"
BLOCK_KINDS: tuple[str, ...] = (KIND_CONNECTOR, KIND_RENDERER, KIND_PROFILE)

#: Sub-folder per kind, in both the shipped and the user tree.
KIND_DIRS: dict[str, str] = {
    KIND_CONNECTOR: "connectors",
    KIND_RENDERER: "renderers",
    KIND_PROFILE: "profiles",
}

#: The largest block source file read.
MAX_BLOCK_BYTES = 1_000_000

SOURCE_SHIPPED = "shipped"
SOURCE_USER = "user"


def shipped_dir() -> Path:
    """Return the folder holding the shipped blocks."""
    return Path(__file__).resolve().parent / "shipped"


@dataclass(frozen=True)
class BlockInfo:
    """One block, as discovered (never executed).

    Attributes:
        kind: ``connector`` / ``renderer`` / ``profile``.
        block_id: Its id (``backend`` for a connector, ``NAME`` for a
            renderer, the file stem for a profile).
        path: Its source file.
        digest: SHA-256 of the source.
        source: ``shipped`` or ``user``.
        display_name: What the GUI shows.
        description: One line.
        contract_version: The contract it declares.
        capabilities: A connector's declared capabilities.
        settings_schema: A connector's declared settings schema.
        error: Why the block is unusable, or ``""``.
    """

    kind: str
    block_id: str
    path: Path
    digest: str = ""
    source: str = SOURCE_USER
    display_name: str = ""
    description: str = ""
    contract_version: int = 0
    capabilities: ElnCapabilities = field(default_factory=ElnCapabilities)
    settings_schema: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def usable(self) -> bool:
        """Whether the block can be run."""
        return not self.error

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form."""
        return {
            "kind": self.kind,
            "id": self.block_id,
            "path": str(self.path),
            "digest": self.digest,
            "source": self.source,
            "display_name": self.display_name,
            "description": self.description,
            "contract_version": self.contract_version,
            "capabilities": self.capabilities.to_dict(),
            "settings_schema": self.settings_schema,
            "error": self.error,
        }


def file_digest(path: str | Path) -> str:
    """Return the SHA-256 of one file, or ``""`` when unreadable."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def _literal(node: ast.AST) -> Any:
    """Evaluate a literal node, or raise ``ValueError``."""
    return ast.literal_eval(node)


def _capabilities(node: ast.AST) -> ElnCapabilities:
    """Read ``ElnCapabilities(flag=literal, ...)`` statically."""
    if isinstance(node, ast.Call):
        values = {kw.arg: _literal(kw.value) for kw in node.keywords if kw.arg}
        return ElnCapabilities.from_dict(values)
    raise ValueError("capabilities must be an ElnCapabilities(...) call with literal flags")


def _read_connector(path: Path, tree: ast.Module) -> dict[str, Any]:
    """Return a connector's static declarations."""
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            (isinstance(b, ast.Name) and b.id == "ElnConnector")
            or (isinstance(b, ast.Attribute) and b.attr == "ElnConnector")
            for b in node.bases
        )
    ]
    if len(classes) != 1:
        raise ValueError("a connector module defines exactly one ElnConnector subclass")
    found: dict[str, Any] = {"contract_version": CONNECTOR_CONTRACT_VERSION}
    for statement in classes[0].body:
        target = None
        value = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            target, value = statement.target, statement.value
        if not isinstance(target, ast.Name) or value is None:
            continue
        name = target.id
        if name == "capabilities":
            found[name] = _capabilities(value)
        elif name in ("backend", "display_name", "contract_version", "settings_schema"):
            found[name] = _literal(value)
    found["description"] = ast.get_docstring(classes[0]) or ""
    return found


def _read_renderer(tree: ast.Module) -> dict[str, Any]:
    """Return a renderer's static declarations."""
    found: dict[str, Any] = {"contract_version": RENDERER_CONTRACT_VERSION}
    has_render = False
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef) and statement.name == "render":
            has_render = True
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id in ("NAME", "DESCRIPTION", "CONTRACT_VERSION"):
                found[target.id] = _literal(statement.value)
    if not has_render:
        raise ValueError("a renderer module defines a top-level render(context) function")
    return found


def describe_block(kind: str, path: str | Path, source: str = SOURCE_USER) -> BlockInfo:
    """Describe one block file statically. Never raises, never executes it.

    Args:
        kind: The block kind.
        path: Its source file.
        source: ``shipped`` or ``user``.

    Returns:
        The block's description; ``error`` says why it is unusable.
    """
    file = Path(path)
    stem = file.stem
    try:
        if file.stat().st_size > MAX_BLOCK_BYTES:
            return BlockInfo(kind, stem, file, source=source, error="file is too large")
        text = file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return BlockInfo(kind, stem, file, source=source, error=f"unreadable: {exc}")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if kind == KIND_PROFILE:
        return BlockInfo(kind, stem, file, digest=digest, source=source, display_name=stem)
    try:
        tree = ast.parse(text, filename=str(file))
        if kind == KIND_CONNECTOR:
            found = _read_connector(file, tree)
            backend = found.get("backend")
            if not isinstance(backend, str) or not backend.isidentifier() or backend != backend.lower():
                raise ValueError("backend must be a lowercase identifier literal")
            version = found.get("contract_version")
            schema = found.get("settings_schema", {"properties": {}})
            if not isinstance(schema, dict):
                raise ValueError("settings_schema must be a literal dict")
            return BlockInfo(
                kind,
                backend,
                file,
                digest=digest,
                source=source,
                display_name=str(found.get("display_name") or backend),
                description=str(found.get("description", "")).split("\n")[0],
                contract_version=version if isinstance(version, int) else 0,
                capabilities=found.get("capabilities", ElnCapabilities()),
                settings_schema=schema,
                error="" if version == CONNECTOR_CONTRACT_VERSION else f"unsupported contract_version {version!r}",
            )
        found = _read_renderer(tree)
        name = found.get("NAME", stem)
        version = found.get("CONTRACT_VERSION", RENDERER_CONTRACT_VERSION)
        return BlockInfo(
            kind,
            str(name),
            file,
            digest=digest,
            source=source,
            display_name=str(name),
            description=str(found.get("DESCRIPTION", "")),
            contract_version=version if isinstance(version, int) else 0,
            error="" if version == RENDERER_CONTRACT_VERSION else f"unsupported CONTRACT_VERSION {version!r}",
        )
    except (SyntaxError, ValueError, TypeError) as exc:
        return BlockInfo(kind, stem, file, digest=digest, source=source, error=str(exc))


def discover_blocks(kind: str, user_root: str | Path | None = None) -> dict[str, BlockInfo]:
    """Return every block of one kind, keyed by id; user blocks override shipped ones.

    Args:
        kind: The block kind.
        user_root: The user's ``blocks`` folder, or ``None`` for shipped only.

    Returns:
        ``{block_id: BlockInfo}``, unusable blocks included (with ``error``).
    """
    patterns = ("*.yaml", "*.yml") if kind == KIND_PROFILE else ("*.py",)
    found: dict[str, BlockInfo] = {}
    roots = [(shipped_dir(), SOURCE_SHIPPED)]
    if user_root is not None:
        roots.append((Path(user_root), SOURCE_USER))
    for root, source in roots:
        folder = root / KIND_DIRS[kind]
        if not folder.is_dir():
            continue
        for pattern in patterns:
            for path in sorted(folder.glob(pattern)):
                if path.name.startswith(("_", ".")):
                    continue
                info = describe_block(kind, path, source)
                found[info.block_id] = info
    return found


def load_block_module(path: str | Path) -> ModuleType:
    """Import one block file by path. ONLY the helper process and the checker call this.

    Args:
        path: The block's source file.

    Returns:
        The imported module.

    Raises:
        ImportError: The file cannot be imported.
    """
    file = Path(path)
    name = f"i2as_block_{hashlib.sha256(str(file).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(name, file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def connector_class(module: ModuleType) -> type[ElnConnector]:
    """Return the one ``ElnConnector`` subclass a module defines.

    Raises:
        ImportError: There is not exactly one.
    """
    classes = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, ElnConnector)
        and value is not ElnConnector
        and value.__module__ == module.__name__
    ]
    if len(classes) != 1:
        raise ImportError(f"expected exactly one ElnConnector subclass, found {len(classes)}")
    return classes[0]
