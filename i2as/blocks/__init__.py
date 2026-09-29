"""i2as.blocks — what a user block is written against (tier 2).

A **block** is a small piece of Python a lab writes for itself, often with a
coding agent, and I2AS runs safely: an ELN **connector** (talks to one
notebook), a **renderer** (lays out a publish's section) or a **profile**
(YAML: which connector, renderer and template, and how fields map). Blocks run
in a helper process (``python -m i2as.blocks.host``), never in the
application's own process and never on the instrument or GUI thread, so a
block that fails costs a visible error and a retry — never the experiment.

Import what you need from here::

    from i2as.blocks import ElnConnector, ElnCapabilities, ElnIdentity, ...
    from i2as.blocks import Section, RenderContext, heading, paragraph, figure, ...

Scaffold, check and list blocks with ``python -m i2as.blocks``. This package
imports only the standard library and ``i2as.analysis.bundle``.
"""

from i2as.blocks.connector import (
    CONNECTOR_CONTRACT_VERSION,
    KIND_ENTRY,
    KIND_ITEM,
    ElnAuthError,
    ElnCapabilities,
    ElnConnector,
    ElnEntryRef,
    ElnError,
    ElnHit,
    ElnIdentity,
    ElnNotFound,
    ElnQuery,
    ElnRecord,
    ElnRef,
    ElnTemplate,
    ElnTransientError,
    ElnValidationError,
)
from i2as.blocks.http import (
    ElnHttpTransport,
    HttpResponse,
    UrllibTransport,
    multipart_file,
    raise_for_status,
)
from i2as.blocks.renderer import (
    RENDERER_CONTRACT_VERSION,
    RenderContext,
    RunContext,
    Section,
    figure,
    heading,
    html,
    key_values,
    link,
    markdown,
    paragraph,
    results,
    table,
)

__all__ = [
    "CONNECTOR_CONTRACT_VERSION",
    "KIND_ENTRY",
    "KIND_ITEM",
    "RENDERER_CONTRACT_VERSION",
    "ElnAuthError",
    "ElnCapabilities",
    "ElnConnector",
    "ElnEntryRef",
    "ElnError",
    "ElnHit",
    "ElnHttpTransport",
    "ElnIdentity",
    "ElnNotFound",
    "ElnQuery",
    "ElnRecord",
    "ElnRef",
    "ElnTemplate",
    "ElnTransientError",
    "ElnValidationError",
    "HttpResponse",
    "RenderContext",
    "RunContext",
    "Section",
    "UrllibTransport",
    "figure",
    "heading",
    "html",
    "key_values",
    "link",
    "markdown",
    "multipart_file",
    "paragraph",
    "raise_for_status",
    "results",
    "table",
]
