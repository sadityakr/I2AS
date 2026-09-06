"""i2as.session.eln — ELN adapters, renderers, outbox, publisher, drafting (L6).

See ``i2as/session/eln/README.md`` for the folder standard, ``adapter.py``
for the ELN adapter standard itself, ``drafting.py`` for the draft prompt
standard and the **Draft client** contract, and ``GLOSSARY.md`` for the **ELN
adapter** / **Outbox** / **Draft entry** / **Draft client** vocabulary.
"""

from i2as.session.eln.adapter import (
    ElnAdapter,
    ElnCapabilities,
    ElnEntryRef,
    ElnError,
    ElnTemplate,
)
from i2as.session.eln.drafting import (
    COST_FIELDS,
    DRAFT_SYSTEM_PROMPT,
    DRAFT_TAG,
    AnthropicDraftClient,
    CompletionResult,
    DraftClient,
    DraftEntry,
    DraftRequest,
    FakeDraftClient,
    cost_line,
    cost_usd,
    draft_entry,
    manifest_from_run,
    parse_completion,
    prompt_digest,
    render_draft_prompt,
)
from i2as.session.eln.elabftw import (
    ElabFtwAdapter,
    ElnHttpTransport,
    HttpResponse,
    UrllibTransport,
)
from i2as.session.eln.outbox import (
    DRAIN_IDLE,
    DRAIN_PUBLISHED,
    DRAIN_RETRY,
    JOB_PUBLISH_RUN,
    JOB_STATE_DONE,
    JOB_STATE_PENDING,
    DrainResult,
    Outbox,
    OutboxJob,
)
from i2as.session.eln.publisher import (
    PUBLISH_DISABLED,
    PUBLISH_OFFLINE,
    PUBLISH_PENDING,
    PUBLISH_SYNCED,
    ElnPublisher,
    discover_backends,
)
from i2as.session.eln.settings import (
    API_KEY_ENV_VAR,
    ASSISTANT_API_KEY_ENV_VAR,
    DEFAULT_ASSISTANT_MODEL,
    DEFAULT_MODEL_PRICES,
    SETTINGS_PATH_ENV_VAR,
    AssistantSettings,
    ElnSettings,
    eln_settings_path,
    load_eln_settings,
)
from i2as.session.eln.sim_eln import SimElnAdapter

__all__ = [
    "ElnAdapter",
    "ElnCapabilities",
    "ElnEntryRef",
    "ElnError",
    "ElnTemplate",
    "ElnSettings",
    "AssistantSettings",
    "eln_settings_path",
    "load_eln_settings",
    "API_KEY_ENV_VAR",
    "ASSISTANT_API_KEY_ENV_VAR",
    "SETTINGS_PATH_ENV_VAR",
    "DEFAULT_ASSISTANT_MODEL",
    "DEFAULT_MODEL_PRICES",
    "SimElnAdapter",
    "ElabFtwAdapter",
    "ElnHttpTransport",
    "HttpResponse",
    "UrllibTransport",
    "Outbox",
    "OutboxJob",
    "DrainResult",
    "JOB_PUBLISH_RUN",
    "JOB_STATE_PENDING",
    "JOB_STATE_DONE",
    "DRAIN_IDLE",
    "DRAIN_PUBLISHED",
    "DRAIN_RETRY",
    "ElnPublisher",
    "discover_backends",
    "PUBLISH_SYNCED",
    "PUBLISH_PENDING",
    "PUBLISH_OFFLINE",
    "PUBLISH_DISABLED",
    "DraftClient",
    "DraftRequest",
    "DraftEntry",
    "CompletionResult",
    "FakeDraftClient",
    "AnthropicDraftClient",
    "DRAFT_SYSTEM_PROMPT",
    "DRAFT_TAG",
    "COST_FIELDS",
    "render_draft_prompt",
    "prompt_digest",
    "parse_completion",
    "manifest_from_run",
    "cost_line",
    "cost_usd",
    "draft_entry",
]
