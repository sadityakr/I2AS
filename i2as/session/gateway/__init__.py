"""i2as.session.gateway — the Agent gateway (L6).

The second adapter of the **Control contract**: an in-process client that
speaks the same ``Command`` / ``Verdict`` / ``Event`` vocabulary the GUI
does, with a permission model in front of it. See
``i2as/session/gateway/README.md`` for the folder standard, ``roles.py``
for the permission standard itself, ``local_server.py`` for the
**Gateway server** that carries that same client over a local socket,
``action_classes.py`` for the (provisional) classification of every action, ``tools.py`` for the **Tool
surface** rendered from the contract and the station declaration, and
``GLOSSARY.md`` for the **Role** / **Action class** / **Attendance** /
**Kill switch** / **Agent gateway** / **Tool surface** / **Tool spec**
vocabulary.
"""

from i2as.session.gateway.action_classes import (
    COMMAND_ACTION_CLASSES,
    LIFECYCLE_ACTION_CLASSES,
    ActionClass,
    ClassifiedAction,
    UnclassifiedActionError,
    classify_command,
    classify_control,
)
from i2as.session.gateway.gateway import EngineClient, Gateway
from i2as.session.gateway.local_server import (
    MAX_FRAME_BYTES,
    SCHEMA_VERSION,
    GatewayServer,
    default_socket_name,
    descriptor_path,
)
from i2as.session.gateway.roles import (
    PERMISSION_MATRIX,
    ROLE_LADDER,
    Permission,
    Role,
    authorize,
    authorize_spooled,
    role_within_ceiling,
)
from i2as.session.gateway.tools import (
    COMMAND_ARG_SCHEMAS,
    SESSION_TOOLS,
    ToolContext,
    ToolError,
    ToolSpec,
    capability_tool_name,
    render_capability_tools,
    render_command_tools,
    render_tools,
    validate_tool_args,
)

__all__ = [
    "ActionClass",
    "ClassifiedAction",
    "UnclassifiedActionError",
    "COMMAND_ACTION_CLASSES",
    "LIFECYCLE_ACTION_CLASSES",
    "classify_command",
    "classify_control",
    "Role",
    "Permission",
    "PERMISSION_MATRIX",
    "ROLE_LADDER",
    "authorize",
    "authorize_spooled",
    "role_within_ceiling",
    "Gateway",
    "EngineClient",
    "GatewayServer",
    "SCHEMA_VERSION",
    "MAX_FRAME_BYTES",
    "descriptor_path",
    "default_socket_name",
    "ToolSpec",
    "ToolContext",
    "ToolError",
    "COMMAND_ARG_SCHEMAS",
    "SESSION_TOOLS",
    "render_tools",
    "render_command_tools",
    "render_capability_tools",
    "capability_tool_name",
    "validate_tool_args",
]
