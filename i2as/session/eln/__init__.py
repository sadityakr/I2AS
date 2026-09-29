"""i2as.session.eln — the notebook bridge: linking, reading back, publishing (L6).

Three independent layers meet here and only here:

* the **analysis stage** writes sealed, notebook-agnostic **bundles**
  (``i2as.analysis.bundle``) and knows nothing about any notebook;
* the **ELN connection** is a user block — a connector, a renderer, a YAML
  profile (``i2as.blocks``) — run in a helper process (``block_runner``);
* **publishing** (``publishing``, ``outbox``, ``ledger``) reads the bundles and
  the experiment's binding and appends one section per publish to the
  experiment's ONE page.

``service.ElnService`` is the application's single door to all of it, and
does every notebook call on its own worker thread.
"""

from i2as.session.eln.block_runner import (
    BlockRunner,
    BlockTimeout,
    InProcessBlockRunner,
    SubprocessBlockRunner,
)
from i2as.session.eln.outbox import Outbox, OutboxJob
from i2as.session.eln.publishing import BlockCatalog, PublishError
from i2as.session.eln.service import ElnService

__all__ = [
    "BlockCatalog",
    "BlockRunner",
    "BlockTimeout",
    "ElnService",
    "InProcessBlockRunner",
    "Outbox",
    "OutboxJob",
    "PublishError",
    "SubprocessBlockRunner",
]
