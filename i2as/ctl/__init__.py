"""i2as.ctl — the **Reference client** of the control contract.

``python -m i2as.ctl`` is the terminal end of the **Agent gateway**: one
subcommand per thing a client can do, JSON in and JSON out, the same **Tool
surface** an in-process agent is offered and the same **Verdict** answering
every request. It exists for three readers at once — a physicist checking a
running experiment from a shell, an agent harness that has no tool-use runtime
yet, and the integration tests, which drive this module's ``main()`` rather
than a hand-built client so the thing under test is the thing that ships.

Two modes, one client object:

* ``--offline <config_dir>`` builds a whole simulated instrument stack in this
  process (``InstrumentHost(mode="inline")``) and hands the gateway the real
  engine. Nothing here talks to hardware that a config does not describe.
* **live** (the default) talks to an application that is already running,
  through the **Request spool**: a command becomes a file the running engine's
  tick drains, and its verdict comes back through the spool's JSONL sink.

See ``i2as/ctl/README.md`` for the folder standard and the exit codes,
``i2as/session/gateway/README.md`` for the surface this publishes, and
``GLOSSARY.md`` for the **Reference client** / **Request spool** vocabulary.
"""

from i2as.ctl.client import CtlClient, SpoolEngine, open_client
from i2as.ctl.cli import build_parser, main

__all__ = [
    "CtlClient",
    "SpoolEngine",
    "open_client",
    "build_parser",
    "main",
]
