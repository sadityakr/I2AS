"""Entry point: ``python -m i2as.troubleshoot <subcommand> ...``."""

import sys

from i2as.troubleshoot.cli import main

sys.exit(main())
