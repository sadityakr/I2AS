"""Entry point: ``python -m i2as.ctl [connection options] <subcommand>``."""

import sys

from i2as.ctl.cli import main

sys.exit(main())
