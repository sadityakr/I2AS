"""facts_only — the pre-analysis notebook entry, kept as a selectable recipe.

Before the analysis stage existed, a finished run went into the notebook as
its fact tables (parameters, per-column statistics, setup) plus the raw HDF5
file. That entry is still the right one for a run nobody wants analysed — a
cooldown, a diagnostic, a run whose analysis lives elsewhere — so it stays
available as a recipe rather than as a special case in the publisher.

It draws nothing and derives nothing. It sets ``include_fact_tables`` and
``attach_data_file``, which is exactly the instruction "render the run's own
facts below this and attach the file", and writes one sentence saying so.
"""

from __future__ import annotations

import logging

from i2as.analysis.base import AnalysisContext, AnalysisRecipe
from i2as.analysis.report import ANY_PROCEDURE, AnalysisReport
from i2as.core.data_reader import RunSource

logger = logging.getLogger(__name__)


class FactsOnlyRecipe(AnalysisRecipe):
    """No figure, no derived value — the run's own fact tables and its data file."""

    name = "facts_only"
    procedures = (ANY_PROCEDURE,)
    description = "No analysis: the run's fact tables and its data file, as the notebook entry"

    def analyse(self, run: RunSource, context: AnalysisContext) -> AnalysisReport:
        """Return the facts-only report for one run.

        Args:
            run: The finished run — read only for its point count, so the
                sentence says how much was measured.
            context: The manifest and the experiment facts.

        Returns:
            An ``ok`` report with one summary sentence, no figure, no table,
            and both fact-table and data-file flags set.
        """
        procedure = str(context.manifest.get("procedure") or "run")
        status = str(context.manifest.get("status") or "unknown")
        summary = (
            f"{procedure} recorded {run.n_points} point(s) and ended with status "
            f"{status!r}; the run's own parameter, statistics and setup tables "
            f"follow, and the data file is attached."
        )
        return AnalysisReport(
            summary=(summary,),
            include_fact_tables=True,
            attach_data_file=True,
        )
