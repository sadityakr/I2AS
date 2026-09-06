"""The recipes shipped with I2AS — one module, one recipe.

Every module here is imported by ``i2as.analysis.discovery`` and every
``AnalysisRecipe`` subclass declaring a ``name`` becomes selectable, exactly
as a driver, a virtual instrument or a procedure does: adding a recipe means
adding one file and changing nothing else. The analysis conformance tests
cover a new module the moment it exists.

What belongs here is a recipe that is useful for a whole FAMILY of runs and
carries no experiment-specific physics — ``generic_sweep`` (an overview of any
run at all) and ``facts_only`` (the pre-analysis notebook entry, selectable).
An analysis that only one experiment wants belongs in that experiment's own
``analysis/recipes/`` folder instead, where it overrides a shipped recipe of
the same ``name``.
"""
