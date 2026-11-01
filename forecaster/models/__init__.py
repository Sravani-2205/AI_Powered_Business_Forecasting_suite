"""
The model registry: which models exist, grouped by family.

The competition asks for the families allowed by the data-size tier, and this file
hands back fresh model objects for them. Families that are not built yet return an
empty list, so the pipeline still runs while we add models step by step.

Each builder receives the frequency and the festival calendar. Models that do not use
festivals simply ignore the calendar.
"""

from forecaster.models.baselines import baseline_models
from forecaster.models.deep_learning import deep_learning_models
from forecaster.models.machine_learning import machine_learning_models
from forecaster.models.prophet_model import prophet_models
from forecaster.models.statistical import statistical_models

BUILDERS = {
    "baselines": lambda frequency, calendar: baseline_models(frequency),
    "statistical": lambda frequency, calendar: statistical_models(frequency),
    "prophet": prophet_models,
    "machine_learning": machine_learning_models,
    "deep_learning": deep_learning_models,
}


def models_for(families: list, frequency: str, calendar=None) -> tuple[list, list]:
    """Return (models to run, families that are allowed but not built yet)."""
    models, not_built = [], []
    for family in families:
        if family in BUILDERS:
            models.extend(BUILDERS[family](frequency, calendar))
        else:
            not_built.append(family)
    return models, not_built
