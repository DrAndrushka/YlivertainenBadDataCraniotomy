"""
Ylivertainen — clinical tabular pipeline: cleaning → cohort → DDA → EDA → model frame.

Public symbols are re-exported below; see README for golden path, task registry, and
pipeline notes (missingness policy in ``build_feature_decisions_table``, datetime handling,
order of ``null_as_feature`` vs feature decisions).

Lower-level: ``ylivertainen.schema`` (ColSpec, SCHEMA), ``ylivertainen.columns_to_canonical``.
"""

__version__ = "0.1.0"

from ylivertainen.config import (
    ANY_CEREBROVASCULAR_MATCH,
    ISCHEMIC_STROKE_MATCH,
    TIA_MATCH,
    TaskConfig,
    big_beautiful_print,
)
from ylivertainen.cleaning import YlivertainenDataCleaningSurg, pre_merge_check
from ylivertainen.cohort import apply_inclusion_criteria, build_stroke_agreement_cohort
from ylivertainen.dda import YlivertainenDDA
from ylivertainen.eda import YlivertainenEDA
from ylivertainen.predictive import build_model_frame, export_model_frame

__all__ = [
    "__version__",
    "ANY_CEREBROVASCULAR_MATCH",
    "ISCHEMIC_STROKE_MATCH",
    "TIA_MATCH",
    "TaskConfig",
    "YlivertainenDDA",
    "YlivertainenDataCleaningSurg",
    "YlivertainenEDA",
    "apply_inclusion_criteria",
    "big_beautiful_print",
    "build_model_frame",
    "build_stroke_agreement_cohort",
    "export_model_frame",
    "pre_merge_check",
]
