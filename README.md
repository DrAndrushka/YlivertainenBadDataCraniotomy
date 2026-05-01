# Ylivertainen 0.2.0 🧠⚡

From trauma-bay chaos to reproducible ML.  
`Ylivertainen` is a clinical tabular toolkit for stroke-family agreement tasks, built for clean pipelines, auditable decisions, and portfolio-grade outputs.

> **Scope:** research, education, and ML workflow engineering.  
> **Not clinical decision support.**

---

## Why this exists 🏥

Clinical tabular projects usually fail in three places: messy schema, unclear cohort logic, and undocumented feature decisions.  
This repo solves those in one flow:

- 🧹 cleaning + harmonization
- 🚧 cohort definition by task
- 📊 DDA + EDA
- 🧪 inferential testing
- 🔮 predictive modeling
- 🧾 HTML reporting

Core clinical question:

> Can prehospital diagnosis signal and discharge diagnosis be converted into reproducible agreement labels (TIA / ischemic / broader cerebrovascular), then into a clean `(X, y)` model frame?

---

## What you get 🔧

- **Single task registry** in `ylivertainen/config.py` using frozen `TaskConfig`
- **Unified pipeline stages** from raw tabular fields to model-ready data
- **Feature-decision layer** that tracks missingness, leakage, redundancy, and keep/drop rationale
- **Notebook-first golden path** plus importable Python modules
- **Release-ready package metadata** (`pyproject.toml`, top-level API exports)

---

## Repository layout 🗂️

```text
YlivertainenBadDataCraniotomy/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── pyproject.toml
└── ylivertainen/
    ├── __init__.py
    ├── aesthetics_helpers.py
    ├── cleaning.py
    ├── cohort.py
    ├── config.py
    ├── dda.py
    ├── eda.py
    ├── predictive_modeling.py
    ├── inferential.py
    ├── predictive.py
    ├── the_report.py
    ├── schema.py          # ColSpec, SCHEMA, DERIVED, COLUMN_RENAME_MAP
    ├── notebooks/
    │   └── YLIVERTAINEN_CraniotomyForBadData.ipynb
    ├── tests/
    │   └── test_cleaning.py
    ├── reports/
    └── example_table/
        └── testtable_synthetic.csv
```

Note: `COLUMN_RENAME_MAP` is defined in `ylivertainen/schema.py`.

---

## Built-in tasks 🎯

Defined in `ylivertainen/config.py`:

- `TIA_MATCH`
- `ISCHEMIC_STROKE_MATCH`
- `ANY_CEREBROVASCULAR_MATCH`

Each `TaskConfig` carries:

- `name`
- `target_column`
- `positive_class`
- `task_type`
- `inclusion_criteria`

---

## Public API 🧪

Top-level imports are exposed via `ylivertainen/__init__.py`:

- **Config/tasks:** `TaskConfig`, `TIA_MATCH`, `ISCHEMIC_STROKE_MATCH`, `ANY_CEREBROVASCULAR_MATCH`
- **Cleaning:** `YlivertainenDataCleaningSurg`, `pre_merge_check`
- **Cohort:** `apply_inclusion_criteria`, `build_stroke_agreement_cohort`
- **DDA/EDA:** `YlivertainenDDA`, `YlivertainenEDA`
- **Model frame:** `build_model_frame`
- **Inferential:** `YlivertainenInferential`
- **Predictive:** `YlivertainenPredictive`
- **Reporting:** `YlivertainenTheReport`

Quick sanity check:

```python
import ylivertainen as y
print(y.__version__)  # 0.2.0
```

---

## Quick start 🚀

Python 3.12+ required.

### Windows (PowerShell)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## Golden notebook flow 📓

Run:

- `ylivertainen/notebooks/CraniotomyForBadData.ipynb`

Recommended sequence:

1. Setup and path resolution
2. Cleaning and derivations
3. Cohort creation from selected `TaskConfig`
4. DDA summaries
5. EDA + feature decisions
6. `(X, y)` creation via `build_model_frame`
7. Optional inferential / predictive / report generation

---

## Testing ✅

From repo root:

```bash
python -m pytest ylivertainen/tests/
# fallback
python -m unittest discover -s ylivertainen/tests -p "test_*.py"
```

---

## Data + privacy 🔒

- Real/sensitive data and generated artifacts are ignored by default in `.gitignore`
- Public synthetic sample folder is intentionally whitelisted:
  - `ylivertainen/example_table/`
- Do not commit patient-identifying information

---

## License 📜

MIT License. See `LICENSE`.
