# Ylivertainen

From trauma bay signal to reproducible ML: clinical data pipelines built for neuro-focused questions.

Clinical tabular data toolkit: cleaning, cohort definition, descriptive data analysis (DDA), exploratory data analysis (EDA), feature decisions, and model-frame construction for binary classification tasks built on EMS–hospital stroke-family agreement labels.

## Clinical question

Given prehospital (EMS) and hospital diagnosis fields, can we define cohorts and targets that reflect **agreement** between EMS impression and discharge diagnosis (e.g. TIA, ischemic stroke, broader cerebrovascular match)? The pipeline turns raw CSVs into a reproducible `(X, y)` model frame for supervised learning—not clinical decision support.

## Golden path (single documented flow)

Run the canonical notebook top to bottom after a clean kernel:

1. **Setup** — resolve project paths; ensure the directory **containing** the `ylivertainen` package is on `PYTHONPATH` (see [Running](#running)).
2. **Cleaning** — merge raw CSVs, schema, derived columns, optional duplicate handling.
3. **Cohort** — inclusion rules, unified `target`, `metadata`.
4. **DDA** — numerical / categorical / binary summaries; optional table exports under `ylivertainen/reports/tables/`.
5. **EDA** — whitelist predictors, associations, feature-decision table; optional exports.
6. **Model frame** — `build_model_frame` → **`(X, y)`** and optional pickle under `ylivertainen/data/processed/`.

Canonical notebook:

- `ylivertainen/notebooks/YLIVERTAINEN_CraniotomyForBadData.ipynb`

## Stroke-family task registry

**Single source of truth:** all modeling tasks are defined in `ylivertainen/config.py` as frozen `@dataclass` instances of `TaskConfig`. The notebook and pipeline take a single `task` object; cohort, metadata, and export filenames use `task.name` and `task.target_column` consistently.

### `TaskConfig` fields

| Field | Role |
|-------|------|
| `name` | Stable string ID for artifacts (e.g. pickles, report names): `tia_match_binary`, … |
| `target_column` | Column in the cleaned dataframe that holds the label; must match derived columns from `schema.py` / cleaning. |
| `positive_class` | Scikit-learn / metrics: which value counts as the positive class (here `True` for agreement flags). |
| `task_type` | `"binary"` (current stroke-family tasks). |
| `inclusion_criteria` | Cohort rules before modeling (e.g. require non-missing EMS and discharge diagnosis codes). |

**Example — shape of a task (same structure as the built-ins):**

```python
from ylivertainen.config import TaskConfig

# Illustrative: fields mirror what lives in config.py for TIA_MATCH / others.
example = TaskConfig(
    name="tia_match_binary",
    target_column="TIA_match",
    positive_class=True,
    task_type="binary",
    inclusion_criteria={
        "nmpd_diag": "non-NaN",
        "izrakstisanas_diag": "non-NaN",
    },
)
```

### Naming story (three predefined tasks)

Each **module-level constant** is the public handle; **`name`** is the filesystem-safe slug; **`target_column`** is the clinical/derived label column.

| Constant | `task.name` (artifacts) | `target_column` (label in data) | Clinical idea |
|----------|-------------------------|----------------------------------|---------------|
| `TIA_MATCH` | `tia_match_binary` | `TIA_match` | EMS vs discharge agreement on TIA pattern (G45.*). |
| `ISCHEMIC_STROKE_MATCH` | `ischemic_stroke_match_binary` | `ischemic_match` | Agreement on ischemic stroke / related codes (I63, I64). |
| `ANY_CEREBROVASCULAR_MATCH` | `any_cerebrovascular_match_binary` | `any_cerebrovascular_match` | Broader cerebrovascular agreement (I6*, G45.*). |

**Default for demos and README examples:** use **`TIA_MATCH`** unless you are comparing tasks explicitly.

```python
from ylivertainen.config import TIA_MATCH, ISCHEMIC_STROKE_MATCH, ANY_CEREBROVASCULAR_MATCH

# Default demo / single-task run:
task = TIA_MATCH

# Switch task for comparison runs (one at a time):
# task = ISCHEMIC_STROKE_MATCH
# task = ANY_CEREBROVASCULAR_MATCH
```

## Repository layout

```text
TheLibraryOfCode/
├── .gitignore
├── README.md
├── requirements.txt
└── ylivertainen/
    ├── __init__.py
    ├── config.py
    ├── cleaning.py
    ├── cohort.py
    ├── dda.py
    ├── eda.py
    ├── predictive.py
    ├── schema.py
    ├── columns_to_canonical.py
    ├── aesthetics_helpers.py
    ├── notebooks/
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── reports/
    │   ├── tables/
    │   └── figures/
    └── tests/
```

## Requirements

- **Python** 3.11+ recommended.
- Locked dependency set: **`requirements.txt`** at the repo root (`pip install -r requirements.txt`).
- Core packages: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `ipython`, `pytest` (smoke tests). Uncomment `jupyterlab` in that file if you want the full notebook UI via pip.
- Packaging metadata is available in **`pyproject.toml`** (editable install supported).

## Running

1. Create and activate a virtual environment (e.g. `python -m venv .venv` then `source .venv/bin/activate` on macOS/Linux).

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Imports:** use `from ylivertainen...`. The notebook setup cell adds the **parent** of the `ylivertainen` folder to `sys.path` (the folder that *contains* `ylivertainen/`). Alternatively set `PYTHONPATH` to that folder, or install the package in editable mode:

   ```bash
   pip install -e .
   ```

4. Open `ylivertainen/notebooks/YLIVERTAINEN_CraniotomyForBadData.ipynb` and run all cells from the top.

## Pipeline design notes

### Missingness policy (no separate `classify_missingness` export)

There is **no** standalone `classify_missingness()` in the package. Imputation / flagging **policy** for the feature-decisions table is implemented inside **`YlivertainenEDA.build_feature_decisions_table()`** in `eda.py` (helper `_missing_action`, plus row `notes`), driven by the **`missingness_dict`** you pass from the notebook (keys like `"{col}_missing"` → labels such as `MNAR`, `STRUCTURAL pattern`, etc.). To change rules, edit that method (or build the dict upstream and keep the table logic as the single consumer).

### Datetime columns (no `analyse_datetime` in DDA)

**DDA** does not include a separate `analyse_datetime()` pass. Raw timestamps are mostly **dropped or engineered** in `schema.py` / cleaning (e.g. hour, day-of-week). In **EDA**, columns whitelisted as `predictor_datetime` are listed for review; **target–predictor association** logic skips `datetime` and `text` predictors as unsupported in the current implementation. If you need datetime-specific summaries, add a notebook section or extend `dda.py` later.

### `null_as_feature` and order vs feature decisions

Run **`YlivertainenDataCleaningSurg.null_as_feature()`** (and the rest of cleaning) **before** EDA builds **`build_feature_decisions_table`**, so `*_missing` indicator columns exist on the frame you pass into **`YlivertainenEDA`**. Recommended order: **cleaned dataframe (with null flags) → cohort → DDA → EDA (whitelist, associations, feature decisions) → `build_model_frame` → `(X, y)`**.

### Optional: run summary next to artifacts

You can manually export a short **Markdown** summary (run ID, task name, row counts, paths to pickles) under `ylivertainen/reports/` next to tables/figures. Nothing in the library writes this automatically yet.

## Smoke tests & import check

From the **repository root** (`TheLibraryOfCode/`, the directory that **contains** `ylivertainen/`):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=.         # Windows PowerShell: $env:PYTHONPATH="."
python -c "import ylivertainen as y; print('ok', y.__version__)"
python -m unittest discover -s ylivertainen/tests -p "test_*.py"
# or (recommended for interpreter consistency):
python -m pytest ylivertainen/tests/
```

For a **full** pipeline smoke test, use the canonical notebook on toy or de-identified data (cohort step needs columns that match `TaskConfig` and schema). The public synthetic sample filename is `ylivertainen/data/raw/testtable_synthetic.csv`.

## Git (first-time repo)

```bash
cd /path/to/TheLibraryOfCode
git init
git add README.md requirements.txt pyproject.toml .gitignore ylivertainen/
git status
```

Do **not** commit `.env` or secrets. This repo’s `.gitignore` keeps `ylivertainen/data/raw/**`, `ylivertainen/data/processed/**`, and `ylivertainen/reports/**` ignored by default, while explicitly allowing the single public synthetic sample `ylivertainen/data/raw/testtable_synthetic.csv`.

## Public API (`__init__.py`)

Convenience imports are re-exported from **`ylivertainen`** (see `ylivertainen/__init__.py` and `__all__`). **`__version__`** follows semantic-ish `0.x.y` bumps when you change behavior or cut a release.

Advanced / lower-level modules (not re-exported at package root): `ylivertainen.schema` (`ColSpec`, `SCHEMA`, `DERIVED`), `ylivertainen.columns_to_canonical` (`COLUMN_RENAME_MAP`).

## Data and privacy

- **`.gitignore`** ignores virtualenvs, `__pycache__`, Jupyter checkpoints, and `*.pickle` / `*.pkl` by default, and it keeps `ylivertainen/data/raw/**`, `ylivertainen/data/processed/**`, and `ylivertainen/reports/**` ignored by default (except `ylivertainen/data/raw/testtable_synthetic.csv`).
- Do **not** commit patient-identifying or sensitive clinical data. Use de-identified or synthetic data for demos.
- Paths in documentation are examples; replace with your local layout.
- This project is for **research, education, and portfolio** use. It is **not** validated for clinical decision-making or deployment in care settings.

## License

MIT License (see `LICENSE`).

## Author

Andris Zaguzovs (Andy) — MD in Emergency Medicine, building portfolio-grade clinical data science and ML pipelines while training toward a Neurosurgery future.

Interests: brain trauma first, then broader brain and nervous system analytics, neuroscience, neurosurgery-oriented clinical research, cerebrovascular pathways, and reproducible EHR ML tooling.

GitHub: https://github.com/DrAndrushka
LinkedIn: https://www.linkedin.com/in/andris-zaguzovs-341308373/
