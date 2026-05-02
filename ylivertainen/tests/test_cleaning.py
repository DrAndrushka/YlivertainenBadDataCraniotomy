"""
Pytest suite for `ylivertainen.cleaning` with priority-based coverage.

Current status:
- Block 3 (HIGH priority) is implemented and green.
- MEDIUM/LOW priority sections are planned and scaffolded below.

Structure of this file:
1) Shared fixtures
2) Implemented test sections (HIGH)
3) Planned test sections with TODO checklists and code-space placeholders

Design note:
- Deliberate double-cover exists for "empty id list does nothing":
  - helper-level contract: `test_empty_id_returns_empty_masks`
  - public API behavior: `test_empty_id_returns_self_and_keeps_row_count`
"""

from pandas.errors import Pandas4Warning
from ylivertainen._pathing import setup_repo_path
root = setup_repo_path()

import pandas as pd
import pytest
import re

from ylivertainen.cleaning import YlivertainenDataCleaningSurg


# ==========================================================
# Roadmap (file-level TODO)
# ==========================================================
# [DONE] HIGH: _resolve_duplicate_masks
# [DONE] HIGH: resolve_dupes
# [DONE] HIGH: merge_dfs core tests
# [TODO] HIGH/REFINE: parametrize merge_dfs scenario trio (shared Arrange pattern)
#
# [TODO] MEDIUM: apply_schema
# [TODO] MEDIUM: apply_derived
# [TODO] MEDIUM: cleanup
# [TODO] MEDIUM: apply_nan_features
#
# [TODO] LOW: __str__
# [TODO] LOW: pre_merge_check (smoke)
# [TODO] LOW: ylivertainen_janitor (smoke)
# [TODO] LOW: explore_values (smoke)
# [TODO] LOW: cleaning_overview_and_commit (smoke)

# ==========================================================
# Shared fixture(s)
# ==========================================================
@pytest.fixture
def make_project():
    def _make_project(df: pd.DataFrame) -> YlivertainenDataCleaningSurg:
        project = YlivertainenDataCleaningSurg.__new__(YlivertainenDataCleaningSurg)
        project.csvs = []
        project.df = df.copy()
        return project
    return _make_project

@pytest.fixture
def synthetic_df():
    location = root/ "ylivertainen" / "example_table" / "testtable_synthetic.csv"
    synthetic_df = YlivertainenDataCleaningSurg([location]).df.iloc[[0]].copy()
    return synthetic_df

# ==========================================================
# Function: _resolve_duplicate_masks (HIGH)
# ==========================================================
# [X] empty id list returns empty masks
# [X] missing id column raises ValueError
# [X] whitespace/case normalization finds duplicates
# [X] incomplete IDs are ignored (not counted as dupes)
#
# Note: `test_empty_id_returns_empty_masks` is intentionally paired with
# `test_empty_id_returns_self_and_keeps_row_count` below. This is deliberate
# double-cover at two levels:
# - private helper contract (`_resolve_duplicate_masks`)
# - public API behavior (`resolve_dupes`)

# ---------------------------
# Implemented tests
# ---------------------------
def test_empty_id_returns_empty_masks(make_project):
    project = make_project(pd.DataFrame({"patient_card_no": ["123", "123"]}))
    id_cols, skipsfirst_dupe_mask, includesfirst_dupe_mask = project._resolve_duplicate_masks(None)
    assert id_cols == []
    assert skipsfirst_dupe_mask.sum() == 0
    assert includesfirst_dupe_mask.sum() == 0

def test_missing_id_col_raises_value_error(make_project):
    project = make_project(pd.DataFrame({"patient_card_no": ["123", "123"]}))
    with pytest.raises(ValueError, match="ID columns not found"):
        project._resolve_duplicate_masks("nonexisting_id_col")

def test_normalization_finds_duplicates(make_project):
    project = make_project(pd.DataFrame({"patient_card_no": ["  YlivertAINen ", " YlIvErTaInEn    "]}))
    _id_cols, skipsfirst_dupe_mask, includesfirst_dupe_mask = project._resolve_duplicate_masks("patient_card_no")
    assert skipsfirst_dupe_mask.sum() == 1
    assert includesfirst_dupe_mask.sum() == 2

def test_incomplete_ids_not_counted_as_dupes(make_project):
    project = make_project(pd.DataFrame({"patient_card_no": ["yliver", "ylivertainen", "", " ", "something else", pd.NA, "yliver"]}))
    _id_cols, skipsfirst_dupe_mask, includesfirst_dupe_mask = project._resolve_duplicate_masks("patient_card_no")
    assert skipsfirst_dupe_mask.sum() == 1
    assert includesfirst_dupe_mask.sum() == 2

# ==========================================================
# Function: resolve_dupes (HIGH)
# ==========================================================
# [X] include_first=True returns full duplicate groups
# [X] include_first=False returns only later duplicates
# [X] drop=True removes only later duplicates, keeps first rows
# [X] empty id list returns self and keeps row count
# [X] missing id column raises clear ValueError

def test_include_first_true_returns_duplicate_groups(make_project):
    project = make_project(pd.DataFrame({"patient_card_no": ["123", "123"]}))
    _id_cols, _skipsfirst_dupe_mask, includesfirst_dupe_mask = project._resolve_duplicate_masks("patient_card_no")
    assert len(project.resolve_dupes(
        id_cols="patient_card_no",
        include_first=True,
        drop=False
        ).df[includesfirst_dupe_mask]) == 2

def test_include_first_false_returns_later_duplicates(make_project):
    project = make_project(pd.DataFrame({"patient_card_no": ["123", "123"]}))
    _id_cols, skipsfirst_dupe_mask, _includesfirst_dupe_mask = project._resolve_duplicate_masks("patient_card_no")
    assert len(project.resolve_dupes(
        id_cols="patient_card_no",
        include_first=False,
        drop=False
        ).df[skipsfirst_dupe_mask]) == 1

def test_drop_true_removes_only_later_duplicates(make_project):
    project = make_project(pd.DataFrame({"patient_card_no": ["123", "123"]}))
    assert len(project.resolve_dupes(
        id_cols="patient_card_no",
        include_first=False,
        drop=True
        ).df) == 1

def test_empty_id_returns_self_and_keeps_row_count(make_project):
    project = make_project(pd.DataFrame({"patient_card_no": ["123", "123"]}))
    before = project.df.copy(deep=True)
    returned = project.resolve_dupes(
        id_cols=None,
        include_first=False,
        drop=False)
    assert returned is project
    pd.testing.assert_frame_equal(before, project.df)

def test_missing_id_column_raises_value_error(make_project):
    project = make_project(pd.DataFrame({"patient_card_no": ["123", "123"]}))
    with pytest.raises(ValueError, match=re.escape("❌ ID columns not found: ['clear_which_column']")):
        project.resolve_dupes(id_cols="clear_which_column")    

# ==========================================================
# Function: merge_dfs (HIGH)
# ==========================================================
# [X] empty csv list raises ValueError
# [X] unknown columns dropped
# [X] alias columns renamed to canonical names
# [X] missing canonical columns become NaN via reindex (no KeyError)
#
# TODO: parametrize candidate
# `test_unknown_columns_dropped`, `test_alias_columns_renames_to_canonical_names`,
# and `test_missing_canonical_columns_become_nan_via_reindex` share the same Arrange pattern
# (build input_df -> write csv via tmp_path -> call merge_dfs). Refactor with parametrize later.

# --- Write merge_dfs tests below ---
def test_empty_csv_list_raises_value_error():
    with pytest.raises(ValueError, match=re.escape("❌ No CSV file/-s provided to merge_dfs")):
        YlivertainenDataCleaningSurg.merge_dfs([])    

@pytest.mark.parametrize(
    "input_df, check", [
        (pd.DataFrame({
            "Karte": ["W-001", "W-002"],
            "commentarios": ["Witcher", "Sorceress"],
            "hospital of admission": ["Kaer Morhen", "Aretuza"],
            "Bloede pest": ["Catriona", "Nilfgaard plague"]}),
        lambda df: "commentarios" not in df.columns),
        (pd.DataFrame({
            "Karte": ["W-001", "W-002"],
            "Bloede pest": ["Catriona", "Nilfgaard plague"]}),
        lambda df: df["patient_card_no"].tolist() == ["W-001", "W-002"]),
        (pd.DataFrame({
            "Kartuka": ["W-001", "W-002"],
            "Bloede pest": ["Catriona", "Nilfgaard plague"]}),
        lambda df: df["patient_card_no"].isna().all())
    ], ids=["unknown_dropped", "alias_renamed", "missing_becomes_nan"]
)

def test_merge_dfs_column_behaviour(tmp_path, input_df, check):
    csv_path = tmp_path / "witcher.csv"
    input_df.to_csv(csv_path, index=False)
    result_df = YlivertainenDataCleaningSurg.merge_dfs([csv_path])
    assert check(result_df)

# ==========================================================
# Planned sections (MEDIUM/LOW): TODO + code spaces
# ==========================================================

# ==========================================================
# Function: apply_schema (MEDIUM)
# ==========================================================
# [ ] configured null replacements are applied
# [ ] numeric conversion coerces invalid values to NaN
# [ ] invalid kind in SCHEMA raises ValueError

# --- Code space: apply_schema tests ---
import numpy as np

def test_apply_schema_null_replacement_works(make_project, synthetic_df):
    project = make_project(synthetic_df)
    with pytest.warns(Pandas4Warning, match="Constructing a Categorical"):
        project = project.apply_schema()
    project = project.apply_schema()
    assert pd.isna(project.df["GKS"].iloc[0])       # works for categorical
    assert pd.isna(project.df["vecums"].iloc[0])    # works for numerical






# ==========================================================
# Function: apply_derived (MEDIUM)
# ==========================================================
# [ ] match branch creates boolean agreement column
# [ ] datetime branch rejects unknown datetime unit
# [ ] timedelta branch requires datetime source columns
# [ ] negative timedeltas converted to NA

# --- Code space: apply_derived tests ---



# ==========================================================
# Function: cleanup (MEDIUM)
# ==========================================================
# [ ] drops columns where SCHEMA keep=False

# --- Code space: cleanup tests ---



# ==========================================================
# Function: apply_nan_features (MEDIUM)
# ==========================================================
# [ ] creates *_missing flags only for columns with NaNs
# [ ] excludes derived columns from *_missing creation

# --- Code space: apply_nan_features tests ---


# ==========================================================
# ==========================================================
# Function: __str__ (LOW)
# ==========================================================
# [ ] returns string containing csv count and dataframe shape

# --- Code space: __str__ tests ---



# ==========================================================
# Function: pre_merge_check (LOW)
# ==========================================================
# [ ] smoke test with tiny temp CSVs (no crash)

# --- Code space: pre_merge_check tests ---



# Function: ylivertainen_janitor (LOW)
# ==========================================================
# [ ] apply_all=False path works end-to-end
# [ ] apply_all=True path works end-to-end

# --- Code space: ylivertainen_janitor tests ---



# ==========================================================
# Function: explore_values (LOW)
# ==========================================================
# [ ] smoke test with mixed dtypes (no crash)

# --- Code space: explore_values tests ---



# ==========================================================
# Function: cleaning_overview_and_commit (LOW)
# ==========================================================
# [ ] returns dataframe object
# [ ] smoke test prints summary without crashing

# --- Code space: cleaning_overview_and_commit tests ---
