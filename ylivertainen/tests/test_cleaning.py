"""
Pytest suite for `ylivertainen.cleaning`.

This module verifies high-priority cleaning behavior first
(`_resolve_duplicate_masks`, `resolve_dupes`, `merge_dfs`),
then expands toward medium/low-priority coverage.

Tests use small synthetic inputs and explicit assertions so
transformations remain deterministic and easy to debug.
"""

import pandas as pd
import pytest
import re

from ylivertainen.cleaning import YlivertainenDataCleaningSurg


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

# ==========================================================
# Function: _resolve_duplicate_masks (HIGH)
# ==========================================================
# [X] empty id list returns empty masks
# [X] missing id column raises ValueError
# [X] whitespace/case normalization finds duplicates
# [X] incomplete IDs are ignored (not counted as dupes)

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
    project = make_project(pd.DataFrame({"patient_card_no": ["YlivertAINen", "YlIvErTaInEn"]}))
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
# [ ] unknown columns dropped
# [ ] alias columns renamed to canonical names
# [ ] missing canonical columns become NaN via reindex (no KeyError)

# --- Write merge_dfs tests below ---
def test_empty_csv_list_raises_value_error():
    with pytest.raises(ValueError, match=re.escape("❌ No CSV file/-s provided to merge_dfs")):
        YlivertainenDataCleaningSurg.merge_dfs([])    

def test_unknown_columns_dropped(tmp_path):
    input_df = pd.DataFrame({
        "Karte": ["W-001", "W-002"],
        "commentarios": ["Witcher", "Sorceress"],
        "hospital of admission": ["Kaer Morhen", "Aretuza"],
        "Bloede pest": ["Catriona", "Nilfgaard plague"]})
    csv_path = tmp_path / "witcher.csv"
    input_df.to_csv(csv_path, index=False)
    result_df = YlivertainenDataCleaningSurg.merge_dfs([csv_path])
    assert "commentarios" not in result_df.columns
    assert "hospital of admission" not in result_df.columns
    assert "Bloede pest" not in result_df.columns

def test_alias_columns_renames_to_canonical_names(tmp_path):
    input_df = pd.DataFrame({
        "Karte": ["W-001", "W-002"],
        "Bloede pest": ["Catriona", "Nilfgaard plague"]})
    csv_path = tmp_path / "witcher.csv"
    input_df.to_csv(csv_path, index=False)
    result_df = YlivertainenDataCleaningSurg.merge_dfs([csv_path])
    assert "patient_card_no" in result_df.columns
    assert result_df["patient_card_no"].tolist() == ["W-001", "W-002"]

def test_missing_canonical_columns_become_nan_via_reindex(tmp_path):
    input_df = pd.DataFrame({
        "Kartuka": ["W-001", "W-002"],
        "Bloede pest": ["Catriona", "Nilfgaard plague"]})
    csv_path = tmp_path / "witcher.csv"
    input_df.to_csv(csv_path, index=False)
    result_df = YlivertainenDataCleaningSurg.merge_dfs([csv_path])
    assert "patient_card_no" in result_df.columns
    assert result_df["patient_card_no"].isna().all()

# ==========================================================
# Function: apply_schema (MEDIUM)
# ==========================================================
# [ ] configured null replacements are applied
# [ ] numeric conversion coerces invalid values to NaN
# [ ] invalid kind in SCHEMA raises ValueError


# --- Write apply_schema tests below ---



# ==========================================================
# Function: apply_derived (MEDIUM)
# ==========================================================
# [ ] match branch creates boolean agreement column
# [ ] datetime branch rejects unknown datetime unit
# [ ] timedelta branch requires datetime source columns
# [ ] negative timedeltas converted to NA

# --- Write apply_derived tests below ---



# ==========================================================
# Function: cleanup (MEDIUM)
# ==========================================================
# [ ] drops columns where SCHEMA keep=False

# --- Write cleanup tests below ---



# ==========================================================
# Function: apply_nan_features (MEDIUM)
# ==========================================================
# [ ] creates *_missing flags only for columns with NaNs
# [ ] excludes derived columns from *_missing creation

# --- Write apply_nan_features tests below ---


# ==========================================================
# ==========================================================
# Function: __str__ (LOW)
# ==========================================================
# [ ] returns string containing csv count and dataframe shape

# --- Write __str__ tests below ---



# ==========================================================
# Function: pre_merge_check (LOW)
# ==========================================================
# [ ] smoke test with tiny temp CSVs (no crash)

# --- Write pre_merge_check tests below ---



# Function: ylivertainen_janitor (LOW)
# ==========================================================
# [ ] apply_all=False path works end-to-end
# [ ] apply_all=True path works end-to-end

# --- Write ylivertainen_janitor tests below ---



# ==========================================================
# Function: explore_values (LOW)
# ==========================================================
# [ ] smoke test with mixed dtypes (no crash)

# --- Write explore_values tests below ---



# ==========================================================
# Function: cleaning_overview_and_commit (LOW)
# ==========================================================
# [ ] returns dataframe object
# [ ] smoke test prints summary without crashing

# --- Write cleaning_overview_and_commit tests below ---