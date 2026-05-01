"""
Test workspace for `ylivertainen/cleaning.py`.

This file is intentionally structured as:
1) function-by-function TODO checklist
2) coding space to implement tests right below each TODO block
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
    id_cols, skipsfirst_dupe_mask, includesfirst_dupe_mask = project._resolve_duplicate_masks("patient_card_no")
    assert skipsfirst_dupe_mask.sum() == 1
    assert includesfirst_dupe_mask.sum() == 2

def test_incomplete_ids_not_counted_as_dupes(make_project):
    project = make_project(pd.DataFrame({"patient_card_no": ["yliver", "ylivertainen"]}))
    id_cols, skipsfirst_dupe_mask, includesfirst_dupe_mask = project._resolve_duplicate_masks("patient_card_no")
    assert skipsfirst_dupe_mask.sum() == 0
    assert includesfirst_dupe_mask.sum() == 0

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
    _, __, includesfirst_dupe_mask = project._resolve_duplicate_masks("patient_card_no")
    assert len(project.resolve_dupes(
        id_cols="patient_card_no",
        include_first=True,
        drop=False
        ).df[includesfirst_dupe_mask]) == 2

def test_include_first_false_returns_later_duplicates(make_project):
    project = make_project(pd.DataFrame({"patient_card_no": ["123", "123"]}))
    _, skipsfirst_dupe_mask, __ = project._resolve_duplicate_masks("patient_card_no")
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
    returned = project.resolve_dupes(
        id_cols=None,
        include_first=False,
        drop=False)
    assert returned is project
    pd.testing.assert_frame_equal(project.df, returned.df)

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
def test_empty_csv_list_raises_value_error(make_project):
    project = make_project(pd.DataFrame({"patient_card_no": ["123", "123"]}))
    with pytest.raises(ValueError, match=re.escape("❌ No CSV file/-s provided to merge_dfs")):
        project.merge_dfs(csvs=project.csvs)    

#def test_unknown_columns_dropped(make_project):


#def test_alias_columns_renames_to_canonical_names(make_project):


#def test_missing_canonical_columns_become_nan_via_reindex(make_project):


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