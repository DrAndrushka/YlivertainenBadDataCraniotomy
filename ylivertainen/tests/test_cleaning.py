"""
Test suite for `ylivertainen.cleaning`.

Layout:
1) Shared fixtures
2) Core behavior tests by method
3) End-to-end and smoke tests

Design note:
- Deliberate double-cover exists for "empty id list does nothing":
  - helper-level contract: `test_empty_id_returns_empty_masks`
  - public API behavior: `test_empty_id_returns_self_and_keeps_row_count`
"""

from pathlib import Path
from ylivertainen._pathing import setup_repo_path
root = setup_repo_path()

import pandas as pd
import pytest
import re
import shutil

from ylivertainen.cleaning import YlivertainenDataCleaningSurg, pre_merge_check
import ylivertainen.cleaning as cleaning
from ylivertainen.schema import ColSpec

# ==========================================================
# Shared Fixtures
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
    location = root / "ylivertainen" / "example_table" / "testtable_synthetic.csv"
    synthetic_df = YlivertainenDataCleaningSurg([location]).df.iloc[:10,:].copy()      # first 10 rows, all columns
    return synthetic_df

# ==========================================================
# _resolve_duplicate_masks
# ==========================================================
# Covers empty IDs, missing ID columns, normalization, and incomplete IDs.
#
# Note: `test_empty_id_returns_empty_masks` is intentionally paired with
# `test_empty_id_returns_self_and_keeps_row_count` below. This is deliberate
# double-cover at two levels:
# - private helper contract (`_resolve_duplicate_masks`)
# - public API behavior (`resolve_dupes`)

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
# resolve_dupes
# ==========================================================
# Covers include/drop behavior, empty IDs, and missing-column errors.

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
# merge_dfs
# ==========================================================
# Covers empty input, unknown-column dropping, alias renaming, and canonical
# column reindexing behavior.

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
    ], ids=["unknown_dropped", "alias_renamed", "missing_becomes_nan"])

def test_merge_dfs_column_behaviour(tmp_path, input_df, check):
    csv_path = tmp_path / "witcher.csv"
    input_df.to_csv(csv_path, index=False)
    result_df = YlivertainenDataCleaningSurg.merge_dfs([csv_path])
    assert check(result_df)

# ==========================================================
# apply_schema
# ==========================================================
# Covers null replacement, numeric coercion, and invalid SCHEMA kinds.

def test_apply_schema_null_replacement_works(make_project, synthetic_df):
    project = make_project(synthetic_df)
    project = project.apply_schema()
    assert pd.isna(project.df["GKS"].iloc[0])       # works for categorical
    assert pd.isna(project.df["vecums"].iloc[0])    # works for numerical

def test_apply_schema_invalid_kind_value_error(make_project, synthetic_df, monkeypatch):
    bad_schema = list(cleaning.SCHEMA)
    bad_schema[-1] = ColSpec(name="patient_card_no", kind="not_a_real_kind")
    monkeypatch.setattr(cleaning, "SCHEMA", bad_schema)
    project = make_project(synthetic_df.copy(deep=True))
    with pytest.raises(ValueError, match="No such dtype"):
        project.apply_schema()

# ==========================================================
# apply_derived
# ==========================================================
# Covers match column creation, datetime coercion visibility, timedelta type
# validation, and negative timedelta handling.

def test_apply_derived_creates_boolean_match_column(make_project, synthetic_df):
    project = make_project(synthetic_df)
    project = project.apply_schema().apply_derived()
    match_cols_expected = 0
    for ColSpec in cleaning.DERIVED:
        if ColSpec.kind == "match":
            match_cols_expected += 1
    match_cols_made = [col for col in project.df.columns if col.endswith("_match")]
    assert match_cols_expected == len(match_cols_made)

def test_apply_derived_datetime_branch_shows_rejected_values(make_project, synthetic_df, capsys):
    project = make_project(synthetic_df)
    project = project.apply_schema().apply_derived()
    out = capsys.readouterr().out
    out = re.sub(r"\x1b\[[0-9;]*m", "", out)
    assert pd.isna(project.df["lidzPSKUS_timedelta_minutes"].iloc[3])
    assert """izsaukuma_laiks converted to "datetime" | Values that became NaN: 1 | List: ['thirty first december 2025']""" in out

def test_timedelta_branch_requires_datetime_source_cols(make_project, synthetic_df, monkeypatch):
    bad_derived = list(cleaning.DERIVED)
    bad_derived[-1] = ColSpec(
        name="lidzPSKUS_timedelta_minutes",
        kind="timedelta",
        timedelta_units='minutes',
        derive_from=("FastTest", "nogadasana_PSKUS_laiks"),
        keep=False)
    monkeypatch.setattr(cleaning, "DERIVED", bad_derived)
    project = make_project(synthetic_df.copy(deep=True))
    with pytest.raises(ValueError, match="Derivable columns must be datetime"):
        project.apply_schema().apply_derived()

def test_negative_timedeltas_converted_to_na(make_project, synthetic_df, capsys):
    project = make_project(synthetic_df)
    project = project.apply_schema().apply_derived()
    out = capsys.readouterr().out
    out = re.sub(r"\x1b\[[0-9;]*m", "", out)
    assert """also trashed NEGATIVE timedeltas: 1""" in out

# ==========================================================
# cleanup
# ==========================================================
# Verifies dropping columns where SCHEMA `keep=False`.

def test_cleanup_drops_cols_where_keep_is_false(make_project, synthetic_df):
    project = make_project(synthetic_df)
    project = project.apply_schema().apply_derived().cleanup()
    assert 'izsaukuma_laiks' not in project.df.columns

# ==========================================================
# apply_nan_features
# ==========================================================
# Verifies `_missing` feature generation for SCHEMA columns with NaNs and
# excludes DERIVED columns.

def test_apply_nan_features_behaviour(make_project, synthetic_df):
    project = make_project(synthetic_df)
    project = project.apply_schema()
    project = project.apply_derived().apply_nan_features()
    expected_nan_cols = [f"{ColSpec.name}_missing" for ColSpec in cleaning.SCHEMA if project.df[ColSpec.name].isna().sum() > 0]
    real_nan_cols = [col for col in project.df if col.endswith("_missing")]
    derived_cols = [ColSpec.name for ColSpec in cleaning.DERIVED if f"{ColSpec.name}_missing" in real_nan_cols]
    assert set(expected_nan_cols) == set(real_nan_cols)         # creates only from columns with NaNs
    assert len(derived_cols) == 0                                         # creates only from SCHEMA

# ==========================================================
# __str__
# ==========================================================
# Verifies string output includes CSV and shape details.

def test_string_when_constructing_ylivertainendatacleaningsurg(make_project, synthetic_df):
    project = make_project(synthetic_df)
    text = str(project)
    assert """CSVs:""" in text
    assert """rows, """ in text
    assert """columns""" in text

# ==========================================================
# pre_merge_check
# ==========================================================
# Smoke test to ensure pre-merge scan runs without crashing.

def test_pre_merge_check_smoke(tmp_path, monkeypatch):
    test_table_location = root / "ylivertainen" / "example_table" / "testtable_synthetic.csv"
    tmp_location = tmp_path / "ylivertainen" / "data" / "raw"
    tmp_location.mkdir(parents=True, exist_ok=True)
    shutil.copy(test_table_location, tmp_location)
    monkeypatch.setattr(cleaning, "setup_repo_path", lambda: tmp_path)
    csvs = pre_merge_check()
    assert len(csvs) > 0

# ==========================================================
# ylivertainen_janitor
# ==========================================================
# End-to-end parity checks for both janitor execution modes.

def test_ylivertainen_janitor_smoke_apply_all_true(make_project, synthetic_df):
    project_test = make_project(synthetic_df.copy(deep=True))
    project_test = project_test.ylivertainen_janitor(
        apply_all=True,
        id_cols="patient_card_no",
        include_first=False,
        drop_dupes=False)
    project_stepped = make_project(synthetic_df.copy(deep=True))
    id_cols, include_first, drop_dupes = "patient_card_no", False, False
    project_stepped = (project_stepped
    .apply_schema()
    .apply_derived()
    .apply_nan_features()
    .resolve_dupes(id_cols, include_first, drop_dupes)
    .cleaning_overview_and_commit())
    assert isinstance(project_test, pd.DataFrame)
    assert isinstance(project_stepped, pd.DataFrame)
    pd.testing.assert_frame_equal(project_test, project_stepped)

def test_ylivertainen_janitor_smoke_apply_all_false(make_project, synthetic_df):
    project_test = make_project(synthetic_df.copy(deep=True))
    project_test = project_test.ylivertainen_janitor(
        apply_all=False,
        id_cols="patient_card_no",
        include_first=False,
        drop_dupes=False)
    project_stepped = make_project(synthetic_df.copy(deep=True))
    id_cols, include_first, drop_dupes = "patient_card_no", False, False
    project_stepped = (project_stepped
    .resolve_dupes(id_cols, include_first, drop_dupes)
    .cleaning_overview_and_commit())
    assert isinstance(project_test, pd.DataFrame)
    assert isinstance(project_stepped, pd.DataFrame)
    pd.testing.assert_frame_equal(project_test, project_stepped)

# ==========================================================
# explore_values
# ==========================================================
# Smoke test with mixed dtypes.

def test_explore_values_w_mixed_dtypes(make_project, synthetic_df):
    project = make_project(synthetic_df.copy(deep=True))
    project.explore_values()

# ==========================================================
# cleaning_overview_and_commit
# ==========================================================
# Verifies return type and summary output markers.

def test_cleaning_overview_and_commit_smoke(make_project, synthetic_df, capsys):
    project = make_project(synthetic_df.copy(deep=True))
    returns_df = project.cleaning_overview_and_commit()
    out = capsys.readouterr().out
    out = re.sub(r"\x1b\[[0-9;]*m", "", out)
    assert isinstance(returns_df, pd.DataFrame)
    assert """🧠 DATA CLEANING COMPLETE — FRAME STABLE""" in out
    assert """✔ CLEANED DF ONLINE""" in out
    assert """→ Next: pass""" in out

