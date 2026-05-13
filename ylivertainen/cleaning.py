#==================================
#             IMPORTS
#==================================
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from ylivertainen._pathing import setup_repo_path
from IPython.display import display, Markdown

from ylivertainen.schema import SCHEMA, DERIVED
#==================================================
#            Pre merge column name check
#==================================================
def pre_merge_check(
    colname_length: int = 15,
    show_dfs: bool = False,
    ) -> None:
    
    def _csv_column_names_pretty(text) -> None:
        text_align = "text-align:" + "center"
        color = "color:" + "white"
        font_weight = "font-weight:" + "700"
        font_size = "font-size:" + "25px"

        st.markdown(
            f"<div style='{text_align}; {color}; {font_weight}; {font_size};'>{text}</div>",
            unsafe_allow_html=True
            )

    root = setup_repo_path()
    raw_dir = root / "ylivertainen" / "data" / "raw"
    csvs = sorted(raw_dir.glob("*.csv"))    # <== takes all CSVs from ylivertainen/data/raw
    
    base_cols = None

    for i, csv in enumerate(csvs):
        if "\\" in str(csv):
            csv_name = str(csv).split(sep='\\')[-1]
        else:
            csv_name = str(csv).split(sep='/')[-1]
        df = pd.read_csv(csv, nrows=2)
        cols = list(df.columns.str.strip())
        df.columns = df.columns.astype(str).str.strip().str[:colname_length]

        if i == 0:
            _csv_column_names_pretty(f"{'=' * 25} {str(csv_name)} {'=' * 25}")
            # Save the first file's columns as the reference
            st.write(f"{'\n'.join([f"`{col}`" for col in cols])}")
            base_cols = cols
            st.success(f" ✅ Using this file as column name reference ✅ ")
        else:
            _csv_column_names_pretty(f"{'=' * 25} {str(csv_name)} {'=' * 25}")
            base_set = set(base_cols)
            curr_set = set(cols)

            missing = base_set - curr_set      # in first, not in current
            extra   = curr_set - base_set      # in current, not in first

            if missing or extra:
                st.warning(" ❌ Schema difference vs FIRST file:")
                if missing:
                    st.warning(f"=== Missing (in first, not here) ===\n\n'{f"'\n\n'".join(missing)}'")
                if extra:
                    st.warning(f"=== Extra (here, not in first) ===\n\n'{f"'\n\n'".join(extra)}'")
            else:
                st.success(" ✅ Same set of column names as FIRST file")

        if show_dfs:
            st.dataframe(df.head(2))

    return csvs

#====================================================================
#====================================================================
#                           Class definition
#====================================================================
class YlivertainenDataCleaningSurg:
    #================================================
    #                   MERGE CSVs 
    #================================================
    @staticmethod
    def merge_dfs(csvs, COLUMN_RENAME_MAP) -> pd.DataFrame:
           
        confident_df = pd.DataFrame()

        for csv in csvs:
            naive_df = pd.read_csv(csv).copy(deep=True)
            naive_df.columns = naive_df.columns.str.strip()
            
            for canonical, old_name in COLUMN_RENAME_MAP.items():
                
                if isinstance(old_name, str):
                    naive_df = naive_df.rename(columns={old_name: canonical}) 

                elif isinstance(old_name, list):
                    for name in old_name:
                        if name in naive_df.columns:
                            naive_df = naive_df.rename(columns={name: canonical})
                        else:
                            pass
                
            if len(confident_df) == 0:
                confident_df = naive_df
            
            else:
                confident_df = pd.concat([confident_df, naive_df], ignore_index=True)
        
        # ====== delete all the non-canonical columns ======
        confident_df = confident_df.reindex(columns=list(COLUMN_RENAME_MAP.keys()))

        super_df = confident_df

        #======================================================================================================
        #TODO: make a function to coalesce the columns with same canonical names (no behaviour like this now)
            # now it gives out the error
        #======================================================================================================

        return super_df
    
    #============ Ship's Log: Registering the Loot ============
    def __init__(self, csvs: list, COLUMN_RENAME_MAP: dict) -> None:
        self.csvs = csvs
        self.df = self.merge_dfs(csvs, COLUMN_RENAME_MAP)
    
    def __str__(self) -> str:
        csv_count = len(self.csvs)
        csv_names = "\n- ".join(Path(str(a)).name for a in self.csvs)
        return (
            f'In YlivertainenDataCleaningSurg got passed:\n'
            f'{csv_count} CSVs:\n'
            f'- {csv_names}'
            f'\n{self.df.shape[0]} rows, {self.df.shape[1]} columns'
        )
    #==============================================================
    #                   SCHEMA FUNCTION DEFINITION 
    #==============================================================
    def apply_schema(self, COLUMN_RENAME_MAP) -> "YlivertainenDataCleaningSurg":
        
        self.df.columns = [col.strip() for col in self.df.columns]

        #=================================================
        #              INITIAL COLUMN CHECK
        #=================================================
        st.write("═" * 70)
        st.write(f"🧱 SCHEMA / COLUMN CHECK")
        st.write("─" * 70)

        CRM_columns = [col for col in COLUMN_RENAME_MAP]
        mismatched_cols = []
        for ColSpec in SCHEMA:
            if ColSpec.name not in CRM_columns:
                mismatched_cols.append(ColSpec.name)
        if len(mismatched_cols) > 0:
            st.error(f'❌ SCHEMA columns differ from COLUMN_RENAME_MAP: {mismatched_cols}')
            st.stop()

        for ColSpec in SCHEMA:  
            st.write(f'{ColSpec.name} successfully found in columns')  
        
        # ===== feedback =====
        st.write(f"✔ SCHEMA OK: COLUMN_RENAME_MAP matches SCHEMA\n")

        #=================================================
        #           REPLACE w/ NaNs in SCHEMA
        #=================================================
        st.write("═" * 70)
        st.write(f"💧 MISSING VALUES / NaNs")
        st.write("─" * 70)
        
        NaN_replaced_cols = []
        for ColSpec in SCHEMA:
            if ColSpec.nulls is not None:
                NaN_replaced_cols.append(ColSpec.name)
                for null in ColSpec.nulls:
                    before_NaNs = self.df[ColSpec.name].isna().sum()
                    self.df[ColSpec.name] = self.df[ColSpec.name].replace(null, np.nan)
                    after_NaNs = self.df[ColSpec.name].isna().sum()
                    st.write(f'{null} ⇒ NaN @ {ColSpec.name} | Values that became NaN: {after_NaNs - before_NaNs}')
            elif ColSpec.ordered:
                vals_to_allow = [val for val in ColSpec.ordered]
                null_vals = [val for val in self.df[ColSpec.name].unique() if val not in vals_to_allow]
                if len(null_vals) != 0:
                    NaN_replaced_cols.append(ColSpec.name)
                for null in null_vals:
                    if not pd.isna(null):
                        before_NaNs = self.df[ColSpec.name].isna().sum()
                        self.df[ColSpec.name] = self.df[ColSpec.name].replace(null, np.nan)
                        after_NaNs = self.df[ColSpec.name].isna().sum()
                        st.write(f'{null} ⇒ NaN @ {ColSpec.name} | Values that became NaN: {after_NaNs - before_NaNs}')
        # ===== give positive feedback for all columns found =====
        if len(NaN_replaced_cols) > 0:
            st.write(f"✔ NaN pass:\n"
            f"replaced values in {NaN_replaced_cols} columns\n")
        else:
            st.write(f"✔ NaN pass: nothing to replace\n")

        #=================================================
        #            REPLACE values in SCHEMA
        #=================================================
        st.write("═" * 70)
        st.write(f"🐾 REPLACE VALUES ")
        st.write("─" * 70)

        for ColSpec in SCHEMA:
            if isinstance(ColSpec.replace, dict):
                for old, new in ColSpec.replace.items():
                    self.df[ColSpec.name] = self.df[ColSpec.name].replace(old, new)
                    st.write(f'"{old}" ⇒ "{new}" @ {ColSpec.name}')
        st.write(f'✔ Replaced: All target values replaced \n')

        #=================================================
        #            Change DTYPES in SCHEMA
        #=================================================
        st.write("═" * 70)
        st.write(f"🧬 DTYPE CONVERSION")
        st.write("─" * 70)
        
        for ColSpec in SCHEMA:
            col_name = f'{ColSpec.name}'
            if ColSpec.kind == 'numeric':
                col_before = self.df[ColSpec.name].copy()
                before_NaNs = col_before.isna().sum()
                col_after = pd.to_numeric(col_before, errors='coerce')
                after_NaNs = col_after.isna().sum()
                gained_nan_mask = col_before.notna() & col_after.isna()
                replaced_values = col_before[gained_nan_mask].unique().tolist()
                st.write(f'{col_name} converted to "numeric" | Values that became NaN: {after_NaNs - before_NaNs} | List: {replaced_values}')
                self.df[ColSpec.name] = col_after

            elif ColSpec.kind == 'timedelta':
                self.df[ColSpec.name] = self.df[ColSpec.name].astype('timedelta64[ns]')
                st.write(f'{col_name} converted to "timedelta"')

            elif ColSpec.kind == 'datetime':
                col_before = self.df[ColSpec.name].copy()
                before_NaNs = col_before.isna().sum()
                col_after = pd.to_datetime(self.df[ColSpec.name], format='mixed', dayfirst=True, errors='coerce')
                after_NaNs = col_after.isna().sum()
                gained_nan_mask = col_before.notna() & col_after.isna()
                replaced_values = col_before[gained_nan_mask].unique().tolist()
                st.write(f'{col_name} converted to "datetime" | Values that became NaN: {after_NaNs - before_NaNs} | List: {replaced_values}')
                self.df[ColSpec.name] = col_after

            elif ColSpec.kind == 'categorical':
                if isinstance(ColSpec.ordered, tuple):
                    self.df[ColSpec.name] = self.df[ColSpec.name].astype("Int64")
                    col_after = pd.Categorical(self.df[ColSpec.name], list(ColSpec.ordered), ordered=True)
                    st.write(f'{col_name} converted to "Categorical (ordered)"')
                    self.df[ColSpec.name] = col_after

                else:
                    self.df[ColSpec.name] = pd.Categorical(self.df[ColSpec.name], ordered=False)
                    st.write(f'{col_name} converted to "Categorical (non-ordered)"')

            elif ColSpec.kind == 'text':
                self.df[ColSpec.name] = self.df[ColSpec.name].astype('string[python]').str.strip()
                st.write(f'{col_name} converted to "text"')

            else:
                st.error(f'❌ No such dtype: {ColSpec.kind}')
                st.stop()
        # ===== feedback =====
        st.write(f"✔ Done: all columns converted to target dtypes\n")
        return self

    #==============================================================
    #                  DERIVED FUNCTION DEFINITION 
    #==============================================================
    def apply_derived(self) -> "YlivertainenDataCleaningSurg":
        st.write("═" * 70)
        st.write(f"🍼 CREATION OF DERIVED ")
        st.write("─" * 70)

        df = self.df.copy()

        for ColSpec in DERIVED:
            #============================
            #           MATCH
            #============================
            if ColSpec.kind == 'match':
                if not ColSpec.derive_from:
                    raise KeyError("❌ Forgot derive_from for one of the match ColSpecs in DERIVED")
                if not ColSpec.match_by:
                    raise KeyError("❌ Forgot match_by for one of the match ColSpecs in DERIVED")
                if not ColSpec.name:
                    raise KeyError("❌ Forgot name for one of the match ColSpecs in DERIVED")

                left_col, right_col = ColSpec.derive_from
                left = df[left_col]
                right = df[right_col]
                col_name = ColSpec.name

                normalized_criteria = []
                for criteria in ColSpec.match_by:
                    text = str(criteria)
                    text = text.strip().upper()
                    normalized_criteria.append(text)
                criteria_tuple = tuple(normalized_criteria)

                left_present = left.notna()
                right_present = right.notna()
                left_in_group = left.str.startswith(criteria_tuple)
                right_in_group = right.str.startswith(criteria_tuple)

                agreement = pd.Series(pd.NA, index=df.index, dtype='boolean')
                evaluable = left_present & right_present & left_in_group
                agreement.loc[evaluable] = right_in_group.loc[evaluable]
                df[col_name] = agreement

                n_evaluable = int(evaluable.sum())
                match_rate = 0.0 if n_evaluable == 0 else round(float(agreement.loc[evaluable].mean()) * 100, 2)

                st.write(
                    f"🎯 {col_name}: {match_rate}%\n"
                    f"of cases labeled as {normalized_criteria} in {left_col}\n"
                    f"were also classified in the same family in {right_col}\n"
                    f"n = {n_evaluable} evaluable cases\n"
                )
                # return updated df
                self.df = df

            #============================
            #          DATETIME
            #============================
            elif ColSpec.kind == 'datetime':
                if not ColSpec.derive_from:
                    raise KeyError(f'❌ Forgot derive_from for one of the ColSpec.datetime in DERIVED')
                if not ColSpec.datetime_units:
                    raise KeyError(f'❌ Forgot datetime_units for one of the ColSpec.datetime in DERIVED')
                if not ColSpec.name:
                    raise KeyError(f'❌ Forgot name for one of the ColSpec.datetime in DERIVED')
            
                source_col = ColSpec.derive_from[0]
                target_col = ColSpec.name
                unit = ColSpec.datetime_units  # e.g. "hour", "dow", "month_name", "year"

                datetime_extractors = {
                    "hour": lambda s: pd.Categorical(
                        s.dt.hour,
                        list(range(0, 24, 1)),
                        ordered=True,
                    ),
                    "dow": lambda s: pd.Categorical(
                        s.dt.day_name(),
                        ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
                        ordered=True,
                    ),
                    "workday_bool": lambda s: s.dt.day_name().isin(
                        ['Monday','Tuesday','Wednesday','Thursday','Friday']
                    ),
                    "month_name": lambda s: pd.Categorical(
                        s.dt.month_name(),
                        ["January","February","March","April","May","June","July",
                            "August","September","October","November","December"],
                        ordered=True,
                    ),
                    "year": lambda s: s.dt.year,
                }

                if unit not in datetime_extractors:
                    raise ValueError(f"❌ Unknown datetime unit '{unit}' for column '{target_col}'")

                extractor = datetime_extractors[unit]
                df[target_col] = extractor(df[source_col])

                st.write(f'📅 Created {ColSpec.name} derived from  "{source_col}"')
                if ColSpec.name in df.select_dtypes(include="number"):
                    bad_mask = df[ColSpec.name] < 0
                    df.loc[bad_mask, ColSpec.name] = pd.NA
                    bad_count = bad_mask.sum()
                
                if ColSpec.name in df.select_dtypes(include="number"):
                    max_val = df[ColSpec.name].max()
                    min_val = df[ColSpec.name].min()
                    mean_val = round(df[ColSpec.name].mean(), 3)
                    std_val = round(df[ColSpec.name].std(), 3)
                    st.write(f'MAX: {max_val}')
                    if bad_count > 0:
                        st.write(f'MIN: {min_val} ==> also trashed NEGATIVE datetimes: {bad_count}')
                    else:
                        st.write(f'MIN: {min_val} ==> no negative timedeltas found')

                    st.write(f'Mean: {mean_val}')
                    st.write(f'STD: {std_val}\n')
                else:
                    unique_count = df[ColSpec.name].nunique()
                    commonest = df[ColSpec.name].value_counts().head(5).to_dict()
                    rarest = df[ColSpec.name].value_counts().tail(5).to_dict()
                    nan_count = df[ColSpec.name].isna().sum()
                    st.write(f'Unique count: {unique_count}')
                    if unique_count > 5:
                        st.write(f'Unique first: {commonest}')
                        st.write(f'Unique last: {rarest}\n')
                    else:
                        st.write(f'Uniques: {commonest}\n')
        
            #============================
            #          TIMEDELTA
            #============================
            elif ColSpec.kind == 'timedelta':
                if not ColSpec.derive_from:
                    raise KeyError(f'❌ Forgot derive_from for one of the ColSpec.timedelta in DERIVED')
                if not ColSpec.timedelta_units:
                    raise KeyError(f'❌ Forgot timedelta_units for one of the ColSpec.timedelta in DERIVED')
                if not ColSpec.name:
                    raise KeyError(f'❌ Forgot name for one of the ColSpec.timedelta in DERIVED')
                
                start_col, end_col = ColSpec.derive_from

                missing_cols = [col for col in (start_col, end_col) if col not in df.columns]
                if missing_cols:
                    raise KeyError(f'❌ Missing derivable columns: {missing_cols}')

                if not pd.api.types.is_datetime64_any_dtype(df[start_col]) or not pd.api.types.is_datetime64_any_dtype(df[end_col]):
                    raise ValueError(
                        f'❌ Derivable columns must be datetime. Got {start_col}={df[start_col].dtype}, {end_col}={df[end_col].dtype}'
                    )

                df[ColSpec.name] = df[end_col] - df[start_col]
                df[ColSpec.name] = df[ColSpec.name].astype('timedelta64[ns]')

                timedelta_units = {}
            
                unit_map = {
                    "seconds": pd.Timedelta(seconds=1),
                    "minutes": pd.Timedelta(minutes=1),
                    "hours":   pd.Timedelta(hours=1),
                    "days":    pd.Timedelta(days=1),
                }
            
                unit = timedelta_units.get(ColSpec.name, ColSpec.timedelta_units)
                base = unit_map[unit]          # this is a real Timedelta

                df[ColSpec.name] = np.floor(df[ColSpec.name] / base)

                #============================
                #    A VERY ARGUABLE LINE
                #============================
                df[ColSpec.name] = df[ColSpec.name].astype("Int64")
                #============================

                st.write(f'⌛ Created {ColSpec.name} derived from the difference between "{end_col}" and "{start_col}"')
                bad_mask = df[ColSpec.name] < 0
                df.loc[bad_mask, ColSpec.name] = pd.NA
                bad_count = bad_mask.sum()
                
                max_val = df[ColSpec.name].max()
                min_val = df[ColSpec.name].min()
                mean_val = round(df[ColSpec.name].mean(), 3)
                std_val = round(df[ColSpec.name].std(), 3)
                st.write(f'MAX: {max_val}')
                if bad_count > 0:
                    st.write(f'MIN: {min_val} ==> also trashed NEGATIVE timedeltas: {bad_count}')
                else:
                    st.write(f'MIN: {min_val} ==> no negative timedeltas found')
                st.write(f'Mean: {mean_val}')
                st.write(f'STD: {std_val}\n')


        # ===== feedback =====
        st.write(f"✔ Derived: creation process completed")
        return self
    
    #==============================================================
    #               CLEAN UP AFTER SCHEMA & DERIVED 
    #==============================================================
    def cleanup(self) -> "YlivertainenDataCleaningSurg":
        st.write("═" * 70)
        st.write(f"🧹 CLEAN UP ")
        st.write("─" * 70)
        
        dropped_columns = []
        for ColSpec in SCHEMA:
            col_name = f'{ColSpec.name}'
            if not ColSpec.keep:
                dropped_columns.append(ColSpec.name)
                st.write(f'🪓 Successfully amputated {col_name}')
        self.df = self.df.drop(columns=dropped_columns)
        # ===== feedback  =====
        st.write(f"✔ Cleaned up: {len(dropped_columns)} unnecessary columns dropped")
        return self
    
    #==============================================================
    #                  Creating NaN feature flags 
    #==============================================================
    def apply_nan_features(self) -> "YlivertainenDataCleaningSurg":
        st.write("═" * 70)
        st.write(f" ⭕ NaN features Flags ")
        st.write("─" * 70)

        df = self.df.copy()
        
        exclude_derived = [ColSpec.name for ColSpec in DERIVED]
        cols = [col for col in df if col not in exclude_derived]
        
        for col in cols:
            null_sum = df[col].isna().sum()
            
            if null_sum > 0:
                df[f'{col}_missing'] = df[col].isna()
                st.write(f'{col} has {null_sum} NaNs. Created feature flag column')

            else:
                st.write(f'✅ No NaNs found in {col}')

        # ===== feedback =====
        total_nans = df.isna().sum().sum()
        if total_nans > 0:
            st.write(f"✔ NaN feature cols: created NaN feature columns")
        else:
            st.write(f"✔ NaN feature cols: {total_nans} NaNs found in the whole dataset")
        self.df = df
        return self
    
    #==============================================================
    #       Finding duped rows based on specified ID columns 
    #==============================================================
    #========== HELPERs ==========
    def _resolve_duplicate_masks(self, id_cols) -> tuple[list[str], pd.Series, pd.Series]:
        if id_cols is None:
            id_cols = []
        elif isinstance(id_cols, str):
            id_cols = [id_cols]
        else:
            id_cols = list(id_cols)

        if len(id_cols) == 0:
            empty_mask = pd.Series(False, index=self.df.index, dtype='boolean')
            return id_cols, empty_mask, empty_mask

        missing_ids = [col for col in id_cols if col not in self.df.columns]
        if missing_ids:
            raise ValueError(f'❌ ID columns not found: {missing_ids}')

        key_df = self.df[id_cols].copy()

        for col in id_cols:
            col_data = key_df[col]
            if (
                pd.api.types.is_object_dtype(col_data)
                or pd.api.types.is_string_dtype(col_data)
                or isinstance(col_data.dtype, pd.CategoricalDtype)
            ):
                normalized = col_data.astype('string[python]').str.strip().str.lower()
                key_df[col] = normalized.replace('', pd.NA)

        complete_id_mask = key_df.notna().all(axis=1)
        skipsfirst_dupe_mask = complete_id_mask & key_df.duplicated(subset=id_cols, keep='first')
        includesfirst_dupe_mask = complete_id_mask & key_df.duplicated(subset=id_cols, keep=False)

        return id_cols, skipsfirst_dupe_mask, includesfirst_dupe_mask
    #=============================

    def resolve_dupes(self, id_cols, include_first: bool = True, drop: bool = False) -> "YlivertainenDataCleaningSurg":
        st.write("═" * 70)
        st.write(f"🔍 DUPE SEARCH ")
        st.write("─" * 70)

        id_cols, skipsfirst_dupe_mask, includesfirst_dupe_mask = self._resolve_duplicate_masks(id_cols)

        if len(id_cols) == 0:
            st.write(f'There are no ID columns')
            display(self.df.iloc[0:0])
            return self

        dup_mask = includesfirst_dupe_mask if include_first else skipsfirst_dupe_mask
        dupe_count = int(dup_mask.sum())

        if dupe_count > 0:
            st.write(f"✔ Dupe search:")
            if include_first:
                st.write(f'There are {dupe_count} rows in duplicate groups based on complete normalized ID: {id_cols}')
                st.write('Note: includes the first row in each duplicate group. Rows with incomplete IDs were ignored')
            else:
                st.write(f'There are {dupe_count} later duplicate rows based on complete normalized ID: {id_cols}')
                st.write('Note: not including the first. Rows with incomplete IDs were ignored')
        else:
            st.write(f'✅ No duplicates found based on complete normalized ID: {id_cols}')
            st.write(f"✔ Dupe search: no duplicates found based on complete normalized ID: {id_cols}")

        dupe_df = self.df.loc[dup_mask]
        display(dupe_df)

        if drop:
            st.write("═" * 70)
            st.write(f"🪚 DUPE REMOVAL ")
            st.write("─" * 70)

            id_cols, skipsfirst_dupe_mask, _ = self._resolve_duplicate_masks(id_cols)

            if len(id_cols) == 0:
                st.write(f"✔ Dupe removal: no ID columns provided — skipping duplicate removal")
                return self

            dupe_count = int(skipsfirst_dupe_mask.sum())

            st.write(f"✔ Dupe removal: removed {dupe_count} later duplicates based on complete normalized ID: {id_cols}")

            self.df = self.df[~skipsfirst_dupe_mask].reset_index(drop=True)
            return self
        else:
            return self

    #==============================================================
    #                   Save cleaned df to csv
    #==============================================================
    def ylivertainen_janitor(
        self,
        apply_all: bool = False,
        id_cols: list[str] = None,
        include_first: bool = True,
        drop_dupes: bool = False,
        ):
        if apply_all:
            self.apply_schema()
            self.apply_derived()
            self.apply_nan_features()
            self.resolve_dupes(id_cols, include_first, drop_dupes)
            self.cleaning_overview_and_commit()
        else:
            self.resolve_dupes(id_cols, include_first, drop_dupes)
            self.cleaning_overview_and_commit()
        
        return self.df
    
    #==============================================================
    #                   Overview of all columns 
    #==============================================================
    def explore_values(self):
        
        numerical_cols = [col for col in self.df.select_dtypes(include='number', exclude=["bool", "boolean"]) if not col.endswith('_missing')]
        categorical_cols = [col for col in self.df.select_dtypes(include='category') if not col.endswith('_missing')]
        bool_cols = [col for col in self.df.select_dtypes(include=["bool", "boolean"]) if not col.endswith('_missing')]

        analyse_those = numerical_cols + categorical_cols + bool_cols

        for col in analyse_those:

            st.write(f'===== {col} =====')
            
            if col in categorical_cols or col in bool_cols:

                dt = self.df[col].dtype
                unique_count = self.df[col].nunique()
                commonest = self.df[col].value_counts().head(5).to_dict()
                rarest = self.df[col].value_counts().tail(5).to_dict()
                nan_count = self.df[col].isna().sum()

                st.write(f'Dtype: {dt}')
                st.write(f'Unique count: {unique_count}')
                if unique_count > 5:
                    st.write(f'Unique first: {commonest}')
                    st.write(f'Unique last: {rarest}')
                else:
                    st.write(f'Uniques: {commonest}')
                if nan_count > 0:
                    st.write(f'NaN count: {nan_count}')
                else:
                    st.write(f'NaN count: {nan_count}')

            if col in numerical_cols:
                
                dt = self.df[col].dtype
                max_val = self.df[col].max()
                min_val = self.df[col].min()
                mean_val = round(self.df[col].mean(), 3)
                std_val = round(self.df[col].std(), 3)
                nan_count = self.df[col].isna().sum()

                st.write(f'Dtype: {dt}')
                st.write(f'MAX: {max_val}')
                if 'timedelta' in col:
                    if min_val < 0:
                        st.write(f'MIN: {min_val}')
                else:
                    st.write(f'MIN: {min_val}')
                st.write(f'Mean: {mean_val}')
                st.write(f'STD: {std_val}')
                if nan_count > 0:
                    st.write(f'NaN count: {nan_count}')
                else:
                    st.write(f'NaN count: {nan_count}')

        for col in self.df.columns:
            if col not in analyse_those:
                st.write(f'===== {col} =====')
                st.write(f'Dtype: {self.df[col].dtype}')
                st.write(f'NaN count: {self.df[col].isna().sum()}')
                st.write(f'Unique count: {self.df[col].nunique()}')
                st.write(f'Commonest: {self.df[col].value_counts().head(5).to_dict()}')
                st.write(f'Rarest: {self.df[col].value_counts().tail(5).to_dict()}')

    #==============================================================
    #                      COMMIT CLEANED DF 
    #==============================================================
    def cleaning_overview_and_commit(self):
        st.write("═" * 70)
        st.write(f"🧠 DATA CLEANING COMPLETE — FRAME STABLE")
        st.write("─" * 70)

        cleaned_df = self.df

        n_rows, n_cols = cleaned_df.shape
        total_nans = cleaned_df.isna().sum().sum()
        nan_pct = (total_nans / (n_rows * n_cols)) * 100 if n_rows and n_cols else 0

        st.write(f"✔ CLEANED DF ONLINE")
        st.write(f"   Shape         : {n_rows} rows x {n_cols} cols")
        st.write(f"   Remaining NaNs: {total_nans} cells ({nan_pct:0.2f}%)")
        st.write("   Status        : noise resected, dtypes aligned, ready for cohort craniotomy 🧠🪓")
        st.write(f"\n→ Next: pass this straight into cohort — no extra rituals needed.")

        display(cleaned_df.head())
        return cleaned_df