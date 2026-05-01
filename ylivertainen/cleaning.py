#==================================
#             IMPORTS
#==================================
import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import display, Markdown

from ylivertainen.schema import COLUMN_RENAME_MAP, SCHEMA, DERIVED

#============ ANSI Escape Codes for formating ============
from ylivertainen.aesthetics_helpers import GREEN, YELLOW, ORANGE, RED, BOLD, BLUE, GRAY, RESET

#==================================================
#            Pre merge column name check
#==================================================
def pre_merge_check(
    root: Path,
    colname_length: int = 15,
    show_dfs: bool = False,
    ) -> None:
    
    raw_dir = root / "data" / "raw"
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
            print(f"{BOLD}{BLUE}=" * 70)
            print(csv_name)
            print("=" * 70, RESET)
            # Save the first file's columns as the reference
            print(f"{'\n'.join([f"`{col}`" for col in cols])}")
            base_cols = cols
            print(f"{BOLD}===== ✅ Using this file as column name reference ✅ ====={RESET}")
        else:
            print('\n' + f"{BOLD}{BLUE}=" * 70)
            print(csv_name)
            print("=" * 70, RESET)
            base_set = set(base_cols)
            curr_set = set(cols)

            missing = base_set - curr_set      # in first, not in current
            extra   = curr_set - base_set      # in current, not in first

            if missing or extra:
                print(" ❌ Schema difference vs FIRST file:")
                if missing:
                    print(f'{ORANGE}{BOLD}=== Missing (in first, not here) ==={RESET}')
                    print(f"'{f"'\n'".join(missing)}'")
                if extra:
                    print(f'{ORANGE}{BOLD}=== Extra (here, not in first) ==={RESET}')
                    print(f"'{f"'\n'".join(extra)}'")
            else:
                print(" ✅ Same set of column names as FIRST file")

        if show_dfs:
            display(df.head(2))

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
    def merge_dfs(csvs) -> pd.DataFrame:
        
        if not csvs:
            raise ValueError("❌ No CSV file/-s provided to merge_dfs")
        
        df_list = []
        dropped_cols = {}
        renamed_cols = {}

        for csv in csvs:
            
            naive_df = pd.read_csv(csv)
            naive_df.columns = naive_df.columns.str.strip()
            confident_cols = []
            
            for col in naive_df.columns:
                # ===== find if it's in canonical =====
                if col in COLUMN_RENAME_MAP:
                    if col not in confident_cols:
                        confident_cols.append(col)
                    continue

                # ===== if not then look in alias =====
                found = False
                for canonical, aliases in COLUMN_RENAME_MAP.items():
                    if col in aliases:
                        naive_df = naive_df.rename(columns={col: canonical})
                        renamed_cols.setdefault(csv, {}).setdefault(canonical, []).append(col)
                        if canonical not in confident_cols:
                            confident_cols.append(canonical)
                        found = True
                        break

                # ===== drop if in neither =====
                if not found:
                    dropped_cols.setdefault(csv, []).append(col)
            
            confident_df = naive_df[confident_cols]
            df_list.append(confident_df)
        
        super_df = pd.concat(df_list, ignore_index=True)        # <---- works for 1 or many CSVs

        sequenced_list = list(COLUMN_RENAME_MAP)
        # If a canonical column never appears in any input CSV, keep it as NaN
        # instead of crashing with a KeyError.
        super_df = super_df.reindex(columns=sequenced_list)

        def compact_columns_text(cols: list[str], per_line: int = 6) -> str:
            chunks = [cols[i:i + per_line] for i in range(0, len(cols), per_line)]
            return "\n".join(", ".join(f"`{c}`" for c in chunk) for chunk in chunks)

        total_rows, total_cols = super_df.shape
        display(
            Markdown(
                "\n".join(
                    [
                        "## 🧩 MERGE SUMMARY",
                        "---",
                        f"**Rows:** `{total_rows}`  |  **Columns:** `{total_cols}`",
                        "",
                        "## 🧱 CANONICAL COLUMNS",
                        compact_columns_text(list(super_df.columns), per_line=6),
                    ]
                )
            )
        )

        removed_section_lines: list[str] = ["---", "## 🗑️ REMOVED COLUMNS", ""]
        if dropped_cols:
            for csv_key in sorted(dropped_cols.keys(), key=lambda p: Path(str(p)).name):
                csv_name = Path(str(csv_key)).name
                dropped = sorted(dropped_cols[csv_key])
                removed_section_lines.append(f"**{csv_name}** ({len(dropped)})\n")
                removed_section_lines.append(", ".join(f"`{c}`" for c in dropped))
                removed_section_lines.append("")
        else:
            removed_section_lines.append("_none_")
        display(Markdown("\n".join(removed_section_lines)))

        canonical_view: dict[str, list[tuple[object, str]]] = {}
        for csv, mapping in renamed_cols.items():
            for canonical, old_names in mapping.items():
                for old_name in old_names:
                    canonical_view.setdefault(canonical, []).append((csv, old_name))

        renamed_section_lines: list[str] = ["---", "## 🔁 RENAMED COLUMNS", ""]
        if canonical_view:
            for canonical in sorted(canonical_view.keys()):
                renamed_section_lines.append(
                    f"<span style='color:#4da3ff; font-weight:700;'>===== {canonical} =====</span>"
                    )
                entries = sorted(
                    canonical_view[canonical],
                    key=lambda t: (Path(str(t[0])).name, str(t[1])),
                )
                for csv_key, old_name in entries:
                    table_name = Path(str(csv_key)).name
                    renamed_section_lines.append(f"\n{old_name} 🪄 {table_name}")
                renamed_section_lines.append("")
        else:
            renamed_section_lines.append("_none_")
        display(Markdown("\n".join(renamed_section_lines)))
        display(Markdown("---" * 70))

        return super_df
    
    #============ Ship's Log: Registering the Loot ============
    def __init__(self, csvs: list) -> None:
        self.csvs = csvs
        self.df = self.merge_dfs(csvs)
    
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
    def apply_schema(self) -> "YlivertainenDataCleaningSurg":
        
        self.df.columns = [col.strip() for col in self.df.columns]

        #=================================================
        #              INITIAL COLUMN CHECK
        #=================================================
        print("═" * 70)
        print(f"{BOLD}🧱 SCHEMA / COLUMN CHECK{RESET}")
        print("─" * 70)

        CRM_columns = [col for col in COLUMN_RENAME_MAP]
        mismatched_cols = []
        for ColSpec in SCHEMA:
            if ColSpec.name not in CRM_columns:
                mismatched_cols.append(ColSpec.name)
        if len(mismatched_cols) > 0:
            raise ValueError(f'❌ SCHEMA columns differ from COLUMN_RENAME_MAP: {mismatched_cols}')

        for ColSpec in SCHEMA:  
            print(f'{BLUE}{BOLD}{ColSpec.name}{RESET} successfully found in columns')  
        
        # ===== feedback =====
        print(f"{GREEN}{BOLD}✔ SCHEMA OK:{RESET} COLUMN_RENAME_MAP matches SCHEMA\n")

        #=================================================
        #           REPLACE w/ NaNs in SCHEMA
        #=================================================
        print("═" * 70)
        print(f"{BOLD}💧 MISSING VALUES / NaNs{RESET}")
        print("─" * 70)
        
        NaN_replaced_cols = []
        for ColSpec in SCHEMA:
            if ColSpec.nulls is not None:
                NaN_replaced_cols.append(ColSpec.name)
                for null in ColSpec.nulls:
                    before_NaNs = self.df[ColSpec.name].isna().sum()
                    self.df[ColSpec.name] = self.df[ColSpec.name].replace(null, np.nan)
                    after_NaNs = self.df[ColSpec.name].isna().sum()
                    print(f'"{null}" ⇒ "NaN" @ {BLUE}{BOLD}{ColSpec.name}{RESET} | Values that became NaN: {after_NaNs - before_NaNs}')
        # ===== give positive feedback for all columns found =====
        if len(NaN_replaced_cols) > 0:
            print(f"{GREEN}{BOLD}✔ NaN pass:{RESET}\n"
            f"replaced values in {NaN_replaced_cols} columns\n")
        else:
            print(f"{GREEN}{BOLD}✔ NaN pass:{RESET} nothing to replace\n")

        #=================================================
        #            REPLACE values in SCHEMA
        #=================================================
        print("═" * 70)
        print(f"{BOLD}🐾 REPLACE VALUES {RESET}")
        print("─" * 70)

        for ColSpec in SCHEMA:
            if isinstance(ColSpec.replace, dict):
                for old, new in ColSpec.replace.items():
                    self.df[ColSpec.name] = self.df[ColSpec.name].replace(old, new)
                    print(f'"{old}" ⇒ "{new}" @ {BLUE}{BOLD}{ColSpec.name}{RESET}')
        print(f'{GREEN}{BOLD}✔ Replaced:{RESET} All target values replaced {RESET}\n')

        #=================================================
        #            Change DTYPES in SCHEMA
        #=================================================
        print("═" * 70)
        print(f"{BOLD}🧬 DTYPE CONVERSION{RESET}")
        print("─" * 70)
        
        for ColSpec in SCHEMA:
            col_name = f'{BLUE}{BOLD}{ColSpec.name}{RESET}'
            if ColSpec.kind == 'numeric':
                col_before = self.df[ColSpec.name].copy()
                before_NaNs = col_before.isna().sum()
                col_after = pd.to_numeric(col_before, errors='coerce')
                after_NaNs = col_after.isna().sum()
                gained_nan_mask = col_before.notna() & col_after.isna()
                replaced_values = col_before[gained_nan_mask].unique().tolist()
                print(f'{col_name} converted to "numeric" | Values that became NaN: {after_NaNs - before_NaNs} | List: {replaced_values}')
                self.df[ColSpec.name] = col_after

            elif ColSpec.kind == 'timedelta':
                self.df[ColSpec.name] = self.df[ColSpec.name].astype('timedelta64[ns]')
                print(f'{col_name} converted to "timedelta"')

            elif ColSpec.kind == 'datetime':
                col_before = self.df[ColSpec.name].copy()
                before_NaNs = col_before.isna().sum()
                col_after = pd.to_datetime(self.df[ColSpec.name], format='mixed', dayfirst=True, errors='coerce')
                after_NaNs = col_after.isna().sum()
                gained_nan_mask = col_before.notna() & col_after.isna()
                replaced_values = col_before[gained_nan_mask].unique().tolist()
                print(f'{col_name} converted to "datetime" | Values that became NaN: {after_NaNs - before_NaNs} | List: {replaced_values}')
                self.df[ColSpec.name] = col_after

            elif ColSpec.kind == 'categorical':
                if isinstance(ColSpec.ordered, tuple):
                    self.df[ColSpec.name] = self.df[ColSpec.name].astype("Int64")
                    col_before = self.df[ColSpec.name].copy()
                    before_NaNs = col_before.isna().sum()
                    col_after = pd.Categorical(self.df[ColSpec.name], list(ColSpec.ordered), ordered=True)
                    after_NaNs = col_after.isna().sum()
                    gained_nan_mask = col_before.notna() & col_after.isna()
                    replaced_values = col_before[gained_nan_mask].dropna().unique().tolist()
                    print(f'{col_name} converted to "Categorical (ordered)" | Values that became NaN: {after_NaNs - before_NaNs} | List: {replaced_values}')
                    self.df[ColSpec.name] = col_after

                else:
                    self.df[ColSpec.name] = pd.Categorical(self.df[ColSpec.name], ordered=False)
                    print(f'{col_name} converted to "Categorical (non-ordered)"')

            elif ColSpec.kind == 'text':
                self.df[ColSpec.name] = self.df[ColSpec.name].astype('string[python]').str.strip()
                print(f'{col_name} converted to "text"')

            else:
                raise ValueError(f'❌ No such dtype: {ColSpec.kind}')
        # ===== feedback =====
        print(f"{GREEN}{BOLD}✔ Done:{RESET} all columns converted to target dtypes\n")
        return self

    #==============================================================
    #                  DERIVED FUNCTION DEFINITION 
    #==============================================================
    def apply_derived(self) -> "YlivertainenDataCleaningSurg":
        print("═" * 70)
        print(f"{BOLD}🍼 CREATION OF DERIVED {RESET}")
        print("─" * 70)

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

                print(
                    f"🎯 {BOLD}{col_name}{RESET}: {GREEN}{BOLD}{match_rate}%{RESET}\n"
                    f"of cases labeled as {BOLD}{normalized_criteria}{RESET} in {BLUE}{BOLD}{left_col}{RESET}\n"
                    f"were also classified in the same family in {BLUE}{BOLD}{right_col}{RESET}\n"
                    f"n = {BOLD}{n_evaluable}{RESET} evaluable cases\n"
                )
                # return updated df
                self.df = df

            #============================
            #          DATETIME
            #============================
            elif ColSpec.kind == 'datetime':
                if not ColSpec.derive_from:
                    raise KeyError(f'❌ Forgot derive_from for one of the ColSpec.timedelta in DERIVED')
                if not ColSpec.datetime_units:
                    raise KeyError(f'❌ Forgot datetime_units for one of the ColSpec.timedelta in DERIVED')
                if not ColSpec.name:
                    raise KeyError(f'❌ Forgot name for one of the ColSpec.timedelta in DERIVED')
            
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

                print(f'📅 Created {BLUE}{BOLD}{ColSpec.name}{RESET} derived from  "{source_col}"')
                if ColSpec.name in df.select_dtypes(include="number"):
                    bad_mask = df[ColSpec.name] < 0
                    df.loc[bad_mask, ColSpec.name] = pd.NA
                    bad_count = bad_mask.sum()
                
                if ColSpec.name in df.select_dtypes(include="number"):
                    max_val = df[ColSpec.name].max()
                    min_val = df[ColSpec.name].min()
                    mean_val = round(df[ColSpec.name].mean(), 3)
                    std_val = round(df[ColSpec.name].std(), 3)
                    print(f'{BOLD}MAX:{RESET} {max_val}')
                    if bad_count > 0:
                        print(f'{BOLD}MIN:{RESET} {min_val} ==> also trashed NEGATIVE datetimes: {bad_count}')
                    else:
                        print(f'{BOLD}MIN:{RESET} {min_val} ==> no negative timedeltas found')

                    print(f'{BOLD}Mean:{RESET} {mean_val}')
                    print(f'{BOLD}STD:{RESET} {std_val}\n')
                else:
                    unique_count = df[ColSpec.name].nunique()
                    commonest = df[ColSpec.name].value_counts().head(5).to_dict()
                    rarest = df[ColSpec.name].value_counts().tail(5).to_dict()
                    nan_count = df[ColSpec.name].isna().sum()
                    print(f'{BOLD}Unique count:{RESET} {unique_count}')
                    if unique_count > 5:
                        print(f'{BOLD}Unique first:{RESET} {commonest}')
                        print(f'{BOLD}Unique last:{RESET} {rarest}\n')
                    else:
                        print(f'{BOLD}Uniques:{RESET} {commonest}\n')
        
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

                print(f'⌛ Created {BLUE}{BOLD}{ColSpec.name}{RESET} derived from the difference between "{end_col}" and "{start_col}"')
                bad_mask = df[ColSpec.name] < 0
                df.loc[bad_mask, ColSpec.name] = pd.NA
                bad_count = bad_mask.sum()
                
                max_val = df[ColSpec.name].max()
                min_val = df[ColSpec.name].min()
                mean_val = round(df[ColSpec.name].mean(), 3)
                std_val = round(df[ColSpec.name].std(), 3)
                print(f'{BOLD}MAX:{RESET} {max_val}')
                if bad_count > 0:
                    print(f'{BOLD}MIN:{RESET} {min_val} ==> also trashed NEGATIVE timedeltas: {bad_count}')
                else:
                    print(f'{BOLD}MIN:{RESET} {min_val} ==> no negative timedeltas found')
                print(f'{BOLD}Mean:{RESET} {mean_val}')
                print(f'{BOLD}STD:{RESET} {std_val}\n')


        # ===== feedback =====
        print(f"{GREEN}{BOLD}✔ Derived:{RESET} creation process completed")
        return self
    
    #==============================================================
    #               CLEAN UP AFTER SCHEMA & DERIVED 
    #==============================================================
    def cleanup(self) -> "YlivertainenDataCleaningSurg":
        print("═" * 70)
        print(f"{BOLD}🧹 CLEAN UP {RESET}")
        print("─" * 70)
        
        dropped_columns = []
        for ColSpec in SCHEMA:
            col_name = f'{BLUE}{BOLD}{ColSpec.name}{RESET}'
            if not ColSpec.keep:
                dropped_columns.append(ColSpec.name)
                print(f'🪓 Successfully amputated {col_name}')
        self.df = self.df.drop(columns=dropped_columns)
        # ===== feedback  =====
        print(f"{GREEN}{BOLD}✔ Cleaned up:{RESET} {len(dropped_columns)} unnecessary columns dropped")
        return self
    
    #==============================================================
    #                  Creating NaN feature flags 
    #==============================================================
    def apply_nan_features(self) -> "YlivertainenDataCleaningSurg":
        print("═" * 70)
        print(f"{BOLD} ⭕ NaN features Flags {RESET}")
        print("─" * 70)

        df = self.df.copy()
        
        exclude_derived = [ColSpec.name for ColSpec in DERIVED]
        cols = [col for col in df if col not in exclude_derived]
        
        for col in cols:
            null_sum = df[col].isna().sum()
            
            if null_sum > 0:
                df[f'{col}_missing'] = df[col].isna()
                print(f'{BLUE}{BOLD}{col}{RESET} has {RED}{BOLD}{null_sum} NaNs{RESET}. Created feature flag column')

            else:
                print(f'{GRAY}✅ No NaNs found in{RESET} {col}')

        # ===== feedback =====
        total_nans = df.isna().sum().sum()
        if total_nans > 0:
            print(f"{GREEN}{BOLD}✔ NaN feature cols:{RESET} created NaN feature columns")
        else:
            print(f"{GREEN}{BOLD}✔ NaN feature cols:{RESET} {total_nans} NaNs found in the whole dataset")
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
        print("═" * 70)
        print(f"{BOLD}🔍 DUPE SEARCH {RESET}")
        print("─" * 70)

        id_cols, skipsfirst_dupe_mask, includesfirst_dupe_mask = self._resolve_duplicate_masks(id_cols)

        if len(id_cols) == 0:
            print(f'{BOLD}There are no ID columns{RESET}')
            display(self.df.iloc[0:0])
            return self

        dup_mask = includesfirst_dupe_mask if include_first else skipsfirst_dupe_mask
        dupe_count = int(dup_mask.sum())

        if dupe_count > 0:
            print(f"{GREEN}{BOLD}✔ Dupe search:{RESET}")
            if include_first:
                print(f'There are {dupe_count} rows in duplicate groups based on complete normalized ID: {id_cols}')
                print('Note: includes the first row in each duplicate group. Rows with incomplete IDs were ignored')
            else:
                print(f'There are {dupe_count} later duplicate rows based on complete normalized ID: {id_cols}')
                print('Note: not including the first. Rows with incomplete IDs were ignored')
        else:
            print(f'✅ No duplicates found based on complete normalized ID: {id_cols}')
            print(f"{GREEN}{BOLD}✔ Dupe search:{RESET} no duplicates found based on complete normalized ID: {id_cols}")

        dupe_df = self.df.loc[dup_mask]
        display(dupe_df)

        if drop:
            print("═" * 70)
            print(f"{BOLD}🪚 DUPE REMOVAL {RESET}")
            print("─" * 70)

            id_cols, skipsfirst_dupe_mask, _ = self._resolve_duplicate_masks(id_cols)

            if len(id_cols) == 0:
                print(f"{GREEN}{BOLD}✔ Dupe removal:{RESET} no ID columns provided — skipping duplicate removal")
                return self

            dupe_count = int(skipsfirst_dupe_mask.sum())

            print(f"{GREEN}{BOLD}✔ Dupe removal:{RESET} removed {dupe_count} later duplicates based on complete normalized ID: {id_cols}")

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

            print(f'{BLUE}{BOLD}===== {col} ====={RESET}')
            
            if col in categorical_cols or col in bool_cols:

                dt = self.df[col].dtype
                unique_count = self.df[col].nunique()
                commonest = self.df[col].value_counts().head(5).to_dict()
                rarest = self.df[col].value_counts().tail(5).to_dict()
                nan_count = self.df[col].isna().sum()

                print(f'{BOLD}Dtype:{RESET} {dt}')
                print(f'{BOLD}Unique count:{RESET} {unique_count}')
                if unique_count > 5:
                    print(f'{BOLD}Unique first:{RESET} {commonest}')
                    print(f'{BOLD}Unique last:{RESET} {rarest}')
                else:
                    print(f'{BOLD}Uniques:{RESET} {commonest}')
                if nan_count > 0:
                    print(f'{BOLD}NaN count:{RESET} {RED}{nan_count}{RESET}')
                else:
                    print(f'{BOLD}NaN count:{RESET} {GREEN}{nan_count}{RESET}')

            if col in numerical_cols:
                
                dt = self.df[col].dtype
                max_val = self.df[col].max()
                min_val = self.df[col].min()
                mean_val = round(self.df[col].mean(), 3)
                std_val = round(self.df[col].std(), 3)
                nan_count = self.df[col].isna().sum()

                print(f'{BOLD}Dtype:{RESET} {dt}')
                print(f'{BOLD}MAX:{RESET} {max_val}')
                if 'timedelta' in col:
                    if min_val < 0:
                        print(f'{BOLD}MIN:{RESET} {RED}{min_val}{RESET}')
                else:
                    print(f'{BOLD}MIN:{RESET} {min_val}')
                print(f'{BOLD}Mean:{RESET} {mean_val}')
                print(f'{BOLD}STD:{RESET} {std_val}')
                if nan_count > 0:
                    print(f'{BOLD}NaN count:{RESET} {RED}{nan_count}{RESET}')
                else:
                    print(f'{BOLD}NaN count:{RESET} {GREEN}{nan_count}{RESET}')

        for col in self.df.columns:
            if col not in analyse_those:
                print(f'{BLUE}{BOLD}===== {col} ====={RESET}')
                print(f'{BOLD}Dtype:{RESET} {self.df[col].dtype}')
                print(f'{BOLD}NaN count:{RESET} {self.df[col].isna().sum()}')
                print(f'{BOLD}Unique count:{RESET} {self.df[col].nunique()}')
                print(f'{BOLD}Commonest:{RESET} {self.df[col].value_counts().head(5).to_dict()}')
                print(f'{BOLD}Rarest:{RESET} {self.df[col].value_counts().tail(5).to_dict()}')

    #==============================================================
    #                      COMMIT CLEANED DF 
    #==============================================================
    def cleaning_overview_and_commit(self):
        print("═" * 70)
        print(f"{BOLD}🧠 DATA CLEANING COMPLETE — FRAME STABLE{RESET}")
        print("─" * 70)

        cleaned_df = self.df

        n_rows, n_cols = cleaned_df.shape
        total_nans = cleaned_df.isna().sum().sum()
        nan_pct = (total_nans / (n_rows * n_cols)) * 100 if n_rows and n_cols else 0

        print(f"{GREEN}✔ CLEANED DF ONLINE{RESET}")
        print(f"   Shape         : {n_rows} rows x {n_cols} cols")
        print(f"   Remaining NaNs: {total_nans} cells ({nan_pct:0.2f}%)")
        print("   Status        : noise resected, dtypes aligned, ready for cohort craniotomy 🧠🪓")
        print(f"\n{BOLD}{GREEN}→ Next: pass this straight into cohort — no extra rituals needed.{RESET}")

        display(cleaned_df.head())
        return cleaned_df
