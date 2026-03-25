#============ imports ============
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
import numpy as np
from IPython.display import display

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

import os

#============ ANSI Escape Codes for formating ============
from ylivertainen.aesthetics_helpers import GREEN, YELLOW, ORANGE, RED, BOLD, BLUE, GRAY, RESET

#================================================
#=============== Class definition ===============
#================================================
def build_model_frame(post_cohort_df: pd.DataFrame, feature_decisions_df: pd.DataFrame, metadata: dict):
    
    #========== show formatted table ==========
    #========== HELPER ==========
    def highlight_drop(row):
        DROP_TEXT_COLOR = "#818181"
        SIMPLE_IMPUTE_TEXT_COLOR = "#CBB255"
        ADVANCED_IMPUTE_TEXT_COLOR = "#D4605C"
        PREDICTOR_TEXT_COLOR = "#1E9101"
        TARGET_TEXT_COLOR = "#FF8C00"
        if (row["action"] == "drop") | (row["role"] == "id"):
            # use the hex variable in your CSS
            return [f"color: {DROP_TEXT_COLOR};"] * len(row)
        elif row["missing_action"] == "simple_impute":
            return [f"color: {SIMPLE_IMPUTE_TEXT_COLOR};"] * len(row)
        elif row["missing_action"] == "advanced_impute":
            return [f"color: {ADVANCED_IMPUTE_TEXT_COLOR};"] * len(row)
        elif row["role"] == "predictor":
            return [f"color: {PREDICTOR_TEXT_COLOR};"] * len(row)
        elif row["role"] == "target":
            return [f"color: {TARGET_TEXT_COLOR};"] * len(row)
        else:
            return [""] * len(row)
    #============================
    display(feature_decisions_df
        .style
        .apply(highlight_drop, axis=1))
    #==========================================

    task = metadata['task_name']

    target_col = feature_decisions_df[feature_decisions_df.role == 'target'].column_name.item()
    viable_predictors = feature_decisions_df[(feature_decisions_df.role == 'predictor') &
                                             (feature_decisions_df.action != 'drop')
                                             ].column_name.tolist()
    drop_definitive = feature_decisions_df[
        (feature_decisions_df.drop_reason == 'leakage')
        # & (feature_decisions_df.action == 'drop')      <=== for future (PLACEHOLDER)
        ].column_name.tolist()
    
    #========== print loop ==========
    print(f'Feature Decisions DF: {BOLD}{YELLOW}{task}{RESET}')
    print(f'{ORANGE}{BOLD}Target Column:{RESET} {target_col}')
    print(f'{GREEN}{BOLD}Viable Predictors:{RESET} {len(viable_predictors)}')
    for inf_type in feature_decisions_df.inferred_type.unique():
        filter = feature_decisions_df[(feature_decisions_df.inferred_type == inf_type) & 
                                             (feature_decisions_df.role == 'predictor')
                                             ].column_name.tolist()
        if len(filter) > 0:
            print(BLUE + BOLD + f'{inf_type}: ' + RESET + ", ".join(f'"{col}"' for col in filter))
    print(BOLD + RED + 'Non-viable predictors: ' + RESET + ", ".join(f'"{col}"' for col in drop_definitive))
    #================================
    
    #========== 🤖 MEEP-MORB 🤖 ==========
    df = post_cohort_df.copy()

    def _dropbad(df):
        print(f'{BOLD}{GRAY}dropping bad cols...{RESET}')
        df = df.drop(columns=drop_definitive)
        return df

    def _simpleimpute(df, specific_cols=None, show_header=True) -> pd.DataFrame:
        if show_header:
            print(f'{BOLD}{GRAY}running simple impute...{RESET}')

        if specific_cols is None:
            simple_impute_cols = feature_decisions_df[
                feature_decisions_df.missing_action == 'simple_impute'
                ].column_name.tolist()
        else:
            simple_impute_cols = list(specific_cols)

        for col in simple_impute_cols:
            s = df[col]
            nans_before = round(s.isna().sum() / len(s) * 100, 2)
            is_cat = pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s)
            #=== categorical ===
            if is_cat:
                mode_series = s.dropna().mode()
                mode_val = mode_series.iloc[0]
                s = s.fillna(mode_val)
                print(f'{GRAY}-> "{col}" is categorical so using mode')
            #=== non-categorical (numeric)===
            else:
                mediana = s.dropna().median()
                s = s.fillna(mediana)
                print(f'{GRAY}-> "{col}" is non-categorical so using median')

            nans_after = s.isna().sum() / len(s) * 100
            print(f'{col} NaN-count: {nans_before}%->{nans_after}%{RESET}')

            df[col] = s
        
        return df

    def _advancedimpute(df: pd.DataFrame, feature_decisions_df: pd.DataFrame) -> pd.DataFrame:
        print(f'{BOLD}{GRAY}running advanced impute...{RESET}')

        advanced_impute_cols = feature_decisions_df[
            feature_decisions_df.missing_action == 'advanced_impute'
            ].column_name.tolist()

        if not advanced_impute_cols:
            return df

        numeric_advanced_cols = []
        fallback_simple_cols = []
        for col in advanced_impute_cols:
            s = df[col]
            is_numeric_like = (
                pd.api.types.is_numeric_dtype(s)
                and not pd.api.types.is_bool_dtype(s)
            )
            if is_numeric_like:
                numeric_advanced_cols.append(col)
            else:
                fallback_simple_cols.append(col)

        if fallback_simple_cols:
            print(
                f'{GRAY}-> non-numeric advanced_impute cols '
                f'falling back to simple impute: '
                + ", ".join(f'"{col}"' for col in fallback_simple_cols)
                + RESET
            )
            df = _simpleimpute(df, specific_cols=fallback_simple_cols, show_header=False)

        if not numeric_advanced_cols:
            return df

        original_dtypes = df.loc[:, numeric_advanced_cols].dtypes

        # IterativeImputer produces float outputs; do imputation in float space.
        X = df.loc[:, numeric_advanced_cols].to_numpy(dtype=float)

        imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=50,
                random_state=0,
                n_jobs=-1,
            ),
            max_iter=10,
            random_state=0,
        )

        X_imp = imputer.fit_transform(X)
        X_imp_df = pd.DataFrame(X_imp, columns=numeric_advanced_cols, index=df.index)

        for col in numeric_advanced_cols:
            orig_dtype = original_dtypes[col]
            if pd.api.types.is_integer_dtype(orig_dtype):
                floored = np.floor(X_imp_df[col])
                if pd.api.types.is_extension_array_dtype(orig_dtype):
                    df.loc[:, col] = floored.astype("int64")
                else:
                    df.loc[:, col] = floored.astype(orig_dtype)
            else:
                df.loc[:, col] = X_imp_df[col]

        return df
    
    
    print('\n' + BOLD + '='*25 + RESET)
    #========== drop ===========
    df = _dropbad(df)
    #====== simple impute ======
    df = _simpleimpute(df)
    #===== advanced impute =====
    df = _advancedimpute(df, feature_decisions_df)
    #===========================

    important_missing_flags = feature_decisions_df[
        (feature_decisions_df.role == 'predictor') &
        (feature_decisions_df.action != 'drop') &
        (
            feature_decisions_df.notes
            .fillna('')
            .str.contains('high_missingness_flag_important', regex=False)
        )
    ].column_name.map(lambda col: f'{col}_missing').tolist()
    important_missing_flags = [col for col in important_missing_flags if col in df.columns]

    X_cols = list(dict.fromkeys(viable_predictors + important_missing_flags))
    X = df[X_cols]
    y = df[target_col]

    return df, X, y

#==============================================================
#                      EXPORT EDA DF 
#==============================================================
def export_model_frame(root, X, y, metadata, export: bool = False):
    print("═" * 70)
    print(f"{BOLD}🧠 MODEL FRAME ONLINE{RESET}")
    print("─" * 70)

    df_model = pd.concat([X, y], axis=1)

    task_name = metadata["task_name"]

    n_rows, n_cols = df_model.shape
    total_nans = df_model.isna().sum().sum()
    nan_pct = (total_nans / (n_rows * n_cols)) * 100 if n_rows and n_cols else 0

    print(f"{GREEN}✔ CLEAN (X, y) EXTRACTED FOR: {task_name}{RESET}")
    print(f"   Shape         : {n_rows} rows x {n_cols} cols")
    print(f"   Remaining NaNs: {total_nans} cells ({nan_pct:0.2f}%)")
    print("   Contents      : predictors after leakage/missingness rules + target.")
    print()
    print(f"{BOLD}{GREEN}→ Attending note:{RESET} next step is baseline models, not more EDA.")

    display(df_model.head())

    if export:
        model_frame_path = root / 'data' / 'processed' / f'(model_frame){task_name}.pickle'
        df_model.to_pickle(model_frame_path)
        print(f"{GREEN}✔ Saved model frame to: {BOLD}{model_frame_path}{RESET}")
    else:
        print(f"{GRAY}ℹ Set export=True to save model frame to data/processed/{RESET}")

    return df_model