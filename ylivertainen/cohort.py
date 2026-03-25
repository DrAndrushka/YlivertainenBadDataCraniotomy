#==================================
#              imports 
#==================================
import pandas as pd
from IPython.display import display
from ylivertainen.config import TaskConfig  # TaskConfig is a class that defines the task (used here as a type hint only, not to be confused with the TaskConfig class)

#============ ANSI Escape Codes for formating ============
from ylivertainen.aesthetics_helpers import GREEN, YELLOW, ORANGE, RED, BOLD, BLUE, GRAY, RESET

#================================================
#           APPLY INCLUSION CRITERIA 
#================================================
def apply_inclusion_criteria(cohort_df, inclusion_criteria):

    temp_df = cohort_df.copy()

    for var, crit in inclusion_criteria.items():
        
        if crit == 'non-NaN': 
            exclude_variable = temp_df[var].isna()
            temp_df.loc[exclude_variable, "included_in_cohort"] = False
            temp_df.loc[exclude_variable, "cohort_exclusion_reason"] = "no_" + str(var)

    cohort_df = temp_df
    return cohort_df

#================================================
#              BUILD AGREEMENT COHORT 
#================================================
def build_stroke_agreement_cohort(pickle: pd.DataFrame, task: TaskConfig) -> tuple[pd.DataFrame, dict]:   # Tuple[...] there is a type that takes other types as parameters
    
    if isinstance(pickle, pd.DataFrame):
        cleaned_df = pickle
    else:
        raise ValueError(f'❌ Incorrect pickle format. Only {BLUE}{BOLD}pd.DataFrame{RESET} allowed')
    # Define df
    cohort_df = cleaned_df.reset_index(drop=True).copy()

    # Create a rock-solid row_id which will follow everywhere (EVERYWHERE)
    cohort_df['row_id'] = cohort_df.index
    
    # Create defaults for inclusion and exclusion
    cohort_df['included_in_cohort'] = True
    cohort_df['cohort_exclusion_reason'] = None

    #=========================
    #      Apply Criteria
    #=========================
    # Create a mask
    exclude_missing_target = cohort_df[task.target_column].isna()    # NaN exclusion mask
    # Change bool to False and give the reason for exclusion
    cohort_df.loc[exclude_missing_target, 'included_in_cohort'] = False
    cohort_df.loc[exclude_missing_target, 'cohort_exclusion_reason'] = 'target_is_NaN'

    # Apple overall inclusion criteria
    cohort_df = apply_inclusion_criteria(cohort_df, task.inclusion_criteria)

    # Know “which task this row belongs to” without juggling separate variables
    cohort_df['task_name'] = task.name

    # Create a unified 'target' column so that the code is simpler
    cohort_df['target'] = cohort_df[task.target_column]
    # Optional drop:
    #cohort_df = cohort_df.drop(columns=[task.target_column])

    metadata = {
        "task_name": task.name,
        "target_col": "target",           # always this
        "positive_class": task.positive_class,
        "task_type": task.task_type,
    }

    display(cohort_df["included_in_cohort"].value_counts())
    display(cohort_df["cohort_exclusion_reason"].value_counts(dropna=False))

    #=================================================#
    #  DROPPING EVERYTHING NOT IN INCLUSION CRITERIA  #
    #=================================================#
    cohort_df = cohort_df[cohort_df.included_in_cohort == True]
    match_unneeded = []
    if 'match' in task.target_column:
        match_unneeded = [col for col in cohort_df.columns if 'match' in col]
    no_missing_missing = [col for col in cohort_df.columns if col.endswith('_missing') and cohort_df[col].sum() == 0]
    drop_cols = (['included_in_cohort', 'cohort_exclusion_reason'] +
                 no_missing_missing +
                 match_unneeded)
    cohort_df = cohort_df.drop(columns=drop_cols, errors='ignore')
    #=================================================#

    display(cohort_df.head())

    return (cohort_df, metadata)