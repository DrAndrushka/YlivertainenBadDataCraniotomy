#================================================================================
#                           ⚙️ SETUP for Greatness ⚙️
#================================================================================
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ylivertainen._pathing import setup_repo_path
root = setup_repo_path()

import streamlit as st
import pandas as pd
#================================================================================
#                         🎨 Overall Background Design 🎨
#================================================================================
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {background: #0b0f14;}

    [data-testid="stAppViewContainer"] > .main {background: transparent;}

    [data-testid="stHeader"] {background: rgba(0,0,0,0);}

    [data-testid="stAppViewContainer"]::before {
        content: "🧠";
        position: fixed;
        top: 55%;
        left: 50%;
        right: 18%;
        transform: translate(-50%, -50%);
        font-size: min(90vw, 90vh);
        opacity: 0.4;
        filter: grayscale(0.5) brightness(0.3) blur(9px);
        pointer-events: none;
        z-index: 0;}

    [data-testid="stAppViewContainer"] > .main * {position: relative; z-index: 1;}
    </style>
    """, unsafe_allow_html=True)


def pretty_text(text, align="center", color="white", weight="700", size="25px") -> None:
    text_align = "text-align:" + align
    colorrr = "color:" + color
    font_weight = "font-weight:" + weight
    font_size = "font-size:" + size

    st.markdown(
        f"<div style='{text_align}; {colorrr}; {font_weight}; {font_size};'>{text}</div>",
        unsafe_allow_html=True)
#=============================
# Little Text Just Because...
#=============================
starting_title = "Ylivertainen Bad Data Craniotomy"
starting_text = "This is The Module for awesome doctors such as yourself. Love the hustle, love the push towards Greatness"
happy_path = st.checkbox(label="⚡ YLIVERTAINEN HAPPY PATH ⚡", value=True)
    
if not happy_path:
    st.title(starting_title)
    st.write(starting_text)

#================================================================================
#                              🧹 DATA CLEANING 🧹
#================================================================================
from ylivertainen.cleaning import YlivertainenDataCleaningSurg, pre_merge_check

st.header("🧹 DATA CLEANING 🧹")

#if not happy_path:
#    st.write("============")
#    col1, col2, col3, col4 = st.columns([1,1,1,4])
#    show_dfs = col1.checkbox(label="Show DataFrames", value=False)
#    csvs = pre_merge_check(colname_length=25,show_dfs=show_dfs)


#=====================================
#       Ultimate Column Renamer
#=====================================
RENAME_FINISHED = False

COLUMN_RENAME_MAP = {}

raw_dir = root / "ylivertainen" / "data" / "raw"
csvs = sorted(raw_dir.glob("*.csv"))

if not csvs:
    st.error("❌ No CSV file/-s provided to ylivertainen/data/raw")
    st.stop()

RAW_COLUMNS = list(dict.fromkeys([col.strip() for csv in csvs for col in pd.read_csv(csv, nrows=0).columns]))

dfs_list = [pd.read_csv(csv, nrows=3).reindex(columns=RAW_COLUMNS) for csv in csvs]
PREVIEW_DF = pd.concat(dfs_list, ignore_index=True)

col1,col2,col3,col4 = st.columns([1,0.25,1,1.5])

#====== HELPERS ======
def submit_rename():
    new_col_name = st.session_state.rename_input.strip()
    old_col_name = PREVIEW_DF.columns[st.session_state.current_idx]
    if not new_col_name:
        new_col_name = "__DROP__"
    st.session_state.rename_log.insert(0, f"{new_col_name} --> {old_col_name}")
    st.session_state.current_idx += 1
    st.session_state.rename_input = ""

def undo_rename():
    if st.session_state.rename_log and st.session_state.current_idx > 0:
        st.session_state.rename_log.pop(0)
        st.session_state.current_idx -= 1
        st.session_state.rename_input = ""

# ===== SESSION STATE INIT =====
if "rename_log" not in st.session_state:
    st.session_state.rename_log = []
if "rename_input" not in st.session_state:
    st.session_state.rename_input = ""
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0

# ===== FUNCTION ======
if st.session_state.current_idx < len(RAW_COLUMNS):
    with col1:
        st.write("")
        st.write("")    
        new_col_name = st.text_input(label="input new_col_name", label_visibility="collapsed", key="rename_input", placeholder="new_col_name", on_change=submit_rename)
    with col2:
        st.write("")
        st.write("")
        st.button("Undo", key="undo_rename", on_click=undo_rename, disabled=not st.session_state.rename_log, width="stretch")
    with col3:
        st.dataframe(PREVIEW_DF[[PREVIEW_DF.columns[st.session_state.current_idx]]].head(3))
    with col4:
        pretty_text("🏗️ AWESOME RENAME LOG 🏗️", "center", "white", "600", "30px")
        st.write("")
        for row in st.session_state.rename_log:
            pretty_text(row, "center", "white", "400", "20px")
else:
    for row in st.session_state.rename_log:
        new_name = row.split(" --> ")[0]
        old_name = row.split(" --> ")[1]            
        if new_name not in COLUMN_RENAME_MAP:
            COLUMN_RENAME_MAP[new_name] = [old_name,]
        else:
            COLUMN_RENAME_MAP[new_name].append(old_name)
    COLUMN_RENAME_MAP.pop("__DROP__", None)

    st.session_state["COLUMN_RENAME_MAP"] = dict(COLUMN_RENAME_MAP)
    RENAME_FINISHED = True

#=====================================
#           build a project
#=====================================
if RENAME_FINISHED:
    project = YlivertainenDataCleaningSurg(csvs, st.session_state["COLUMN_RENAME_MAP"])

#=====================================
#          COLUMN RELOCATOR
#=====================================
SEQUENCING_FINISHED = False

# ===== SESSION STATE INIT =====
if "rearrange_idx" not in st.session_state:
    st.session_state.rearrange_idx = 0
if "rearrange_log" not in st.session_state:
    st.session_state.rearrange_log = []
if "rearrange_input" not in st.session_state:
    st.session_state.rearrange_input = ""
if "rearrange_positions" not in st.session_state:
    st.session_state.rearrange_positions = [] 
if "SEQUENCE" not in st.session_state:
    st.session_state.SEQUENCE = []
# ===== HELPERS =====
def submit_rearrange():
    raw = st.session_state.rearrange_input.strip()
    if not raw:
        return
    else:
        loc = None
        try: loc = int(raw)
        except ValueError: 
            col1.warning("LAST WARNING: has to be integer")
            return

        if loc is not None and loc > len(st.session_state.SEQUENCE):
            col1.warning("LAST WARNING: oversized number")
            return
        if loc is not None and loc < 0:
            col1.warning("LAST WARNING: negative number")
            return

    df = project.df
    if st.session_state.rearrange_idx >= len(df.columns):
        return
    col_name = df.columns[st.session_state.rearrange_idx]

    st.session_state.SEQUENCE.insert(loc, col_name)
    st.session_state.rearrange_log.insert(0, f"{loc} <-- {col_name}")
    st.session_state.rearrange_positions.append(loc)
    st.session_state.rearrange_idx += 1
    st.session_state.rearrange_input = ""

def undo_rearrange():    
    if st.session_state.rearrange_positions and st.session_state.SEQUENCE:
        prev_loc = st.session_state.rearrange_positions.pop()
        if 0 <= prev_loc < len(st.session_state.SEQUENCE):
            st.session_state.SEQUENCE.pop(prev_loc)
            st.session_state.rearrange_log.pop(0)
            st.session_state.rearrange_idx -= 1
            st.session_state.rearrange_input = ""

# ===== FUNCTION ======
if RENAME_FINISHED:

    if st.session_state.rearrange_idx < len(project.df.columns):

        preview_col = project.df[[project.df.columns[st.session_state.rearrange_idx]]]

        with col1:
            st.write("")
            st.write("")
            new_col_location = st.text_input(
                label="input the sequence number",
                label_visibility="collapsed",
                key="rearrange_input",
                placeholder="new_col_sequence",
                on_change=submit_rearrange)
        with col2:
            st.write("")
            st.write("")
            st.button("Undo", key="undo_one_step", on_click=undo_rearrange, disabled=not st.session_state.rearrange_log, width="stretch")
        with col3:
            st.dataframe(preview_col.head(3))
            st.write(project.df.columns)    
        with col4:
            pretty_text("🔄 AWESOME REARRANGE LOG 🔄", "center", "white", "600", "30px")
            st.write(st.session_state.SEQUENCE)
            
    else:
        SEQUENCING_FINISHED = True

if SEQUENCING_FINISHED:
    project.df = project.df[st.session_state.SEQUENCE]
#=====================================
#      SCHEMA AND DERIVED WRITER
#=====================================
from dataclasses import dataclass
from typing import Literal
import numpy as np

SCHEMA_FINISHED = False
DERIVED_FINISHED = False
# ===== HELPERS =====
@dataclass(frozen=True)
class ColSpec:
    # ===== SCHEMA =====
    name: str
    kind: Literal["numeric", "timedelta", "datetime", "categorical", "text", "boolean", "match", "delta"]
    nulls: tuple[object, ...] | None = None                                                         # one-item tuple ==> trailing comma ("something",)
    replace: dict[object, object] | None = None
    keep: bool = True                                                                               # keep=False = OK to drop after we’re done with it.
    # ===== dtype specific =====
    # Category:
    ordered: tuple[object, ...] | None = None                                                       # one-item tuple ==> trailing comma ("something",)
    # ===== DERIVED =====
    derive_from: tuple[str, str] | None = None                                                   # one-item tuple ==> trailing comma ("something",)
    # Match:
    match_by: tuple[object, ...] | None = None
    # Timedelta:
    timedelta_units: Literal['seconds', 'minutes', 'hours', 'days', 'months'] | None = None
    # Datetime:
    datetime_units: Literal["hour", "dow", "workday_bool", "month_name", "year"] | None = None

    def __post_init__(self):
        #print("DEBUG:", self.kind, self.timedelta_from)
        if self.kind != "categorical" and self.ordered is not None:
                raise ValueError("❌ trying to order non-categorical data")
        if self.ordered is not None and len(self.ordered) <= 1:
                raise ValueError("❌ While ordering categoricals: input is <=1 value")
        if self.kind == 'timedelta' and not self.timedelta_units:
                raise ValueError("❌ Specify which units to put in timedelta")
        if self.kind == 'match' and not self.match_by:
                raise ValueError("❌ Specify how to create the match column")
        if self.ordered is not None and self.nulls is not None:
                raise ValueError("❌ Redundant 'nulls' added. Unused values in 'ordered' are automatically filtered out")

# ====== SESSION STATE INIT ======
if "schema_idx" not in st.session_state:
    st.session_state.schema_idx = 0
if "derived_idx" not in st.session_state:
    st.session_state.derived_idx = 0
if "preview_done" not in st.session_state:
    st.session_state.preview_done = False
# ===== FUNCTION =====
SCHEMA = []
DERIVED = []

if SEQUENCING_FINISHED and not st.session_state.preview_done:
    col1, col2 = st.columns([1, 10])

    col2.write(project)
    col1.button(
        "Undo",
        key="back_to_rearranging",
        on_click=undo_rearrange,
        disabled=not st.session_state.rearrange_log,
        width="stretch")
    
    st.dataframe(project.df.head(5))    

    if col1.button(label="AWESOME STUFF", key="finish_sequenced_preview", width="stretch"):
        st.session_state.preview_done = True
        st.rerun()

if st.session_state.preview_done:

    col1,col2,col3,col4 = st.columns([1,0.25,1.5,0.9])

    with col1:
        if st.button("numeric", key="numeric_dtype", width="stretch"):
            pass
        
        if st.button("categorical", key="categorical_dtype", width="stretch"):
            pass
    
        if st.button("datetime", key="datetime_dtype", width="stretch"):
            pass
        
        if st.button("timedelta", key="timedelta_dtype", width="stretch"):
            pass

        if st.button("boolean", key="boolean_dtype", width="stretch"):
            pass

        for x in range(3):
            st.write("")
        if st.button("Undo", key="undo_to_previous_action", width="stretch"):
            pass

    with col2:
        if st.button("*match", key="derived_match", width="stretch"):
            pass

        if st.button("delta", key="derived_delta", width="stretch"):
            pass
        
        if st.button("text", key="text_dtype", width="stretch"):
            pass

    with col3:
        pretty_text("🔢 Value Overview 🔢")


    with col4:
        pretty_text("🆎 ACTION LOG 🆎")
        
        st.write("A Total NaNs: 5 ✅")
        st.write("'Geralt' --> 'awesome'")
        st.write("'alga' --> 'zoom'")
        st.write("Chopped to NaNs --> ['m', '1', 'gh'] (NaNs: 5)")
        st.write("A --> numeric (NaNs: 2)")

#TODO:
    # choose datatype
    # chop NaNs one-by-one
    # change values one-by-one
    # count NaNs
    # overview at the end

    # delete scheam.py
    # change cleaning.py to fit the function 
        # (real time changing, not the damn schema build and then it all comes together at once)
    #

    for x in range(2):
        st.write("")
    
    st.dataframe(project.df.head(2))




#================================================
#=============== SCHEMA & DERIVED ===============
#================================================
SCHEMA = [
    ColSpec(name="nmpd_diag",
            kind="categorical",
            replace={1: 'G45.9', 2: 'I64', 3: 'I63.9'}),
    ColSpec(name="izrakstisanas_diag",
            kind="categorical"),
    ColSpec(name="vecums",
            kind="numeric"),
    ColSpec(name="dzimums",
            kind="categorical",
            replace={1: 'sieviete', 2: 'vīrietis'}),
    ColSpec(name="GKS",
            kind="categorical",
            ordered=tuple(range(3, 16, 1))),
    ColSpec(name="FastTest",
            kind="categorical",
            ordered=(1, 3, 4, 2)),
    ColSpec(name="izsaukuma_laiks",
            kind="datetime",
            keep=False),
    ColSpec(name="nogadasana_PSKUS_laiks",
            kind="datetime", 
            keep=False),
    ColSpec(name="patient_card_no",
            kind="text",
            keep=True),
]

DERIVED = [
    ColSpec(name="lidzPSKUS_timedelta_minutes",
            kind="timedelta",
            timedelta_units='minutes',
            derive_from=("izsaukuma_laiks", "nogadasana_PSKUS_laiks"),
            keep=False),
]

















#================================================================================
#                                 🚧 COHORT 🚧
#================================================================================
from ylivertainen.config import big_beautiful_print
from ylivertainen.cohort import build_stroke_agreement_cohort

#================================================================================
#                                   🙊 DDA 🙊
#================================================================================
from ylivertainen.dda import YlivertainenDDA

#================================================================================
#                                   🎨 EDA 🎨
#================================================================================
from ylivertainen.eda import YlivertainenEDA

#================================================================================
#                            🔮 PREDICTIVE MODELING 🔮
#================================================================================
from ylivertainen.predictive_modeling import build_model_frame

#================================================================================
#                                🧙‍♂️ INFERENTIAL 🧙‍♂️
#================================================================================
from ylivertainen.inferential import YlivertainenInferential

#================================================================================
#                                🤖 PREDICTIVE 🤖
#================================================================================
from ylivertainen.predictive import YlivertainenPredictive


#================================================================================
#                             💪🧠 AWESOME REPORT 💪🧠
#================================================================================
from ylivertainen.the_report import YlivertainenTheReport



