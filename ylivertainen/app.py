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


def pretty_text(text, align, color, weight, size) -> None:
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

RAW_COLUMNS = list(dict.fromkeys([col.strip() for csv in csvs for col in pd.read_csv(csv, nrows=0).columns]))

dfs_list = [pd.read_csv(csv, nrows=3).reindex(columns=RAW_COLUMNS) for csv in csvs]
PREVIEW_DF = pd.concat(dfs_list, ignore_index=True)

col1,col2,col3,col4 = st.columns([1,0.25,1,1.5])

#====== HELPERS ======
def submit_rename():
    new_col_name = st.session_state.rename_input.strip()
    if new_col_name:
        st.session_state.rename_log.insert(0, f"{new_col_name} --> {PREVIEW_DF.columns[st.session_state.current_idx]}")
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
        st.button("Undo", on_click=undo_rename, disabled=not st.session_state.rename_log)
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
    COLUMN_RENAME_MAP.pop("r", None)

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
if "rearange_positions" not in st.session_state:
    st.session_state.rearange_positions = [] 
if "SEQUENCE" not in st.session_state:
    st.session_state.SEQUENCE = []
# ===== HELPERS =====
def submit_rearrange():
    raw = st.session_state.rearrange_input.strip()
    if not raw:
        return

    loc = int(raw)

    df = project.df
    col_name = df.columns[st.session_state.rearrange_idx]

    st.session_state.rearrange_log.insert(0, f"{loc} <-- {col_name}")
    if col_name not in st.session_state.SEQUENCE:
        st.session_state.SEQUENCE.insert(loc, col_name)
        st.session_state.rearrange_idx += 1
    st.session_state.rearange_positions.append(loc)
    st.session_state.rearrange_input = ""

def undo_rearrange():
    if st.session_state.rearrange_log and st.session_state.rearrange_idx > 0:
        pop_idx = st.session_state.rearrange_idx - 1
        
        if 0 <= pop_idx < len(st.session_state.SEQUENCE):
            st.session_state.SEQUENCE.pop(pop_idx)
        
        st.session_state.rearrange_log.pop(0)
        st.session_state.rearrange_idx -= 1
        st.session_state.rearrange_input = ""
    
    if st.session_state.rearange_positions and st.session_state.SEQUENCE:
        prev_loc = st.session_state.rearange_positions.pop()
        if 0 <= prev_loc < len(st.session_state.SEQUENCE):
            st.session_state.SEQUENCE.pop(prev_loc)

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
            st.button("Undo", on_click=undo_rearrange, disabled=not st.session_state.rearrange_log)
        with col3:
            st.dataframe(preview_col.head(3))
        with col4:
            pretty_text("🔄 AWESOME REARRANGE LOG 🔄", "center", "white", "600", "30px")
            st.write(st.session_state.SEQUENCE)
            st.write(project.df.columns)
            for row in st.session_state.rearrange_log:
                pretty_text(row, "center", "white", "400", "20px")

    else:
        col1, col2 = st.columns([10, 1])
        col2.button("Undo", on_click=undo_rearrange, disabled=not st.session_state.rearrange_log)
        col1.write(project)
        
        project.df = project.df[st.session_state.SEQUENCE]
        st.dataframe(project.df.head())
        SEQUENCING_FINISHED = True







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



