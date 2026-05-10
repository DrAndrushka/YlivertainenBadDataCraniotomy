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

if not happy_path:
    st.write("============")
    col1, col2, col3, col4 = st.columns([1,1,1,4])
    show_dfs = col1.checkbox(label="Show DataFrames", value=False)
    csvs = pre_merge_check(
        colname_length=25,
        show_dfs=show_dfs
        )


#==============================
#    Ultimate Column Renamer
#==============================

#====== HELPERS ======
def pretty_text(text, align, color, weight, size) -> None:
    text_align = "text-align:" + align
    colorrr = "color:" + color
    font_weight = "font-weight:" + weight
    font_size = "font-size:" + size

    st.markdown(
        f"<div style='{text_align}; {colorrr}; {font_weight}; {font_size};'>{text}</div>",
        unsafe_allow_html=True)

def submit_rename():
    new_col_name = st.session_state.rename_input.strip()
    if new_col_name:
        st.session_state.rename_log.insert(0, f"{new_col_name} --> {PREVIEW_DF.columns[st.session_state.current_idx]}")
        st.session_state.current_idx += 1
        st.session_state.rename_input = ""


#===========
#Example
#===========
#COLUMN_RENAME_MAP = {'nmpd_diag': ('NMP pamata diagnoze: G45.9 -1; I64 - 2; I63.9 - 3',),}
#===========
raw_dir = root / "ylivertainen" / "data" / "raw"
csvs = sorted(raw_dir.glob("*.csv"))
RAW_COLUMNS = list(dict.fromkeys([col for csv in csvs for col in pd.read_csv(csv, nrows=0).columns]))
dfs_list = [pd.read_csv(csv, nrows=3).reindex(columns=RAW_COLUMNS) for csv in csvs]
PREVIEW_DF = pd.concat(dfs_list, ignore_index=True)


# ===== SESSION STATE INIT =====
if "rename_log" not in st.session_state:
    st.session_state.rename_log = []
if "rename_input" not in st.session_state:
    st.session_state.rename_input = ""
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0

# ===== TEMPORARY VALUES (to delete afterwards) =====


# ===== OUTPUT =====
col1,col2,col3,col4 = st.columns([1,0.1,1,1.5])

if st.session_state.current_idx < len(RAW_COLUMNS):
    with col1:
        new_col_name = st.text_input(label="", key="rename_input", placeholder="new_col_name", on_change=submit_rename)
    with col3:
        st.dataframe(PREVIEW_DF[[PREVIEW_DF.columns[st.session_state.current_idx]]].head(3))
    with col4:
        pretty_text("🏗️ AWESOME RENAME LOG 🏗️", "center", "white", "600", "30px")
        st.write("")

        for row in st.session_state.rename_log:
            pretty_text(row, "center", "white", "400", "20px")


else:
    st.success("YLIVERTAINEN SUCCESS")
    
    pretty_text("🏗️ AWESOME RENAME LOG 🏗️", "center", "white", "600", "30px")
    st.write("")

    for row in st.session_state.rename_log:
        pretty_text(row, "center", "white", "400", "20px")



    #TODO: creates COLUMN_RENAME_MAP in the background
    
    #TODO: decide how to make it very fast to arrange the columns
        # no need for mouse (can do with just keyboard)
        # very fast and intuitive
        # same idea like in the UCR but!!! it shows the column (just one like in UCR) 
        # and I input the number where to place it relatively to the other columns that I have right now in the shown df 
            # creates the DF in real time
            
            # TEXT BOX   --   COLUMN NAME      
            # (below) DF.head(3) that updates with every input

    #TODO: show df with all new names (.head(5))
        # with the button to show/hide it

    
    
    





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





