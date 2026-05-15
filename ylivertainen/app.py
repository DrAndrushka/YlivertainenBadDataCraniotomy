#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧
#                           ⚙️ SETUP for Greatness ⚙️
#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ylivertainen._pathing import setup_repo_path
root = setup_repo_path()
raw_dir = root / "ylivertainen" / "data" / "raw"

import streamlit as st
import pandas as pd

from ylivertainen.cleaning import YlivertainenDataCleaningSurg, pre_merge_check
from typing import Literal
from dataclasses import dataclass

#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧
#                            🎨 Background Design 🎨
#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧
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

#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧
#                           html text editing function
#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧
def pretty_text(text, align="center", color="white", weight=700, size=25) -> None:
    text_align = "text-align:" + align
    colorrr = "color:" + color
    font_weight = "font-weight:" + str(weight)
    font_size = "font-size:" + str(size) + "px"

    st.markdown(
        f"<div style='{text_align}; {colorrr}; {font_weight}; {font_size};'>{text}</div>",
        unsafe_allow_html=True)

#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧
#                           Starting intro text
#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧
starting_title = "Ylivertainen Bad Data Craniotomy"
starting_text = "This is The Module for awesome doctors such as yourself. Love the hustle, love the push towards Greatness"


#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧
#                           skipping to HAPPY PATH
#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧
happy_path = st.checkbox(label="⚡ YLIVERTAINEN HAPPY PATH ⚡", value=True)

if not happy_path:
    st.title(starting_title)
    st.write(starting_text)


#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧
#                              🧹 DATA CLEANING 🧹
#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧
st.header("🧹 DATA CLEANING 🧹")


#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧 GLOBAL STATES 🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧
DfState = Literal[
    "column_actions",
    "sequencing",]
if "df_state" not in st.session_state:
    st.session_state.df_state: DfState = "column_actions"

ColState = Literal[
    "renaming",
    "datatyping",
    "awaiting_ordered_categories",
    "nan_placing",
    "value_renaming",
    "overview",]
if "col_state" not in st.session_state:
    st.session_state.col_state: ColState = "rename"

if "col_idx" not in st.session_state:
    st.session_state.col_idx = 0
    # 0 1 2 3 4 5 ...

if "project" not in st.session_state:
    st.session_state.project = None
    
    # pd.DataFrame
project = None
if st.session_state.project is not None:
    project = st.session_state.project
    if project is None:
        st.stop()
#🟧
#🟧
#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧 Ultimate Column Renamer 🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧

if "COLUMN_RENAME_MAP" not in st.session_state:
    st.session_state.COLUMN_RENAME_MAP = {}
    # dict: "canonical": ["csv1_old_name", "csv2_old_name",]

if "rename_log" not in st.session_state:
    st.session_state.rename_log = []
    # new_col_name --> old_col_name  |  (new one appended at 0)

if "rename_input" not in st.session_state:
    st.session_state.rename_input = ""
    # str   |   if any number of spaces goes to __DROP__

#🟧🟧🟧🟧🟧 button and text input functionality
def submit_rename():
    new_col_name = st.session_state.rename_input.strip()
    old_col_name = PREVIEW_DF.columns[st.session_state.col_idx]
    if not new_col_name:
        new_col_name = "__DROP__"
    st.session_state.rename_log.insert(0, f"{new_col_name} --> {old_col_name}")
    st.session_state.col_idx += 1
    st.session_state.rename_input = ""

def undo_rename():
    if st.session_state.rename_log and st.session_state.col_idx > 0:
        st.session_state.rename_log.pop(0)
        st.session_state.col_idx -= 1
        st.session_state.rename_input = ""

#🟧🟧🟧🟧🟧 pathing for table input
csvs = sorted(raw_dir.glob("*.csv"))
if not csvs:
    st.error("❌ No CSV file/-s provided to ylivertainen/data/raw")
    st.stop()

RAW_COLUMNS = list(dict.fromkeys([col.strip() for csv in csvs for col in pd.read_csv(csv, nrows=0).columns]))
dfs_list = [pd.read_csv(csv, nrows=3).reindex(columns=RAW_COLUMNS) for csv in csvs]
PREVIEW_DF = pd.concat(dfs_list, ignore_index=True)

#🟧🟧🟧🟧🟧 data division into columns
col1,col2,col3,col4 = st.columns([
    1,      # col 1 - INPUT
    0.25,   # col 2 - UNDO button or rarely used buttons
    1,      # col 3 - data visualisation
    1.5     # col 4 - LOG
    ])

#🟧🟧🟧🟧🟧 FUNCTIONALITY
if st.session_state.df_state == "column_actions":
    
    if st.session_state.col_state == "renaming":

        if st.session_state.col_idx < len(RAW_COLUMNS):
            
            with col1:
                st.write("")
                st.write("")    
                new_col_name = st.text_input(
                    label="--> New Canonical Name (blank=DROP)",
                    label_visibility="collapsed",
                    key="rename_input",
                    placeholder="e.g. age or __DROP__",
                    on_change=submit_rename)
            
            with col2:
                st.write("")
                st.write("")
                st.button("Undo", key="undo_rename", on_click=undo_rename, disabled=not st.session_state.rename_log, width="stretch")
            
            with col3:
                st.dataframe(PREVIEW_DF[[PREVIEW_DF.columns[st.session_state.col_idx]]].head(3))
            
            with col4:
                pretty_text("🏗️ AWESOME RENAME LOG 🏗️", "center", "white", "600", "30px")
                st.write("")
                for row in st.session_state.rename_log:
                    pretty_text(row, "center", "white", "400", "20px")
        else:
            for row in st.session_state.rename_log:
                new_name = row.split(" --> ")[0]
                old_name = row.split(" --> ")[1]            
                if new_name not in st.session_state.COLUMN_RENAME_MAP:
                    st.session_state.COLUMN_RENAME_MAP[new_name] = [old_name,]
                else:
                    st.session_state.COLUMN_RENAME_MAP[new_name].append(old_name)
            st.session_state.COLUMN_RENAME_MAP.pop("__DROP__", None)

            st.session_state.project = YlivertainenDataCleaningSurg(csvs, st.session_state.COLUMN_RENAME_MAP)

            st.session_state.df_state = "sequencing"


#🟧
#🟧
#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧 Ultimate Column Relocator 🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧

if "rearrange_log" not in st.session_state:
    st.session_state.rearrange_log = []
    # loc <-- col_name

if "rearrange_input" not in st.session_state:
    st.session_state.rearrange_input = ""
    # int | NOT negative | NOT oversized | NOT letter

if "rearrange_positions" not in st.session_state:
    st.session_state.rearrange_positions = [] 
    # list[int] where each col is put | new one appends at the end

if "SEQUENCE" not in st.session_state:
    st.session_state.SEQUENCE = []
    # list of columns in a manually input sequence

#🟧🟧🟧🟧🟧 button and text input functionality
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
    if st.session_state.col_idx >= len(df.columns):
        return
    col_name = df.columns[st.session_state.col_idx]

    st.session_state.SEQUENCE.insert(loc, col_name)
    st.session_state.rearrange_log.insert(0, f"{loc} <-- {col_name}")
    st.session_state.rearrange_positions.append(loc)
    st.session_state.col_idx += 1
    st.session_state.rearrange_input = ""

def undo_rearrange():
    
    if st.session_state.rearrange_positions and st.session_state.SEQUENCE:
        
        prev_loc = st.session_state.rearrange_positions.pop()
        
        if 0 <= prev_loc < len(st.session_state.SEQUENCE):
            st.session_state.SEQUENCE.pop(prev_loc)
            st.session_state.rearrange_log.pop(0)
            st.session_state.col_idx -= 1
            st.session_state.rearrange_input = ""

#🟧🟧🟧🟧🟧 FUNCTIONALITY
if st.session_state.df_state == "sequencing":
    
    if st.session_state.col_idx < len(project.df.columns):

        preview_col = project.df[[project.df.columns[st.session_state.col_idx]]]

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
            st.button(
                "Undo",
                key="undo_one_step",
                on_click=undo_rearrange,
                disabled=not st.session_state.rearrange_log,
                width="stretch")

        with col3:
            st.dataframe(preview_col.head(3))
            st.write(project.df.columns)    

        with col4:
            pretty_text("🔄 AWESOME REARRANGE LOG 🔄", "center", "white", "600", "30px")
            st.write(st.session_state.SEQUENCE)
            
    else:
        project.df = project.df[st.session_state.SEQUENCE]
        st.session_state.df_state = "column_actions"
        st.session_state.col_state = "datatyping"

#🟧
#🟧
#🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧 SCHEMA WRITER 🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧🟧

if "preview_done" not in st.session_state:
    st.session_state.preview_done = False

if "category_chosen" not in st.session_state:
    st.session_state.category_chosen = ""

if "SCHEMA" not in st.session_state:
    st.session_state.SCHEMA = []
if "SCHEMA_log" not in st.session_state:
    st.session_state.SCHEMA_log = []
if "SCHEMA_FINISHED" not in st.session_state:
    st.session_state.SCHEMA_FINISHED = False

if "DERIVED" not in st.session_state:
    st.session_state.DERIVED = []
if "DERIVED_FINISHED" not in st.session_state:    
    st.session_state.DERIVED_FINISHED = False

if "undo_stack" not in st.session_state:
    st.session_state.undo_stack = []

#🟧🟧🟧🟧🟧 DEFINE DATACLASS FOR THE SCHEMA
@dataclass(frozen=False)
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
        if self.kind == 'timedelta' and not self.timedelta_units:
                raise ValueError("❌ Specify which units to put in timedelta")
        if self.kind == 'match' and not self.match_by:
                raise ValueError("❌ Specify how to create the match column")
        if self.ordered is not None and self.nulls is not None:
                raise ValueError("❌ Redundant 'nulls' added. Unused values in 'ordered' are automatically filtered out")

#🟧🟧🟧🟧🟧 button and text input functionality
def push_undo_snapshot(col_name: str):
    st.session_state.undo_stack.append({
        "col_idx": st.session_state.col_idx,
        "category_chosen": st.session_state.category_chosen,
        "col_name": col_name,
        "col_state": st.session_state.col_state,
        "col_data": project.df[col_name].copy(deep=True),
        "schema_len": len(st.session_state.SCHEMA),
        "log_len": len(st.session_state.SCHEMA_log),
    })

def undo_last_action():
    if not st.session_state.undo_stack:
        return
    snap = st.session_state.undo_stack.pop()

    project.df[snap["col_name"]] = snap["col_data"].copy(deep=True)

    st.session_state.col_idx = snap["col_idx"]
    st.session_state.category_chosen = snap["category_chosen"]

    while len(st.session_state.SCHEMA) > snap["schema_len"]:
        st.session_state.SCHEMA.pop()
    while len(st.session_state.SCHEMA_log) > snap["log_len"]:
        st.session_state.SCHEMA_log.pop(0)

def submit_ordered_categories():
    col_name = project.df.columns[st.session_state.col_idx]
    
    all_categories = st.session_state.ordered_categories.strip().split()

    st.session_state.SCHEMA_log.insert(0, f"Categories (ordered): {all_categories}")

    col_after = pd.Categorical(project.df[col_name], list(all_categories), ordered=True)
    project.df[col_name] = col_after

    st.session_state.SCHEMA.append(ColSpec(
        name=col_name,
        kind="categorical",
        ordered=tuple(all_categories)))
    st.session_state.SCHEMA_log.insert(0, f"categories: {all_categories}")

    st.session_state.ordered_categories = ""
    
#🟧🟧🟧🟧🟧 FUNCTIONALITY
if st.session_state.col_state == "overview":
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


if st.session_state.df_state == "column_actions":
    
    if st.session_state.col_state == "datatyping":
        
        if st.session_state.col_idx < len(project.df.columns):
            
            col_name = project.df.columns[st.session_state.col_idx]

            if col1.button("numeric", key="numeric_dtype", width="stretch"):
                st.session_state.category_chosen = "numeric"
                st.session_state.col_state = "chosen"
                st.session_state.SCHEMA.append(ColSpec(name=col_name, kind=st.session_state.category_chosen))
                st.session_state.SCHEMA_log.insert(0, f"{col_name} converted to 'numeric'")

            if "ordered_categories" not in st.session_state:
                st.session_state.ordered_categories = ""
            if col1.button("categorical (ordered)", key="categorical_ordered", width="stretch"):
                st.session_state.category_chosen = "categorical_ordered"
                st.session_state.SCHEMA_log.insert(0, f"{col_name} converted to 'Categorical(ordered)'")
                st.session_state.col_state = "awaiting_ordered_categories"
            
            if col1.button("categorical (non-ordered)", key="categorical_non_ordered", width="stretch"):
                st.session_state.category_chosen = "categorical_non_ordered"
                st.session_state.col_state = "chosen"
                st.session_state.SCHEMA.append(ColSpec(name=col_name, kind="categorical"))

                project.df[col_name] = pd.Categorical(project.df[col_name], ordered=False)
                st.session_state.SCHEMA_log.insert(0, f"{col_name} converted to 'Categorical(non-ordered)'")

            if col1.button("boolean", key="boolean_dtype", width="stretch"):
                st.session_state.category_chosen = "boolean"
                st.session_state.col_state = "chosen"
                st.session_state.SCHEMA.append(ColSpec(name=col_name, kind=st.session_state.category_chosen))
                st.session_state.SCHEMA_log.insert(0, f"{col_name} converted to 'boolean'")

            if col1.button("datetime", key="datetime_dtype", width="stretch"):
                st.session_state.category_chosen = "datetime"
                st.session_state.col_state = "chosen"
                st.session_state.SCHEMA.append(ColSpec(name=col_name, kind=st.session_state.category_chosen))
                st.session_state.SCHEMA_log.insert(0, f"{col_name} converted to 'datetime'")

            if col1.button("timedelta", key="timedelta_dtype", width="stretch"):
                st.session_state.category_chosen = "timedelta"
                st.session_state.col_state = "chosen"
                st.session_state.SCHEMA.append(ColSpec(name=col_name, kind=st.session_state.category_chosen))
                st.session_state.SCHEMA_log.insert(0, f"{col_name} converted to 'timedelta'")

            if col2.button("text", key="text_dtype", width="stretch"):
                st.session_state.category_chosen = "text"
                st.session_state.col_state = "chosen"
                st.session_state.SCHEMA.append(ColSpec(name=col_name, kind=st.session_state.category_chosen))
                st.session_state.SCHEMA_log.insert(0, f"{col_name} converted to 'text'")
    

    with col3:
        pretty_text("🔢 Value Overview 🔢")
    
        dt = project.df[col_name].dtype
        counts = project.df[col_name].count()
        unique_count = project.df[col_name].nunique()
        commonest = project.df[col_name].value_counts().head(5).to_dict()
        rarest = project.df[col_name].value_counts().tail(5).to_dict()
        nan_count = project.df[col_name].isna().sum()

        for x in range(2): st.write("")

        pretty_text(f"===== {col_name} =====", weight=600, size=30)
        pretty_text(f'Dtype: {dt}', weight=600, size=20)
        pretty_text(f'Unique count: {unique_count}', weight=600, size=20)
        if unique_count > 5:
            pretty_text(f'Unique first: {commonest}', weight=600, size=20)
            pretty_text(f'Unique last: {rarest}', weight=600, size=20)
        else:
            pretty_text(f'Uniques: {commonest}', weight=600, size=20)
        if nan_count > 0:
            pretty_text(f'NaN count: {nan_count}', weight=600, size=20)
        else:
            pretty_text(f'NaN count: {nan_count}', weight=600, size=20)
    
    if st.session_state.col_state == "await_ordered_categories":
        push_undo_snapshot(col_name)
        ordered_categories = col1.text_input(
            label="input order of categories",
            label_visibility="collapsed",
            key="ordered_categories",
            placeholder="e.g. cat1 cat2 cat3...",
            on_change=submit_ordered_categories)

        # ColSpec added inside the function
        # SCHEMA_log updated inside the function
        # col_state updated inside the function

    with col3:
        if st.session_state.category_chosen in {"numeric", "timedelta", "datetime"}:
            dt = project.df[col_name].dtype
            max_val = project.df[col_name].max()
            min_val = project.df[col_name].min()
            mean_val = round(project.df[col_name].mean(), 3)
            std_val = round(project.df[col_name].std(), 3)
            nan_count = project.df[col_name].isna().sum()

            pretty_text("🔢 Value Overview 🔢")
            pretty_text(f"===== {col_name} =====", weight=600, size=30)
            pretty_text(f'Dtype: {dt}', weight=600, size=20)
            pretty_text(f'MAX: {max_val}', weight=600, size=20)
            pretty_text(f'MIN: {min_val}', weight=600, size=20)
            pretty_text(f'Mean: {mean_val}', weight=600, size=20)
            pretty_text(f'STD: {std_val}', weight=600, size=20)
            if nan_count > 0:
                pretty_text(f'NaN count: {nan_count}', weight=600, size=20)
            else:
                pretty_text(f'NaN count: {nan_count}', weight=600, size=20)            

        if st.session_state.category_chosen in {"categorical_ordered", "categorical_non_ordered", "boolean", "text"}:
            dt = project.df[col_name].dtype
            counts = project.df[col_name].count()
            unique_count = project.df[col_name].nunique()
            commonest = project.df[col_name].value_counts().head(5).to_dict()
            rarest = project.df[col_name].value_counts().tail(5).to_dict()
            nan_count = project.df[col_name].isna().sum()

            for x in range(2): st.write("")

            pretty_text("🔢 Value Overview 🔢")
            pretty_text(f"===== {col_name} =====", weight=600, size=30)
            pretty_text(f'Dtype: {dt}', weight=600, size=20)
            pretty_text(f'Unique count: {unique_count}', weight=600, size=20)
            if unique_count > 5:
                pretty_text(f'Unique first: {commonest}', weight=600, size=20)
                pretty_text(f'Unique last: {rarest}', weight=600, size=20)
            else:
                pretty_text(f'Uniques: {commonest}', weight=600, size=20)
            if nan_count > 0:
                pretty_text(f'NaN count: {nan_count}', weight=600, size=20)
            else:
                pretty_text(f'NaN count: {nan_count}', weight=600, size=20)


    if "nan_values" not in st.session_state:
        st.session_state.nan_values = ""
    def submit_nans():
        col_name = project.df.columns[st.session_state.col_idx]
        col = project.df[col_name].copy()
        nan_input = st.session_state.nan_values.strip()

        if not nan_input:
            return

        nan_before = col.isna().sum()
        col = col.replace(to_replace=nan_input, value=pd.NA)
        nan_after = col.isna().sum()

        st.session_state.SCHEMA_log.insert(0, f"{col_name} Values became NaNs: '{nan_input}' ({nan_after - nan_before})")
        st.session_state.nan_values = ""

        project.df[col_name] = col
    nan_values = col1.text_input(
            label="input values to convert to NaNs",
            label_visibility="collapsed",
            key="nan_values",
            placeholder="one value e.g. ten",
            on_change=submit_nans)
    
    for x in range(2): col1.write("")
    col1.button("Undo", key="undo_one_action", on_click=undo_last_action, width="stretch")

    if col1.button("Finished placing NaNs", key="finish_nans", width="stretch"):
        st.session_state.col_state = "nan_placed"     
    
    
    if "value_replacement" not in st.session_state:
        st.session_state.value_replacement = ""
    def submit_replacement():
        col_name = project.df.columns[st.session_state.col_idx]
        col = project.df[col_name].copy()

        if ":" not in st.session_state.value_replacement:
            col1.warning("Divide old_val and new_val with ':'")
            return
                    
        old_val, new_val = [x.strip() for x in st.session_state.value_replacement.split(":", 1)]
        
        if not old_val or not new_val:
            col1.warning("Has to be values on both sides of ':'")
            return

        col = col.replace(to_replace=old_val, value=new_val)
        
        st.session_state.SCHEMA_log.insert(0, f"{old_val} --> {new_val} @ {col_name}")
        st.session_state.value_replacement = ""

        project.df[col_name] = col
    if st.session_state.col_state == "nan_placed":
        value_replacement = col1.text_input(
                label="input value replacement",
                label_visibility="collapsed",
                key="value_replacement",
                placeholder="one value e.g. old1:new1",
                on_change=submit_replacement)
        for x in range(2): col1.write("")
        col1.button("Undo", key="undo_one_action", on_click=undo_last_action, width="stretch")

        if st.button("Finished replacing values", key="finish_replacing", width="stretch"):
            st.session_state.col_state = "values_replaced"     

    if st.session_state.col_state == "values_replaced":
        st.session_state.col_idx += 1
        st.session_state.col_state = "choose_kind"
    

    with col4:
        pretty_text("🆎 ACTION LOG 🆎")
        st.write("")
        for row in st.session_state.SCHEMA_log:
            pretty_text(row, "center", "white", "400", "20px")
        
if st.session_state.col_idx >= len(project.df.columns):
    st.session_state.SCHEMA_FINISHED = True





        
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

    for x in range(2): st.write("")
    st.dataframe(project.df.head(2))


# ====== COMPLETE OVERVIEW ======
if st.session_state.SCHEMA_FINISHED: #and st.session_state.DERIVED_FINISHED:
    project.explore_values()
    
    for x in range(2): st.write("")

    st.dataframe(project.df.head(5))

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



