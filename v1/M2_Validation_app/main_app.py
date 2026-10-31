import streamlit as st
import sys
import os

st.set_page_config(page_title="M1/M2 Analytics Suite", layout="wide")
# Mark page config as set to prevent sub-tools from setting it again
os.environ["PAGE_CONFIG_SET"] = "1"

# Sidebar for project selection
st.sidebar.title("Project Selection")
suite = st.sidebar.radio(
    "",
    ["M1", "M2"],
)

if suite == "M1":
    selected_project = st.sidebar.selectbox(
        "Choose a project:",
        [
            "M1 Model Validation Tool",
            "M1 6 Months Forecasting Tool",
            "M1 Underlying Table",
        ]
    )
else:
    selected_project = st.sidebar.selectbox(
        "Choose a project:",
        [
            "M2 Model Validation Tool",
            "M2 6 Months Forecasting Tool",
            "Underlying Table",
        ]
    )

if selected_project == "M1 Model Validation Tool":
    exec(open('M1_Validation_app.py', 'r', encoding='utf-8').read())

elif selected_project == "M1 6 Months Forecasting Tool":
    exec(open('forecast_tool_4.py', 'r', encoding='utf-8').read())

elif selected_project == "M1 Underlying Table":
    exec(open('m1_underlying_table.py', 'r', encoding='utf-8').read())

elif selected_project == "M2 Model Validation Tool":
    # Import and run the validation tool
    exec(open('M2_Validation_app_gcs_only.py', 'r', encoding='utf-8').read())
    
elif selected_project == "M2 6 Months Forecasting Tool":
    # Import and run the forecasting tool
    exec(open('forecast_tool_m2.py', 'r', encoding='utf-8').read())
    
elif selected_project == "Underlying Table":
    # Import and run the underlying table tool
    exec(open('underlying_table.py', 'r', encoding='utf-8').read()) 