import streamlit as st
import pandas as pd
import os

# Import your transformation functions
from my_utils.missing_values import missing_values
from my_utils.pattern import restore_original_list
from my_utils.pattern import shuffle_without_pattern
#from my_utils.
from my_utils.prompt import create_prompt
import os
from my_utils.Interface_functionV1 import upload_file, sidebar_config, select_columns, handle_description_and_examples, preview_or_submit

# Initialize session state variables if not already set
if "example_input" not in st.session_state:
    st.session_state["example_input"] = []

if "example_output" not in st.session_state:
    st.session_state["example_output"] = []

if "columns_confirmed" not in st.session_state:
    st.session_state["columns_confirmed"] = False

if "description_confirmed" not in st.session_state:
    st.session_state["description_confirmed"] = False

if "input" not in st.session_state:
    st.session_state["input"] = None

if "description" not in st.session_state:
    st.session_state["description"] = None

if "few_shot_added" not in st.session_state:
    st.session_state["few_shot_added"] = False

# Main app logic
st.title('Data Transformation Interface')

# Step 1: Upload file
uploaded_file = upload_file(200)
if uploaded_file is not None:
    df = st.session_state["df"]
    # Step 0: Display DataFrame (Optional)
    if st.checkbox('Show raw data', key="show_raw_data"):
        st.subheader('Raw Data')
        st.write(df)
    
    # Sidebar configuration
    selected_model, api_key = sidebar_config()

    # Step 2: Select columns for transformation
    select_columns(df)

    # Step 3: Handle description and few-shot examples
    handle_description_and_examples()
    os.environ["MISTRAL_API_KEY"] = api_key

    # Step 4: Preview or submit transformation
    preview_or_submit(api_key, selected_model)
