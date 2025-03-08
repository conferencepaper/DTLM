import streamlit as st
import pandas as pd
import os

from my_utils.missing_values import missing_values
from my_utils.pattern import restore_original_list
from my_utils.pattern import shuffle_without_pattern
from my_utils.prompt import create_prompt
#from my_utils.Transformation_llmV2 import transform_llm_structuredV4
from my_utils.Transformation_llmv4 import transform_llm_structured# Function to handle file upload

def your_transformation_function(input_data, description, example_input_list, example_output_list, model_name, api_key):
    try:
        # Shuffle the input data and get the indices
        shuffled_input, shuffled_index = shuffle_without_pattern(input_data)
        # Create the prompt using the shuffled input and description
        prompt = create_prompt(shuffled_input, description, example_input_list, example_output_list)
        # Transform the data using the selected AI model
        structured_output = transform_llm_structured(prompt, model_name, api_key)
        
        # Check if structured_output is None or empty
        if structured_output is None or len(structured_output) == 0:
            st.error("Transformation failed: structured_output is None or empty")
            st.error(f"prompt --->{prompt}")
            st.error(f"model_name-->{model_name}")
            st.error(f"api_key --->{api_key}")
            return None
        
        # Initialize variables
        padding_output_ordered = None
        diff = 0
        
        # Check if the length of the output matches the input
        if len(structured_output) < len(shuffled_input):
            missing_output = missing_values(shuffled_input, structured_output)
            missing_output_ordered = restore_original_list(missing_output, shuffled_index)
            padding = [""] * (len(shuffled_input) - len(structured_output))
            padding_output = structured_output + padding
            padding_output_ordered = restore_original_list(padding_output, shuffled_index)
            diff = len(shuffled_input) - len(structured_output)
        else:
            # If the lengths match, restore the original order
            missing_output_ordered = restore_original_list(structured_output, shuffled_index)
            padding_output_ordered = missing_output_ordered
        
        # Create a DataFrame to display
        transformed_df = pd.DataFrame({
            'Input': input_data,
            'Transformed': missing_output_ordered,
            'Original_output': padding_output_ordered,
            "Diff": diff
        })
        
        return transformed_df
    
    except IndexError as e:
        st.error(f"Index error: {e}")
        st.error(f"Shuffled input length: {len(shuffled_input)}, Structured output length: {len(structured_output)}")
        return None


def upload_file(n):
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df.head(n)
        st.success("File uploaded successfully!")
        st.session_state["new_dataset_loaded"] = True
        # Step 0: Display DataFrame (Optional)
    else:
        st.error("Please upload a CSV file.")
    return uploaded_file

# Function to handle model selection and API key input
def sidebar_config():
    st.sidebar.title("Model Selection and API Keys")
    model_options = ["OpenAI GPT", "Anthropic Claude", "Mistral AI", "Mistral AI tiny"]
    selected_model = st.sidebar.selectbox("Select a model", model_options)

    if selected_model == "OpenAI GPT":
        api_key = st.sidebar.text_input(
            "Enter your OpenAI API Key", 
            type="password", 
            value=os.environ.get("OPENAI_API_KEY", "")
        )
    elif selected_model == "Anthropic Claude":
        api_key = st.sidebar.text_input(
            "Enter your Anthropic API Key", 
            type="password", 
            value=os.environ.get("ANTHROPIC_API_KEY", "")
        )
    elif selected_model in ["Mistral AI", "Mistral AI tiny"]:
        api_key = st.sidebar.text_input(
            "Enter your HuggingFace API Key", 
            type="password", 
            value=os.environ.get("HUGGINGFACE_API_KEY", "")
        )
    else:
        api_key = None

    if api_key:
        os.environ["MISTRAL_API_KEY"] = api_key
        st.session_state["api_key"] = api_key

    return selected_model, api_key


# Function to handle column selection
def select_columns(df):
    columns = df.columns.tolist()
    selected_columns = st.multiselect(
        'Select columns or provide specific data for transformation (optional):', 
        columns
    )
    
    if st.button("Confirm Selection", key="confirm_columns") and selected_columns:
        st.session_state["columns_confirmed"] = True
        df["Input"] = df[selected_columns[0]].astype(str)  # Initialize with the first column
        for col in selected_columns[1:]:
            df["Input"] += "," + df[col].astype(str)
        st.session_state["input"] = list(df["Input"])
        st.subheader('Data To Transform')
        st.write(df[["Input"]].head())  # Show only the "Input" column
def handle_description_and_examples():
    # Step 1: Keyword Description (required)
    description = st.text_area(
        'Describe the transformation you want to perform:',
        st.session_state.get('description', 'e.g., Convert temperature from Celsius to Fahrenheit.'),
        key='description_area'
    )

    # Confirm or reset description
    if st.button("Confirm Description", key="confirm_description"):
        st.session_state["description_confirmed"] = True
        st.session_state["description"] = description
        st.success("Description confirmed!")
    
    if st.session_state.get("description_confirmed", False):
        if st.button("Reset Description", key="reset_description"):
            st.session_state["description_confirmed"] = False
            st.session_state["description"] = ""
            st.info("Description reset, please provide a new description.")

    # Check if a new dataset is loaded and reset examples if necessary
    if st.session_state.get("new_dataset_loaded", False):
        st.session_state["example_input_list"] = []
        st.session_state["example_output_list"] = []
        st.session_state["new_dataset_loaded"] = False  # Reset the flag

    # Step 2: Input-output example pairs
    if st.session_state.get("description_confirmed", False) and st.session_state.get("columns_confirmed", False):
        # Initialize session state for input-output examples if not done already
        if "example_input_list" not in st.session_state:
            st.session_state["example_input_list"] = []
        if "example_output_list" not in st.session_state:
            st.session_state["example_output_list"] = []

        # Suggest a value from the dataset for the input
        default_input_value = str(st.session_state["input"][0])
        default_output_value = f'{st.session_state["description"]}: Output for {default_input_value}:'

        # Input and output text areas for examples
        example_input = st.text_area(
            'Provide an example of the input you expect:',
            value=default_input_value,
            key="example_input_area"
        )

        example_output = st.text_area(
            'Provide the corresponding output for the input:',
            value=default_output_value,
            key="example_output_area"
        )

        cols = st.columns(3)

        # Add example button
        if cols[0].button("Add Example", key="add_example"):
            if example_input.strip() and example_output.strip():
                # Add both input and output to the session state lists
                st.session_state["example_input_list"].append(example_input)
                st.session_state["example_output_list"].append(example_output)
                st.success(f"Added Example - Input: {example_input} → Output: {example_output}")
            else:
                st.error("Both input and output fields are required to add an example.")
        
        # Delete last example button
        if cols[1].button("Delete Last Example", key="delete_last_example"):
            if st.session_state["example_input_list"] and st.session_state["example_output_list"]:
                removed_input = st.session_state["example_input_list"].pop()
                removed_output = st.session_state["example_output_list"].pop()
                st.success(f"Deleted Last Example - Input: {removed_input} → Output: {removed_output}")
            else:
                st.warning("No examples to delete.")

        # Clear all examples button
        if cols[2].button("Clear All Examples", key="clear_all_examples"):
            st.session_state["example_input_list"] = []
            st.session_state["example_output_list"] = []
            st.success("All examples have been cleared.")

        # Display current examples
        if st.session_state["example_input_list"]:
            st.write("### Current Examples:")
            for idx, (inp, outp) in enumerate(zip(st.session_state["example_input_list"], st.session_state["example_output_list"])):
                st.write(f"**{idx + 1}.** Input: `{inp}` → Output: `{outp}`")
        else:
            st.info("No examples added yet.")
    
    else:
        st.warning("Please confirm the description and column selection before adding examples.")


# Function to preview or submit transformation
def preview_or_submit(api_key, selected_model):
    cols = st.columns(2)
    
    # Preview and Submit buttons
    preview = cols[0].button('Preview')
    submit = cols[1].button('Submit')

    # Session state access
    description = st.session_state.get("description", "No description provided.")
    input_data = st.session_state.get("input", [])
    example_inputs = st.session_state.get("example_input_list", [])
    example_outputs = st.session_state.get("example_output_list", [])
    
    # Handle Preview
    if preview:
        st.subheader('Session State Data Preview')

        # Display Description
        st.markdown("### **Description**")
        st.write(description)

        # Display Selected Columns/Input Data
        st.markdown("### **Selected Columns/Input Data**")
        if isinstance(input_data, pd.DataFrame):
            st.dataframe(input_data.head())
        else:
            st.write(input_data)

        # Display Example Inputs and Outputs
        st.markdown("### **Example Inputs and Outputs**")
        if example_inputs and example_outputs:
            for idx, (inp, outp) in enumerate(zip(example_inputs, example_outputs)):
                st.write(f"{idx + 1}. Input: {inp} ----> Output: {outp}")
        else:
            st.write("No examples provided.")

        # Display Selected Model
        st.markdown("### **Selected Model**")
        st.write(selected_model)

        # Display Other Session Information
        other_keys = set(st.session_state.keys()) - {
            "description", "input", "example_input_list", "example_output_list"
        }
        if other_keys:
            st.markdown("### **Other Session State Data**")
            for key in other_keys:
                st.write(f"**{key}:** {st.session_state[key]}")

    # Handle Submit
    if submit:
        if not input_data or not description:
            st.error("Please confirm your column selection and description before proceeding.")
        elif not api_key:
            st.error("API key is required to perform the transformation.")
        else:
            # Call your transformation function
            try:
                transformed_df = your_transformation_function(
                    input_data=input_data,
                    description=description,
                    example_input_list=example_inputs,
                    example_output_list=example_outputs,
                    model_name=selected_model,
                    api_key=api_key
                )

                if not transformed_df.empty:
                    st.subheader('Transformed Data')
                    st.dataframe(transformed_df.head())

                    # Prepare session state data for inclusion in transformed_df
                    session_data = {
                        'Description': description,
                        'Selected Model': selected_model,
                        'Example Inputs': '; '.join(example_inputs),
                        'Example Outputs': '; '.join(example_outputs)
                    }

                    # Add session data as columns to the transformed DataFrame
                    for key, value in session_data.items():
                        transformed_df[key] = value

                    # Allow the user to download the transformed data as CSV
                    csv = transformed_df.to_csv(index=False)
                    st.download_button(
                        label='Download Transformed Data',
                        data=csv,
                        file_name='transformed_data.csv',
                        mime='text/csv'
                    )
            except Exception as e:
                st.error(f"An error occurred during the transformation: {e}")

