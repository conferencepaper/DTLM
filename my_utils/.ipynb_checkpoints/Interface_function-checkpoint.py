
# Import your transformation functions
from my_utils.missing_values import missing_values
from my_utils.pattern import restore_original_list
from my_utils.pattern import shuffle_without_pattern
from my_utils.prompt import create_prompt
from my_utils.Transformation_llmV2 import transform_llm_structuredV3
import tiktoken

import tiktoken

def calculate_optimal_batch_size(model_name, input_data, example_instance, max_alpha=64):
    """Calculates the optimal batch size for a given language model and input data.

    Args:
        model_name (str): The name of the language model.
        input_data (list): A list of input data instances.
        example_instance (dict): An example instance of the input data.
        max_alpha (int): The maximum number of tokens allowed for alpha.

    Returns:
        int: The optimal batch size.
    """

    # Get the encoding for the model
    tiktoken_name = {
        "Mistral AI": "mistral",
        "Mistral AI tiny": "mistral",
        "OpenAI GPT": "gpt-4",
        "Anthropic Claude": "claude"
    }
    encoding = tiktoken.encoding_for_model(tiktoken_name[model_name])

    # Calculate tokens used by the prompt and example instance
    prompt_tokens = len(encoding.encode(str(input_data)))
    example_tokens = len(encoding.encode(str(example_instance)))

    # Get model's max context length
    max_context_length = {
        'gpt-3.5-turbo': 4096,
        'gpt-4': 8192,
        'claude': 8192,
        'mistral': 4096
    }.get(tiktoken_name[model_name], 4096)  # Default to 4096
    instance_token=prompt_tokens/len(input_data)
    max_context_length-=max_alpha
    optimal_batch_size= max_context_length//instance_token

    # Calculate optimal batch size
    if available_tokens > 0:
        optimal_batch_size = max(1, available_tokens // len(input_data))
    else:
        optimal_batch_size = 1

    return optimal_batch_size
def your_transformation_functionV1(input_data, description, example_input_list, example_output_list, model_name, api_key):
    try:
        # Shuffle the input data and get the indices
        shuffled_input, shuffled_index = shuffle_without_pattern(input_data)
        # Create the prompt using the shuffled input and description
        prompt = create_prompt(shuffled_input, description, example_input_list, example_output_list)
        # Transform the data using the selected AI model
        structured_output = transform_llm_structuredV3(prompt, model_name, api_key)
        
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



def your_transformation_function(input_data, description, example_input_list,example_output_list, model_name, api_key):
    try :
        # Shuffle the input data and get the indices
        shuffled_input, shuffled_index = shuffle_without_pattern(input_data)
        # Create the prompt using the shuffled input and description
        prompt = create_prompt(shuffled_input, description, example_input_list,example_output_list)
        # Transform the data using the selected AI model
        structured_output = transform_llm_structuredV3(prompt, model_name, api_key)
        # Check if structured_output is None
        if structured_output is None or len(structured_output)==0:
            st.error("Transformation failed: structured_output is None")
            st.error(f"prompt --->{prompt}")
            st.error(f"model_name-->{model_name}")
            st.error(f"api_key --->{api_key}")
                     
            return None 
        
        # Check if the length of the output matches the input
        if len(structured_output) < len(shuffled_input):
            missing_output = missing_values(shuffled_input, structured_output)
            missing_output_ordered = restore_original_list(missing_output, shuffled_index)
            padding=[""]*(len(structured_output) < len(shuffled_input))
            padding_output=structured_output+padding
            padding_output_ordered=restore_original_list(padding_output, shuffled_index)
        
        else:
            # If the lengths match, restore the original order
            missing_output_ordered = restore_original_list(structured_output, shuffled_index)
        
        # Create a DataFrame to display
        transformed_df = pd.DataFrame({
            'Input': input_data,
            'Transformed': missing_output_ordered,
            'Original_output':padding_output_ordered,
            "Diff": len(shuffled_input) - len(structured_output)
        })
        return transformed_df
    except IndexError as e:
        st.error(f"Index error: {e}")
        st.error(f"Shuffled input length: {len(shuffled_input)}, Structured output length: {len(structured_output)}")
        return None

import streamlit as st
import pandas as pd
import os

# Function to handle file upload
def upload_file():
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df.head(20)
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
import streamlit as st

def handle_description_and_examples2():
    # Step 2: Keyword Description
    description = st.text_area(
        'Describe the transformation you want to perform:', 
        'e.g., Convert temperature from Celsius to Fahrenheit.'
    )

    if st.button("Confirm Description", key="confirm_description"):
        st.session_state["description_confirmed"] = True
        st.session_state["description"] = description

    # Suggest one value from the input we are working with
    if st.session_state.get("columns_confirmed", False):
        # Use the first value from the input data as a suggestion
        suggested_input = str(st.session_state["input"][0])
    else:
        suggested_input = 'e.g., 32°C'

    # Input field for example input
    example_input = st.text_input(
        'Provide an example of the input you expect:', 
        value=suggested_input,
        key="example_input_field"
    )

    # Input field for expected output
    example_output = st.text_input(
        'Provide the expected output for the given input:', 
        value='',
        key="example_output_field"
    )

    # Initialize lists in session state if they don't exist
    if "example_input_list" not in st.session_state:
        st.session_state["example_input_list"] = []
    if "example_output_list" not in st.session_state:
        st.session_state["example_output_list"] = []

    # Add example to the few-shot list when the button is clicked
    if st.button("Add example", key="add_few_shot"):
        st.session_state["example_input_list"].append(example_input)
        st.session_state["example_output_list"].append(example_output)
        st.write(f'Added Example - Input: {example_input} ----> Output: {example_output}')
        st.write(f"Total Examples: {len(st.session_state["example_input_list"])}")

    # Display all added examples
    if st.session_state["example_input_list"]:
        st.write("**All Added Examples:**")
        for idx, (inp, outp) in enumerate(zip(st.session_state["example_input_list"], st.session_state["example_output_list"])):
            st.write(f"{idx+1}. Input: {inp} ----> Output: {outp}")

# Function to capture transformation description and examples
def handle_description_and_examples():
    # Step 2: Keyword Description
    description = st.text_area(
        'Describe the transformation you want to perform:', 
        'e.g., Convert temperature from Celsius to Fahrenheit.'
    )

    if st.button("Confirm Description", key="confirm_description"):
        st.session_state["description_confirmed"] = True
        st.session_state["description"] = description

    # Example of Desired Output
    if st.session_state["columns_confirmed"]:
        example_input = st.text_area(
            'Provide an example of the input you expect (optional):', 
            st.session_state["input"][0]
        )
    else:
        example_input = st.text_area(
            'Provide an example of the input you expect (optional):', 
            'e.g., Input: 32°C'
        )

    if st.session_state["description_confirmed"] and st.session_state["columns_confirmed"]:
        example_output = st.text_area(
            'Provide an example of the output you expect (optional)', 
            f'{st.session_state["description"]}: {st.session_state["input"][0]}'
        )
    else:
        example_output = st.text_area('Provide an example of the output you expect (optional)')

    # Add example to the few-shot list
    if st.button("Add example", key="add_few_shot"):
        st.session_state["few_shot_added"] = True
        st.session_state["example_input"].append(example_input)
        st.session_state["example_output"].append(example_output)
        st.write(f'Added Example - Input: {example_input},  ----> Output: {example_output}')
        st.write(f"Total Examples: {len(st.session_state['example_input'])}")
import streamlit as st

def handle_description_and_examplesV3():
    # Step 1: Keyword Description (required)
    description = st.text_area(
        'Describe the transformation you want to perform:',
        st.session_state.get('description', 'e.g., Convert temperature from Celsius to Fahrenheit.'),
        key='description_area'
    )

    if st.button("Confirm Description", key="confirm_description"):
        st.session_state["description_confirmed"] = True
        st.session_state["description"] = description

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

        # Use placeholders for the input and output text areas
        input_placeholder = st.empty()
        output_placeholder = st.empty()

        # Initialize variables to store the input and output values
        #default_input_value = 'e.g., 32°C'
        default_input_value=str(st.session_state["input"][0])
        default_output_value = f'{st.session_state["description"]}: Output for {default_input_value}:'

        # Render the input and output text areas within the placeholders
        with input_placeholder:
            example_input = st.text_area(
                'Provide an example of the input you expect:',
                value=default_input_value
            )

        with output_placeholder:
            example_output = st.text_area(
                'Provide the corresponding output for the input:',
                value=default_output_value
            )

        cols = st.columns(3)

        # Add example button
        if cols[0].button("Add Example", key="add_example"):
            if example_input.strip() and example_output.strip():
                # Add both input and output to the session state lists
                st.session_state["example_input_list"].append(example_input)
                st.session_state["example_output_list"].append(example_output)

                # Display success message
                st.success(f"Added Example - Input: {example_input} ----> Output: {example_output}")
                st.write(f"Total Examples: {len(st.session_state['example_input_list'])}")

                # Clear the input and output areas for the next example
                # Empty the placeholders and re-render the text areas with empty values
                input_placeholder.empty()
                output_placeholder.empty()

                with input_placeholder:
                    example_input = st.text_area(
                        'Provide an example of the input you expect:',
                        value=''
                    )

                with output_placeholder:
                    example_output = st.text_area(
                        'Provide the corresponding output for the input:',
                        value=''
                    )
            else:
                st.error("Both input and output fields are required to add an example.")

        # Delete last example button
        if cols[1].button("Delete Last Example", key="delete_last_example"):
            if st.session_state["example_input_list"] and st.session_state["example_output_list"]:
                # Remove the last elements from the lists
                removed_input = st.session_state["example_input_list"].pop()
                removed_output = st.session_state["example_output_list"].pop()

                # Display success message
                st.success(f"Deleted Last Example - Input: {removed_input} ----> Output: {removed_output}")
                st.write(f"Total Examples: {len(st.session_state['example_input_list'])}")
            else:
                st.warning("No examples to delete.")

        # Clear all examples button
        if cols[2].button("Clear All Examples", key="clear_all_examples"):
            st.session_state["example_input_list"] = []
            st.session_state["example_output_list"] = []
            st.success("All examples have been cleared.")
            st.write(f"Total Examples: {len(st.session_state['example_input_list'])}")

        # Display current examples
        if st.session_state["example_input_list"]:
            st.write("**Current Examples:**")
            for idx, (inp, outp) in enumerate(zip(st.session_state["example_input_list"], st.session_state["example_output_list"])):
                st.write(f"{idx+1}. Input: {inp} ----> Output: {outp}")
    else:
        st.warning("Please confirm the description and column selection before adding examples.")


def handle_description_and_examplesV2():
    # Step 1: Keyword Description (required)
    description = st.text_area(
        'Describe the transformation you want to perform:',
        'e.g., Convert temperature from Celsius to Fahrenheit.'
    )

    if st.button("Confirm Description", key="confirm_description"):
        st.session_state["description_confirmed"] = True
        st.session_state["description"] = description

    # Check if a new dataset is loaded and reset examples if necessary
    if st.session_state.get("new_dataset_loaded", False):
        st.session_state["example_input"] = []
        st.session_state["example_output"] = []
        st.session_state["new_dataset_loaded"] = False  # Reset the flag

    # Step 2: Input-output example pairs
    if st.session_state.get("description_confirmed", False) and st.session_state.get("columns_confirmed", False):
        # Initialize session state for input-output examples if not done already
        if "example_input" not in st.session_state:
            st.session_state["example_input"] = []
        if "example_output" not in st.session_state:
            st.session_state["example_output"] = []

        # Text area for input example
        example_input = st.text_area(
            'Provide an example of the input you expect:', 
            'e.g., 32°C', 
            key="example_input_area"
        )

        # Text area for output example
        example_output = st.text_area(
            'Provide the corresponding output for the input:', 
            f'{st.session_state["description"]}: Output for your input example', 
            key="example_output_area"
        )

        cols = st.columns(3)

        # Add example button
        if cols[0].button("Add Example", key="add_example"):
            if example_input and example_output:
                # Add both input and output to the session state lists
                st.session_state["example_input"].append(example_input)
                st.session_state["example_output"].append(example_output)

                # Display success message
                st.success(f"Added Example - Input: {example_input} ----> Output: {example_output}")
                st.write(f"Total Examples: {len(st.session_state['example_input'])}")

                # Clear the input and output areas for the next example
                st.session_state["example_input_area"] = ''
                st.session_state["example_output_area"] = ''
                st.experimental_rerun()  # Refresh the app to clear fields
            else:
                st.error("Both input and output fields are required to add an example.")

        # Delete last example button
        if cols[1].button("Delete Last Example", key="delete_last_example"):
            if st.session_state["example_input"] and st.session_state["example_output"]:
                # Remove the last elements from the lists
                removed_input = st.session_state["example_input"].pop()
                removed_output = st.session_state["example_output"].pop()

                # Display success message
                st.success(f"Deleted Last Example - Input: {removed_input} ----> Output: {removed_output}")
                st.write(f"Total Examples: {len(st.session_state['example_input'])}")
            else:
                st.warning("No examples to delete.")

        # Clear all examples button
        if cols[2].button("Clear All Examples", key="clear_all_examples"):
            st.session_state["example_input"] = []
            st.session_state["example_output"] = []
            st.success("All examples have been cleared.")
            st.write(f"Total Examples: {len(st.session_state['example_input'])}")

        # Display current examples
        if st.session_state["example_input"]:
            st.write("**Current Examples:**")
            for idx, (inp, outp) in enumerate(zip(st.session_state["example_input"], st.session_state["example_output"])):
                st.write(f"{idx+1}. Input: {inp} ----> Output: {outp}")
    else:
        st.warning("Please confirm the description and column selection before adding examples.")
        
def handle_description_and_examplesV1():
    # Step 1: Keyword Description (required)
    description = st.text_area(
        'Describe the transformation you want to perform:',
        'e.g., Convert temperature from Celsius to Fahrenheit.'
    )

    if st.button("Confirm Description", key="confirm_description"):
        st.session_state["description_confirmed"] = True
        st.session_state["description"] = description

    # Step 2: Input-output example pairs
    if st.session_state["description_confirmed"] and st.session_state["columns_confirmed"]:
        # Initialize session state for input-output examples if not done already
        if "example_input" not in st.session_state:
            st.session_state["example_input"] = []
        if "example_output" not in st.session_state:
            st.session_state["example_output"] = []

        # Text area for input example
        example_input = st.text_area(
            'Provide an example of the input you expect:', 
            'e.g., 32°C', 
            key="example_input_area"
        )

        # Text area for output example
        example_output = st.text_area(
            'Provide the corresponding output for the input:', 
            f'{st.session_state["description"]}: Output for your input example', 
            key="example_output_area"
        )

        # Add example button
        if st.button("Add Example", key="add_example"):
            if example_input and example_output:
                # Add both input and output to the session state lists
                st.session_state["example_input"].append(example_input)
                st.session_state["example_output"].append(example_output)

                # Display success message
                st.success(f"Added Example - Input: {example_input} ----> Output: {example_output}")
                st.write(f"Total Examples: {len(st.session_state['example_input'])}")

                # Clear the input and output areas for the next example
                st.experimental_rerun()
            else:
                st.error("Both input and output fields are required to add an example.")
    else:
        st.warning("Please confirm the description and column selection before adding examples.")

import streamlit as st
import pandas as pd

# Function to preview or submit transformation
def preview_or_submit2(api_key, selected_model):
    preview = st.button('Preview')
    submit = st.button('Submit')

    if preview:
        # Show the data in st.session_state
        st.subheader('Session State Data Preview')
        
        # Display Description
        st.write("**Description:**")
        st.write(st.session_state.get("description", "No description provided."))

        # Display Selected Columns/Input Data
        st.write("**Selected Columns/Input Data:**")
        input_data = st.session_state.get("input", [])
        if isinstance(input_data, pd.DataFrame):
            st.dataframe(input_data.head())
        else:
            st.write(input_data)

        # Display Example Inputs and Outputs
        st.write("**Example Inputs and Outputs:**")
        example_inputs = st.session_state.get("example_input_list", [])
        example_outputs = st.session_state.get("example_output_list", [])
        if example_inputs and example_outputs:
            for idx, (inp, outp) in enumerate(zip(example_inputs, example_outputs)):
                st.write(f"{idx+1}. Input: {inp} ----> Output: {outp}")
        else:
            st.write("No examples provided.")

        # Display Selected Model
        st.write("**Selected Model:**")
        st.write(selected_model)

        # Display any other session information
        other_keys = set(st.session_state.keys()) - {
            "description", "input", "example_input_list", "example_output_list"
        }
        if other_keys:
            st.write("**Other Session State Data:**")
            for key in other_keys:
                st.write(f"**{key}:** {st.session_state[key]}")

    if submit:
        if st.session_state.get("input") is None or st.session_state.get("description") is None:
            st.error("Please confirm your column selection and description before proceeding.")
        elif not api_key:
            st.error("API key is required to perform the transformation.")
        else:
            # Call your transformation function
            try:
                transformed_df = your_transformation_functionV1(
                    input_data=st.session_state["input"],
                    description=st.session_state["description"],
                    example_input_list=st.session_state.get("example_input_list", []),
                    example_output_list=st.session_state.get("example_output_list", []),
                    model_name=selected_model,
                    api_key=api_key
                )

                if not transformed_df.empty:
                    st.subheader('Transformed Data')
                    st.write(transformed_df.head())

                    # Include session state data into transformed_df
                    # Prepare session state data for inclusion in CSV
                    session_data = {}

                    # Include relevant session state variables
                    session_data['Description'] = st.session_state.get('description', '')
                    session_data['Selected Model'] = selected_model

                    # Join lists into strings for example inputs and outputs
                    session_data['Example Inputs'] = '; '.join(
                        st.session_state.get('example_input_list', [])
                    )
                    session_data['Example Outputs'] = '; '.join(
                        st.session_state.get('example_output_list', [])
                    )

                    # Add these as columns to transformed_df
                    for key, value in session_data.items():
                        transformed_df[key] = value

                    # Provide option to download transformed data
                    csv = transformed_df.to_csv(index=False)
                    st.download_button(
                        label='Download CSV',
                        data=csv,
                        file_name='transformed_data.csv',
                        mime='text/csv'
                    )
            except Exception as e:
                st.error(f"An error occurred during transformation: {e}")

# Function to preview or submit transformation
def preview_or_submit(api_key, selected_model):
    preview = st.button('Preview')
    submit = st.button('Submit')

    if preview or submit:
        if st.session_state["input"] is None or st.session_state["description"] is None:
            st.error("Please confirm your column selection and description before proceeding.")
        elif not api_key:
            st.error("API key is required to perform the transformation.")
        else:
            # Call your transformation function
            try:
                transformed_df = your_transformation_function(
                    input_data=st.session_state["input"],
                    description=st.session_state["description"],
                    example_input_list=st.session_state["example_input"],
                    example_output_list=st.session_state["example_output"],
                    model_name=selected_model,
                    api_key=api_key
                )

                if not transformed_df.empty:
                    st.subheader('Transformed Data')
                    st.write(transformed_df.head())

                    if submit:
                        # Provide option to download transformed data
                        csv = transformed_df.to_csv(index=False)
                        st.download_button(
                            label='Download CSV', 
                            data=csv, 
                            file_name='transformed_data.csv', 
                            mime='text/csv'
                        )
            except Exception as e:
                st.error(f"An error occurred during transformation: {e}")