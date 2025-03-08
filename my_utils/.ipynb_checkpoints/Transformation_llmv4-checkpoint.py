import json
import re


#Langchain
from langchain import PromptTemplate, LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import ChatPromptTemplate
import langchain
import warnings
warnings.filterwarnings('ignore')

#Model supporting model json strucutre.
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI  # Or the appropriate LLM class for your model
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

def extract_json_from_response(response):
    # Use regex to find the JSON object
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        json_str = match.group(0)
        return json_str
    return response


def fix_json_string(json_str):
    # Replace single quotes with double quotes
    json_str = json_str.replace("'", '"')
    return json_str



def get_llm(model_name: str, api_key: str):
    if model_name == "OpenAI GPT":
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        return ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=api_key)
    elif model_name == "Anthropic Claude":
        if not api_key:
            raise ValueError("Anthropic API key is required.")
        return ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0, anthropic_api_key=api_key)
    elif model_name == "Mistral AI" or model_name == "Mistral AI tiny":
        if not api_key:
            raise ValueError("Mistral API key is required.")
        if model_name == "Mistral AI":
            repo_id = "mistral-large-latest"
        elif model_name == "Mistral AI tiny":
            repo_id = "mistral-tiny"
        return ChatMistralAI(model=repo_id, temperature=0,Mmistral_api_key=api_key)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def transform_llm_structured(prompt_in, model_name, api_key):
    try:
        # Adjusted prompt template and format instructions
        format_instructions = """Please output the result strictly in JSON format as per the following schema:

{
    "values": [list of strings]
}

Ensure that:
- The JSON uses double quotes (") for keys and string values.
- There is no additional text or explanations before or after the JSON.
- The list of values is properly formatted as a JSON array of strings.
"""

        prompt_template = PromptTemplate(
            template="{prompt}\n\n{format_instructions}",
            input_variables=["prompt", "format_instructions"]
        )

        # Get the appropriate LLM
        llm = get_llm(model_name, api_key)
        if llm is None:
            raise ValueError(f"Model '{model_name}' not found.")

        # Initialize the LLM chain
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)

        # Run the chain
        response = llm_chain.run(
            prompt=prompt_in,
            format_instructions=format_instructions
        )

        # Print the LLM response for debugging
        print("LLM Response:", response)

        # Extract the JSON part of the response
        json_str = extract_json_from_response(response)
        if json_str is None:
            raise ValueError("Failed to find JSON in the LLM response.")
            

        # Fix the JSON string
        json_str = fix_json_string(json_str)

        # Parse the JSON
        structured_output = json.loads(json_str)

        # Return the structured values
        return structured_output.get("values", response)

    except Exception as e:
        print(f"Error during LLM transformation: {e}")
        if 'response' in locals():
            print(f"LLM Response that caused the error: {response}")
        return []