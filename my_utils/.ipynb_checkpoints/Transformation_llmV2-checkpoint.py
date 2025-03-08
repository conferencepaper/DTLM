#Model supporting model json strucutre.
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI


#Langchain
from langchain import PromptTemplate, LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import ChatPromptTemplate
import langchain
import warnings
warnings.filterwarnings('ignore')

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





def transform_llm_structured(prompt: str, model_name: str, api_key: str):
    try:
        # Get the appropriate LLM client
        llm = get_llm(model_name, api_key)
    
        # Initialize the LLM chain with the selected LLM and prompt template
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    
        # Run the chain with the prompt and format instructions
        try:
            response = llm_chain.run(
                prompt=prompt,
                format_instructions=format_instructions
            )
            # Parse the response into structured output
            structured_output = output_parser.parse(response)
            # Return the structured values
            return structured_output.get("values", [])
        except Exception as e:
            print(f"Error during LLM transformation: {e}")
            return []
    except Exception as e:
        print(f"Error during Loading Model: {e}")
# Define the schema for the response (static, only needs to be done once)
response_schema = [
    ResponseSchema(name="values", description="A list of values")
]

# Create a structured output parser using the schema (static initialization)
output_parser = StructuredOutputParser.from_response_schemas(response_schema)

# Get format instructions from the parser (static initialization)
format_instructions = output_parser.get_format_instructions()
def transform_llm_structuredV2(prompt_in, model_name,api_key):
    # Create a prompt template using the provided prompt example and format instructions
    prompt_template = PromptTemplate(
        template="{prompt}\n{format_instructions}",
        input_variables=["prompt", "format_instructions"]
    )

    # Get the appropriate LLM
    llm = get_llm(model_name, api_key)
    if llm is None:
        raise ValueError(f"Model '{model_name}' not found in llm_dict.")
    
    # Initialize the LLM chain with the selected LLM and prompt template
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Run the chain with the prompt and format instructions
    try:
        response = llm_chain.run(
            prompt=prompt,
            format_instructions=format_instructions
        )
        # Parse the response into structured output
        structured_output = output_parser.parse(response)
        # Return the structured values
        return structured_output.get("values", [])
    except Exception as e:
        print(f"Error during LLM transformation: {e}")
        return []
def transform_llm_structuredV4(prompt_in, model_name, api_key, format_instructions=None):
    """
    Transforms the input prompt using the specified LLM model and returns structured output.

    Args:
        prompt_in (str): The input prompt to transform.
        model_name (str): The name of the LLM model to use.
        api_key (str): The API key to access the model.
        format_instructions (str, optional): The format instructions for parsing. Defaults to None.

    Returns:
        list: A list of structured values extracted from the LLM response.
    """
    try:
        # Initialize the LLM with the given model and API key
        llm = get_llm(model_name, api_key)
        if llm is None:
            raise ValueError(f"Model '{model_name}' not found.")

        # Prepare the prompt template
        prompt_template = PromptTemplate(
            template="{prompt}\n{format_instructions}" if format_instructions else "{prompt}",
            input_variables=["prompt", "format_instructions"] if format_instructions else ["prompt"]
        )
        
        # Initialize the LLM chain with the selected LLM and prompt template
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        
        # Prepare input variables for the LLM chain
        prompt_vars = {"prompt": prompt_in}
        if format_instructions:
            prompt_vars["format_instructions"] = format_instructions
        
        # Run the LLM chain and obtain a response
        response = llm_chain.run(**prompt_vars)

        # Parse the LLM response into structured output
        structured_output = output_parser.parse(response)
        
        # Return structured values
        return structured_output.get("values", [])

    except ValueError as ve:
        # Log value errors (e.g., missing model)
        print(f"Value error: {ve}")
        return []

    except KeyError as ke:
        # Handle potential KeyError when accessing the 'values' key
        print(f"Key error when parsing response: {ke}")
        return []

    except Exception as e:
        # Log any other errors
        print(f"An error occurred during LLM transformation: {e}")
        return []

def transform_llm_structuredV3(prompt_in, model_name, api_key):
    # Define the format instructions and prompt template
    prompt_template = PromptTemplate(
        template="{prompt}\n{format_instructions}",
        input_variables=["prompt", "format_instructions"]
    )
    
    # Get the appropriate LLM
    llm = get_llm(model_name, api_key)
    if llm is None:
        raise ValueError(f"Model '{model_name}' not found.")
    
    # Initialize the LLM chain with the selected LLM and prompt template
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Prepare input for the LLM chain
    prompt_vars = {
        "prompt": prompt_in,  # Use the input prompt here
        "format_instructions": format_instructions  # The format instructions from the parser
    }

    # Run the LLM chain with the prompt and format instructions
    try:
        response = llm_chain.run(**prompt_vars)
        
        # Parse the response into structured output
        structured_output = output_parser.parse(response)
        
        # Return the structured values
        return structured_output.get("values", [])
    except Exception as e:
        print(f"Error during LLM transformation: {e}")
        return []
