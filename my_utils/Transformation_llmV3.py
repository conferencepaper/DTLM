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
# prompt: genertate a dictionnary with llm , ChatOpenAI, ChatMistralAI and ChatAnthropic
import os


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
        os.environ["MISTRAL_API_KEY"] = api_key 
        return ChatMistralAI(model=repo_id, temperature=0,api_key=api_key)
    else:
        raise ValueError(f"Unsupported model: {model_name}")




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