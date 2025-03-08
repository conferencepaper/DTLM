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





def transform_llm_structured(prompt: str, model_name: str, api_key: str):
    try:
        os.environ["MISTRAL_API_KEY"] = api_key

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