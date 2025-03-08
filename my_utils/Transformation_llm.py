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
llm_dict = {
    "OpenAI": ChatOpenAI(model="gpt-4o",temperature=0),
    "MistralAI": ChatMistralAI(model="mistral-large-latest", temperature=0),
    "MistralAI_tiny": ChatMistralAI(model="mistral-tiny", temperature=0),
    # Assuming you have the necessary API key and setup for ChatAnthropic
    "Anthropic": ChatAnthropic(model_name="claude-3-5-sonnet-20240620",temperature=0)
}

# Define the schema for the response (static, only needs to be done once)
response_schema = [
    ResponseSchema(name="values", description="A list of values")
]

# Create a structured output parser using the schema (static initialization)
output_parser = StructuredOutputParser.from_response_schemas(response_schema)

# Get format instructions from the parser (static initialization)
format_instructions = output_parser.get_format_instructions()
def transform_llm_structured(prompt, model_name):
    # Create a prompt template using the provided prompt example and format instructions
    prompt_template = PromptTemplate(
        template="{prompt}\n{format_instructions}",
        input_variables=["prompt", "format_instructions"]
    )

    # Get the appropriate LLM
    llm = llm_dict.get(model_name)
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