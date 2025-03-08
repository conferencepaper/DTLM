import os
open_ai_key="Insert your key"
os.environ["OPENAI_API_KEY"] = open_ai_key
#https://console.mistral.ai/api-keys/
mistral_ai_key="Insert your key"
os.environ["MISTRAL_API_KEY"] = mistral_ai_key
claude_key="Insert your key"
#https://console.anthropic.com/settings/keys
os.environ["ANTHROPIC_API_KEY"] = claude_key  # Replace with your actual Anthropic API key
