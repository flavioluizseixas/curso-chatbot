from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings

llm = OpenAI(api_base='http://localhost:1234/v1', temperature=0.7)
Settings.llm = llm

# Test a simple completion query
response = llm.complete('Who is Lionel Messi?')
print(response)