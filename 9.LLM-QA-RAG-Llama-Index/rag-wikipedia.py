from llama_index.core import Document, SummaryIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.wikipedia import WikipediaReader

#import logging
#logging.basicConfig(level=logging.DEBUG)

from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings

Settings.llm = OpenAI(
    api_base='http://localhost:1234/v1',
    temperature=0.7
)
#print(llm.complete('Who is Lionel Messi?'))

from llama_index.core.schema import TextNode
from llama_index.core import SummaryIndex
nodes = [
TextNode(text="Lionel Messi's hometown is Rosario."),
TextNode(text="He was born on June 24, 1987.")
]
index = SummaryIndex(nodes)
query_engine = index.as_query_engine()
response = query_engine.query(
"What is Messi's hometown?"
)
print(response)


# Carregar dados do Wikipedia sobre Lionel Messi
#loader = WikipediaReader()
#documents = loader.load_data(pages=["Messi Lionel"])

# Parsear documentos em nós
#parser = SimpleNodeParser.from_defaults()
#nodes = parser.get_nodes_from_documents(documents)

# Criar o índice
#index = SummaryIndex(nodes)

# Configurar o Query Engine para utilizar o índice com o LLM customizado
#query_engine = index.as_query_engine()

# Testar a query
#print("Ask me anything about Lionel Messi!")
#response = query_engine.query("Who is Lionel Messi?")
#print(response)