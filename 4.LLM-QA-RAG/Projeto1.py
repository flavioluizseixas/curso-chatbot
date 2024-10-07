from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

reader = SimpleDirectoryReader(input_dir="data")
documents = reader.load_data()

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-large"
)

# ollama
Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()
response = query_engine.query("O que é a Covid-19?")

# Exibir a resposta gerada
print("Resposta gerada:")
print(response.response)

# Exibir o contexto recuperado dos documentos
print("\nContexto recuperado:")
for i, source in enumerate(response.source_nodes):
    print(f"\nTrecho {i+1}:")
    print(source.node.get_text())

#Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
