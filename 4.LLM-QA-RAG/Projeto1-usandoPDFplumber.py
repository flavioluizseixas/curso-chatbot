from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Document
import pdfplumber
import os
import re

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Carregar os documentos a partir dos PDFs extraídos
pdf_texts = []
for pdf_file in os.listdir('data'):
    if pdf_file.endswith('.pdf'):
        pdf_texts.append(extract_text_from_pdf(os.path.join('data', pdf_file)))

def clean_text(text):
    # Remover quebras de linha e espaços extras
    text = re.sub(r'\s+', ' ', text)
    # Remover números de páginas ou cabeçalhos se forem padrão
    text = re.sub(r'Page \d+', '', text)
    return text

cleaned_pdf_texts = clean_text(pdf_texts[0])

Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-large"
)
#documents = [Document(text=t) for t in text_list]
documents = [Document(text=cleaned_pdf_texts)]

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
    