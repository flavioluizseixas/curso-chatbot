# Laboratório Hands-On: Desenvolvimento de Chatbots com Modelos de Linguagem

Este laboratório hands-on apresenta três experimentos com grau crescente de complexidade, utilizando modelos de linguagem open-source e ferramentas modernas. Cada experimento inclui instruções detalhadas, código comentado em português e perguntas para reflexão. Todos os experimentos utilizam Python e bibliotecas como Gradio para interface, Hugging Face para modelos e outros componentes relevantes.

---

## Experimento 1: Desenvolvimento de um Chatbot com Llama 3.2 e Interface Gradio

**Objetivo**: Criar um chatbot simples utilizando o modelo Llama 3.2 (ou similar, como o Llama 3.1 para compatibilidade local) baixado do Hugging Face, com uma interface web amigável via Gradio e uma API compatível com OpenAI.

**Pré-requisitos**:
- Python 3.8+
- Bibliotecas: `transformers`, `gradio`, `torch`, `openai` (para compatibilidade de API)
- GPU recomendada para desempenho, mas CPU é suficiente para testes
- Modelo Llama 3.1 baixado do Hugging Face (ex.: `meta-llama/Llama-3.1-8B-Instruct`)

**Instruções**:
1. Instale as dependências:
   ```bash
   pip install transformers gradio torch openai
   ```
2. Baixe o modelo Llama 3.1 do Hugging Face (necessita de autenticação para modelos restritos).
3. Execute o código abaixo para criar o chatbot com interface Gradio e API compatível com OpenAI.

**Código**:
```python
# Importações
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch
from openai import OpenAI

# Configuração do modelo
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Substitua pelo caminho local se baixado
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Função para gerar resposta
def gerar_resposta(mensagem, historico=[]):
    # Formatar entrada com histórico
    inputs = tokenizer(mensagem, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=512,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )
    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resposta

# Interface Gradio
def chatbot_interface(mensagem):
    resposta = gerar_resposta(mensagem)
    return resposta

# Configuração da API compatível com OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="fake-key")  # Simulação local

def api_resposta(mensagem):
    response = client.chat.completions.create(
        model="llama-3.1-8b",
        messages=[{"role": "user", "content": mensagem}]
    )
    return response.choices[0].message.content

# Lançar interface
iface = gr.Interface(
    fn=chatbot_interface,
    inputs="text",
    outputs="text",
    title="Chatbot Llama 3.1",
    description="Converse com um modelo de linguagem local!"
)
iface.launch()

# Para testar a API, execute em outro terminal:
# python -m http.server 8000
# Então use a função api_resposta("Olá, como você está?")
```

**Anunciado do Experimento**:
Neste experimento, você construirá um chatbot funcional baseado no Llama 3.1, hospedado localmente. A interface Gradio permitirá interagir via navegador, e a API compatível com OpenAI simulará integrações externas. O foco é entender como carregar modelos pré-treinados, tokenizar entradas e gerar respostas.

**Perguntas para Reflexão**:
1. Como o parâmetro `temperature` afeta as respostas do modelo? Teste valores entre 0.1 e 1.5.
2. Qual é a vantagem de usar `torch_dtype=torch.float16` no carregamento do modelo?
3. Por que a API compatível com OpenAI é útil para integração com outros sistemas?
4. Como o tamanho do modelo (ex.: 8B vs 70B) impacta o desempenho e os requisitos de hardware?

---

## Experimento 2: Chatbot com Pipeline RAG e PDF como Fonte

**Objetivo**: Estender o chatbot do Experimento 1 para incorporar Retrieval-Augmented Generation (RAG), utilizando conteúdo de um arquivo PDF como base de conhecimento. O pipeline usará modelos de tokenização e vetorização do Hugging Face, mantendo a interface Gradio.

**Pré-requisitos**:
- Bibliotecas adicionais: `langchain`, `sentence-transformers`, `PyPDF2`, `faiss-cpu`
- Um arquivo PDF com conteúdo relevante (ex.: um manual ou artigo)
- Dependências do Experimento 1

**Instruções**:
1. Instale as dependências adicionais:
   ```bash
   pip install langchain sentence-transformers PyPDF2 faiss-cpu
   ```
2. Coloque um arquivo PDF (ex.: `documento.pdf`) no mesmo diretório do script.
3. Execute o código abaixo para criar o chatbot com RAG.

**Código**:
```python
# Importações
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import gradio as gr
import torch
import PyPDF2

# Configuração do modelo
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Extrair texto do PDF
def extrair_texto_pdf(caminho_pdf):
    with open(caminho_pdf, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        texto = ""
        for page in reader.pages:
            texto += page.extract_text()
    return texto

# Criar base de conhecimento com RAG
def criar_base_rag(caminho_pdf):
    texto = extrair_texto_pdf(caminho_pdf)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    textos = text_splitter.split_text(texto)
    documentos = [Document(page_content=chunk) for chunk in textos]
    
    # Modelo de embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documentos, embedding_model)
    return vectorstore

# Função para gerar resposta com RAG
def gerar_resposta_rag(pergunta, vectorstore):
    # Buscar documentos relevantes
    docs = vectorstore.similarity_search(pergunta, k=3)
    contexto = " ".join([doc.page_content for doc in docs])
    
    # Montar prompt com contexto
    prompt = f"Contexto: {contexto}\nPergunta: {pergunta}\nResposta:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=512,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )
    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resposta

# Interface Gradio
vectorstore = criar_base_rag("documento.pdf")  # Substitua pelo caminho do seu PDF
def chatbot_rag(pergunta):
    resposta = gerar_resposta_rag(pergunta, vectorstore)
    return resposta

iface = gr.Interface(
    fn=chatbot_rag,
    inputs="text",
    outputs="text",
    title="Chatbot RAG com PDF",
    description="Faça perguntas baseadas no conteúdo do PDF!"
)
iface.launch()
```

**Anunciado do Experimento**:
Este experimento aprimora o chatbot com um pipeline RAG, permitindo respostas baseadas em um documento PDF. Você aprenderá a extrair texto, vetorizar conteúdo com embeddings e usar FAISS para busca semântica, integrando tudo ao modelo Llama 3.1.

**Perguntas para Reflexão**:
1. Como o tamanho do `chunk_size` no `RecursiveCharacterTextSplitter` afeta a qualidade das respostas?
2. Por que usamos embeddings do modelo `all-MiniLM-L6-v2` em vez de tokenizar diretamente com o Llama?
3. Como a busca semântica (FAISS) melhora a precisão comparada a um chatbot sem RAG?
4. Qual é o impacto de aumentar o parâmetro `k` na busca de documentos?

---

## Experimento 3: Fine-Tuning do Modelo Llama 3.1

**Objetivo**: Realizar fine-tuning do modelo Llama 3.1 em um dataset personalizado (ex.: perguntas e respostas específicas) para melhorar suas respostas em um domínio particular, mantendo a interface Gradio.

**Pré-requisitos**:
- Bibliotecas adicionais: `trl`, `datasets`, `peft`
- Um dataset em formato JSON ou CSV (ex.: pares de perguntas e respostas)
- Hardware com GPU fortemente recomendado
- Dependências dos experimentos anteriores

**Instruções**:
1. Instale as dependências adicionais:
   ```bash
   pip install trl datasets peft
   ```
2. Prepare um dataset simples (ex.: `dataset.json` com formato `[{"pergunta": "...", "resposta": "..."}, ...]`).
3. Execute o código abaixo para realizar o fine-tuning e testar o modelo ajustado.

**Código**:
```python
# Importações
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import gradio as gr
import torch

# Configuração do modelo
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Configuração do LoRA para fine-tuning eficiente
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Carregar dataset
dataset = load_dataset("json", data_files="dataset.json")
def formatar_exemplo(exemplo):
    return {"text": f"Pergunta: {exemplo['pergunta']}\nResposta: {exemplo['resposta']} <|EOS|>"}

dataset = dataset.map(formatar_exemplo)

# Tokenizar dataset
def tokenizar(exemplo):
    return tokenizer(exemplo["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenizar, batched=True)

# Configurar treinamento
training_args = TrainingArguments(
    output_dir="./resultados_finetune",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Realizar fine-tuning
trainer.train()

# Salvar modelo ajustado
model.save_pretrained("./modelo_finetuned")
tokenizer.save_pretrained("./modelo_finetuned")

# Função para gerar resposta com modelo ajustado
def gerar_resposta_finetuned(pergunta):
    inputs = tokenizer(pergunta, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=512,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )
    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resposta

# Interface Gradio
def chatbot_finetuned(pergunta):
    resposta = gerar_resposta_finetuned(pergunta)
    return resposta

iface = gr.Interface(
    fn=chatbot_finetuned,
    inputs="text",
    outputs="text",
    title="Chatbot Ajustado",
    description="Teste o modelo após fine-tuning!"
)
iface.launch()
```

**Anunciado do Experimento**:
Neste experimento, você ajustará o Llama 3.1 para um domínio específico usando LoRA (Low-Rank Adaptation), uma técnica eficiente de fine-tuning. O foco é aprender a preparar datasets, configurar hiperparâmetros e avaliar melhorias no desempenho do modelo.

**Perguntas para Reflexão**:
1. Como o uso de LoRA reduz os requisitos de memória em comparação com o fine-tuning completo?
2. Por que o parâmetro `r` no LoRA afeta a capacidade de adaptação do modelo?
3. Como você avaliaria se o fine-tuning melhorou as respostas em relação ao modelo original?
4. Quais são os desafios de treinar com datasets pequenos versus grandes?

---