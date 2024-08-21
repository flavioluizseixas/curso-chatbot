from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import streamlit as st
#from langchain.memory import ConversationBufferMemory

import torch

# Load the tokenizer and model for BERT
tokenizer = AutoTokenizer.from_pretrained("pierreguillou/bert-large-cased-squad-v1.1-portuguese")
model = AutoModelForQuestionAnswering.from_pretrained("pierreguillou/bert-large-cased-squad-v1.1-portuguese")

# Page setup
st.set_page_config(page_title="Open AI Agent", page_icon=":sparkles:")

st.title(":robot_face: Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# source: https://pt.wikipedia.org/wiki/Pandemia_de_COVID-19
context = r"""
A pandemia de COVID-19, também conhecida como pandemia de coronavírus, é uma pandemia em curso de COVID-19,
uma doença respiratória causada pelo coronavírus da síndrome respiratória aguda grave 2 (SARS-CoV-2).
O vírus tem origem zoonótica e o primeiro caso conhecido da doença remonta a dezembro de 2019 em Wuhan, na China.
Em 20 de janeiro de 2020, a Organização Mundial da Saúde (OMS) classificou o surto
como Emergência de Saúde Pública de Âmbito Internacional e, em 11 de março de 2020, como pandemia.
Em 18 de junho de 2021, 177 349 274 casos foram confirmados em 192 países e territórios,
com 3 840 181 mortes atribuídas à doença, tornando-se uma das pandemias mais mortais da história.
Os sintomas de COVID-19 são altamente variáveis, variando de nenhum a doenças com risco de morte.
O vírus se espalha principalmente pelo ar quando as pessoas estão perto umas das outras.
Ele deixa uma pessoa infectada quando ela respira, tosse, espirra ou fala e entra em outra pessoa pela boca, nariz ou olhos.
Ele também pode se espalhar através de superfícies contaminadas.
As pessoas permanecem contagiosas por até duas semanas e podem espalhar o vírus mesmo se forem assintomáticas.
"""

def display_chat():
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.write(entry["content"])


def create_answer(question):
    # Definir o comprimento máximo para o truncamento
    max_length = 512  # ou qualquer valor adequado ao seu contexto

    # Tokenize the input question
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=max_length)

    with torch.no_grad():
        outputs = model(**inputs)
    
    # Obter as posições de início e fim da resposta
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    # Converter tokens para string
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

    # Adicionar a resposta ao histórico
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
    })


if question := st.chat_input(placeholder="Let's chat"):
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })
    create_answer(question)
    display_chat()
