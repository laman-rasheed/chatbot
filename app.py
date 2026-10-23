from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

if langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
else:
    print("LANGSMITH_API_KEY is not set; tracing will be unavailable.")

##prompt template 

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant. Answer the user's question."),
    ("user", "Question: {question}")]
)


##streamlit freamework

st.title("LangChain demo with Ollama")
input_text=st.text_input("Search the topic you want to know about:")

## openai llm

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)
output_parser=StrOutputParser()
chain=prompt | llm | output_parser

if input_text:
    st.write("Answer:", chain.invoke({"question": input_text}))
