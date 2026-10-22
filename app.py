from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["Langchain_TRACING_V2"] = "true"
os.environ["Langchain_API_KEY"] = os.getenv("langchain_api_key")
if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "YOUR_NEW_API_KEY":
    st.error("Set a valid OPENAI_API_KEY in your .env file before using the app.")
    st.stop()

##prompt template 

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant. Answer the user's question."),
    ("user", "Question: {question}")]
)


##streamlit freamework

st.title("LangChain demo with OpenAI")
input_text=st.text_input("Search the topic you want to know about:")

## openai llm
llm=ChatOpenAI(model_name="gpt-3.5-turbo")
output_parser=StrOutputParser()
chain=prompt | llm | output_parser

if input_text:
    st.write("Answer:", chain.invoke({"question": input_text}))