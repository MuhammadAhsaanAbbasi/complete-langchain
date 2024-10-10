import streamlit as st
import requests

def get_response(prompt: str):
    response = requests.get(f"http://localhost:8000/chat?prompt={prompt}", stream=True)
    return response 

st.title("Multiple Data Sources RAG Pipeline ChatBot")

prompt = st.text_input("Enter your message")

if prompt:
    response = get_response(prompt)
    data = response.json()
    st.write(data["response"])