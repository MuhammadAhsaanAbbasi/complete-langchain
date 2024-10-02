import streamlit as st
import requests

def get_response(prompt:str):
    response = requests.get(f"http://localhost:8000/chat?prompt={prompt}", stream=True)
    return response.json()


# Streamlit Framework
st.title("Multi-Modal Chatbot Automobiles FAQ & Content Generation")

prompt = st.text_input("Enter prompt")

if prompt:
    response = get_response(prompt)
    st.write(response)