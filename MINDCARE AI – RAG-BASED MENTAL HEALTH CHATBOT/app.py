import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq 
import re
from utils import *

# Initialize Pinecone and Sentence Transformer
pc = Pinecone(api_key="cbe27b2f-d42d-4d50-a611-fb4ce34ae7e7")
index = pc.Index("mental-health-responses")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Groq model
groq_api_key = "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"

# Streamlit layout
st.title("Mental Health Chatbot")

# Main chat input
if "input_text" not in st.session_state:
    st.session_state.input_text = ""  # Initialize session state for the input field

user_input = st.text_input(
    "Enter your concern", 
    value=st.session_state.input_text,  
    key="input_text", 
    label_visibility="collapsed",
    placeholder="Type your message here...",
    max_chars=None,
)

# Check if user input exists before querying Pinecone
if user_input:
    # Query Pinecone for response
    response = preprocess_and_query(user_input, index, embedder)

    # If no match is found, show the option to generate a response
    if response == "No matches found." or "No match found" in response:
        st.write("Output: No match found related to your query from Database.")
        st.write("Do you want me to generate one for you?")
        
        if st.button("Generate Response", key="generate_response"):
            solution = generate_solution(user_input, groq_api_key)
            st.write(solution)
    else:
        st.write(f"Bot: {response}")
        
        if st.button("Solution", key="solution"):
            solution = generate_solution(response, groq_api_key)
            st.write(solution)

        if st.button("Talk to Me", key="talktome"):
            talk_to_me(response, groq_api_key)
