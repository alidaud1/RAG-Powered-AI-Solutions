import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq 
import re

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to query Pinecone with preprocessed text
def preprocess_and_query(query, index, embedder, top_k=1, threshold=0.8):
    preprocessed_query = preprocess_text(query)
    query_embedding = embedder.encode(preprocessed_query).tolist()
    
    result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    if result["matches"]:
        match = result["matches"][0]
        score = match['score']
        if score >= threshold:
            return match['metadata']['output'] if 'metadata' in match and 'output' in match['metadata'] else "No response found in metadata."
        else:
            return "No match found."
    else:
        return "No matches found."

# Function to generate concise solution using the LLM with specific instruction
def generate_solution(pinecone_response, groq_api_key):
    llm = ChatGroq(temperature=0.7, groq_api_key=groq_api_key, model_name="gemma2-9b-it")
    
    prompt = (
        "Instruction: 'If you are a licensed psychologist, please provide this patient with a helpful response to their concern.'\n"
        f"Based on this response: '{pinecone_response}', provide a concise explanation of the problem and then offer an actionable solution to help the user overcome this issue."
    )
    
    response = llm.invoke(prompt)
    return response.content

# Function to handle "Talk to Me" conversation loop
def talk_to_me(pinecone_response, groq_api_key):
    if 'conversation_count' not in st.session_state:
        st.session_state.conversation_count = 0  # Initialize conversation counter

    llm = ChatGroq(temperature=0.7, groq_api_key=groq_api_key, model_name="gemma2-9b-it")

    # Display initial prompt or last response
    if st.session_state.conversation_count == 0:
        st.write(f"Let's chat! You can ask me questions about: {pinecone_response}")
    else:
        st.write(f"Bot: {st.session_state.last_response}")
    
    # Create a dynamic input field with a unique key based on conversation count
    user_query = st.text_input(
        "Your question or query:", 
        key=f"talk_input_{st.session_state.conversation_count}",  
        placeholder="Ask me anything or type 'thankyou' to end the conversation."
    )
    
    if user_query:
        if user_query.lower() == "thankyou":
            st.write("You're welcome! Chat ended.")
        else:
            # Generate a response from the LLM based on the user's query
            prompt = (
                "Instruction: 'If you are a licensed psychologist, provide a concise, helpful response to the user's concern.'\n"
                f"The user asked: '{user_query}'. Please give a concise response."
            )
            response = llm.invoke(prompt).content

            # Display the response and save it to session state
            st.session_state.last_response = response
            st.session_state.conversation_count += 1  
            st.experimental_rerun()