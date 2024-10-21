import os
import streamlit as st
import warnings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from utils import *

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Streamlit app
st.title("RAG-based PDF Q&A Chatbot")

# Sidebar inputs for user
st.sidebar.header("Configure Settings")
pdf_path = st.sidebar.text_input("PDF Directory", "documents/")
index_name = st.sidebar.text_input("Pinecone Index Name", "chatwithpdf")
num_values = st.sidebar.slider("Number of Chunks to Retrieve", 1, 5, 3)
query_text = st.text_input("Enter your question")

# Load API keys
pinecone_api_key = os.getenv('PINECONE_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')

# Load the embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Cache the chunking and embedding to avoid recomputing each time
@st.cache_data
def load_and_embed_documents(pdf_path):
    documents = chunking(pdf_path)
    embeddings, text_chunks = chunk_embeddings(documents, embedding_model)
    return embeddings, text_chunks

# Cache the Pinecone connection and upsert data only once
@st.cache_resource
def setup_pinecone_and_upsert(embeddings):
    index = db_connect(pinecone_api_key, index_name)
    data_upsert(embeddings, index)
    return index

# Main functionality
if st.button("Get Answer"):
    try:
        # For the first run, load and embed documents, then upsert
        embeddings, text_chunks = load_and_embed_documents(pdf_path)
        index = setup_pinecone_and_upsert(embeddings)

        # Process the query and get embedding
        query_embedding = query_preprocess(query_text, embedding_model)

        # Retrieve data from Pinecone and get relevant chunks
        result_data = retrieve_data(index, query_embedding, num_values)
        retrieved_chunks = store_retrieved_data(result_data, text_chunks)

        # Check if relevant chunks were found
        if not retrieved_chunks or len(retrieved_chunks) == 0:
            st.warning("The provided document doesn't contain any information about this.")
        else:
            # Generate response from Gemini API
            response = generate_response(query_text, retrieved_chunks, gemini_api_key)

            # Check if the response is relevant
            if not response.text.strip():
                st.warning("The provided document doesn't contain any information about this.")
            else:
                # Display the generated response
                st.write("Generated Response:")
                st.success(response.text)

    except Exception as e:
        st.error(f"Error: {e}")

# Add footer text with small font size
st.markdown(
    """
    <style>
    .small-font {
        font-size:10px;
        color: #999999;
    }
    </style>
    <p class="small-font">Developed by Ali Daud</p>
    """, unsafe_allow_html=True
)
