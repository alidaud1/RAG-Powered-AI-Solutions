import re
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Pinecone
import google.generativeai as genai

## Preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

## Read and Preprocess Documents
def read_preprocess_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    for doc in documents:
        doc.page_content = preprocess_text(doc.page_content)
    return documents

## Load Data from Directory
def data_load(path):
    documents = read_preprocess_doc(path)
    print(f"Total pages:  {len(documents)}")
    return documents

## Chunk the Data
def chunk_data(docs, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc_chunks = text_splitter.split_documents(docs)
    return doc_chunks

def chunking(path):
    docs = data_load(path)
    documents = chunk_data(docs)
    print(f"Total no. of chunks: {len(documents)}")
    return documents

## Generate Embeddings for Chunks
def chunk_embeddings(documents, embedding_model):
    text_chunks = [doc.page_content for doc in documents]
    embeddings = embedding_model.encode(text_chunks)
    print(f"Generated {embeddings.shape[0]} embeddings")
    return embeddings, text_chunks

## Connect to Pinecone
def db_connect(pinecone_api_key, index_name):
    from pinecone import Pinecone

    pc = Pinecone(api_key="3047202b-9952-452a-bf68-e9efda959c77")
    index = pc.Index("pdfchatbot")
    return index

## Upsert Data to Pinecone
def data_upsert(embeddings, index):
    upsert_data = [(str(i), embedding) for i, embedding in enumerate(embeddings)]
    index.upsert(vectors=upsert_data)
    print("Data upserted successfully")

## Query text preprocessing
def query_preprocess(query_text,model):
  text = query_text
  text = query_text.lower()
  text = re.sub(r'[^a-zA-Z0-9]', ' ', query_text)
  text = ' '.join(query_text.split())

  query_embedding = model.encode(query_text).tolist()
  return query_embedding

## Retrieve Data from Pinecone
def retrieve_data(index, query_embedding, num_results):
    result = index.query(vector=query_embedding, top_k=num_results, include_values=True)
    return result

## Store Retrieved Chunks
def store_retrieved_data(result, text_chunks):
    retrieved_chunks = ""
    for match in result['matches']:
        doc_id = int(match['id'])
        retrieved_chunks += text_chunks[doc_id] + " "
    return retrieved_chunks

def generate_response(query_text, retrieved_chunks, gemini_key):
    genai.configure(api_key=gemini_key)

    generation_config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "max_output_tokens": 500,
        "response_mime_type": "text/plain",
    }

    chat_session = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config
    ).start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    f"Based on the following document: '{retrieved_chunks}', answer this question: '{query_text}'"
                ],
            },
        ]
    )

    # Send the message with the query
    response = chat_session.send_message(f"{query_text}")

    return response
