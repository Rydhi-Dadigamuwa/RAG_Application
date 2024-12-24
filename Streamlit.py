import streamlit as st
from streamlit_chat import message  # For default bot and human icons
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import numpy as np
import google.generativeai as ggai
import os


# Load environment variables
load_dotenv()

# Configure the Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Load from environment variables
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
ggai.configure(api_key=GEMINI_API_KEY)


#Load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Function to initialize FAISS index
def initialize_faiss():
    dimension = 384  # Embedding dimension (e.g., BERT)
    return faiss.IndexFlatL2(dimension), []


# Function to load and process PDF
def process_pdf(file, index, texts):
    pdf_reader = PdfReader(file)
    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text()

    # Split text into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(full_text)

    # Convert chunks to embeddings
    embeddings = embedding_model.embed_documents(chunks)
    embeddings_np = np.array(embeddings, dtype="float32")

    # Add embeddings to FAISS index
    index.add(embeddings_np)
    texts.extend(chunks)


# Function to query FAISS
def query_faiss(query, index, texts):
    query_vector = embedding_model.embed_query(query)
    query_vector_np = np.array(query_vector, dtype="float32").reshape(1, -1)
    distances, indices = index.search(query_vector_np, k=1)  # Top match

    results = []
    for idx in indices[0]:
        if idx < len(texts):
            results.append(texts[idx])
    return results


# Function to interact with Gemini
def get_answer_from_gemini(query, context):
    context_str = "\n\n".join(context)
    prompt = f"""
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer 
    the question. If you don't know the answer, say that you 
    don't know. Use three sentences maximum and keep the 
    answer concise.

    {context_str}
    Question: {query}
    """
    model = ggai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text


# Streamlit app
def main():
    st.set_page_config(page_title="ChatBot PDFs", page_icon="🤖")
    st.header("ChatBot PDFs 🤖")

    # Initialize session state variables
    if "index" not in st.session_state:
        st.session_state.index, st.session_state.texts = initialize_faiss()

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    user_input = st.text_input("Ask your question", key="user_input")

    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

        if st.button("Process PDFs"):
            if pdf_docs:
                for pdf in pdf_docs:
                    process_pdf(pdf, st.session_state.index, st.session_state.texts)
                st.sidebar.success("PDFs processed successfully!")
            else:
                st.sidebar.warning("Please upload some PDFs.")

    if user_input:
        if len(st.session_state.texts) > 0:
            context = query_faiss(user_input, st.session_state.index, st.session_state.texts)
            response = get_answer_from_gemini(user_input, context)
            st.session_state.conversation.append({"role": "user", "content": user_input})
            st.session_state.conversation.append({"role": "bot", "content": response})
        else:
            st.warning("Please process PDFs before asking questions.")

    if st.session_state.conversation:
        for i, message_dict in enumerate(st.session_state.conversation):
            if message_dict["role"] == "user":
                message(message_dict["content"], is_user=True, key=f"{i}_user")
            else:
                message(message_dict["content"], is_user=False, key=f"{i}_bot")


if __name__ == "__main__":
    main()
