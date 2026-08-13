
import streamlit as st
import os
import faiss
import numpy as np
import torch

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Document Q&A",
    page_icon="📄",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📄 AI Document Q&A System")
st.write(
    "Upload a PDF and ask questions using a RAG-based AI system."
)


# --------------------------------------------------
# Load Embedding Model
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# --------------------------------------------------
# Load LLM
# --------------------------------------------------

@st.cache_resource
def load_llm():

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    return tokenizer, model, device


tokenizer, model, device = load_llm()


# --------------------------------------------------
# PDF Text Extraction
# --------------------------------------------------

def extract_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# --------------------------------------------------
# Text Chunking
# --------------------------------------------------

def create_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_text(text)


# --------------------------------------------------
# Create FAISS Index
# --------------------------------------------------

def create_faiss_index(chunks):

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


# --------------------------------------------------
# Generate Answer
# --------------------------------------------------

def generate_answer(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=150
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer


# --------------------------------------------------
# Retrieve + Generate
# --------------------------------------------------

def ask_document(
    question,
    chunks,
    index,
    k=3
):

    query_embedding = embedding_model.encode(
        [question]
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k
    )

    retrieved_chunks = [
        chunks[i]
        for i in indices[0]
    ]

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
Answer the question using ONLY the context below.

If the answer is not available in the context,
say:

"I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    answer = generate_answer(prompt)

    return answer, retrieved_chunks


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "chunks" not in st.session_state:

    st.session_state.chunks = None

if "index" not in st.session_state:

    st.session_state.index = None

if "messages" not in st.session_state:

    st.session_state.messages = []


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📁 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    process_button = st.button(
        "🚀 Process Document"
    )


# --------------------------------------------------
# Process PDF
# --------------------------------------------------

if process_button:

    if uploaded_file is None:

        st.warning(
            "Please upload a PDF first."
        )

    else:

        with st.spinner(
            "Processing document..."
        ):

            text = extract_text(
                uploaded_file
            )

            if not text.strip():

                st.error(
                    "No readable text found in this PDF."
                )

            else:

                chunks = create_chunks(
                    text
                )

                index = create_faiss_index(
                    chunks
                )

                st.session_state.chunks = chunks

                st.session_state.index = index

                st.success(
                    f"Document processed successfully! "
                    f"{len(chunks)} chunks created."
                )


# --------------------------------------------------
# Chat Interface
# --------------------------------------------------

st.subheader("💬 Ask Questions")

if (
    st.session_state.chunks is not None
    and st.session_state.index is not None
):

    question = st.chat_input(
        "Ask something about your PDF..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.spinner(
            "Thinking..."
        ):

            answer, sources = ask_document(
                question,
                st.session_state.chunks,
                st.session_state.index
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources
            }
        )


    # Display chat history

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

            if (
                message["role"] == "assistant"
                and "sources" in message
            ):

                with st.expander(
                    "📚 Retrieved Sources"
                ):

                    for i, source in enumerate(
                        message["sources"],
                        1
                    ):

                        st.markdown(
                            f"**Source {i}**"
                        )

                        st.write(
                            source
                        )

else:

    st.info(
        "👈 Upload a PDF from the sidebar "
        "and click 'Process Document'."
    )
