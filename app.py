import streamlit as st
import numpy as np


from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Document Q&A",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📄 AI Document Q&A System")

st.write(
    "Upload a PDF and ask questions using "
    "Retrieval-Augmented Generation (RAG)."
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# ============================================================
# LOAD LLM
# ============================================================

@st.cache_resource
def load_llm():

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    return tokenizer, model, device


tokenizer, llm_model, device = load_llm()


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text()

        if page_text:

            text += (
                f"\n\n[Page {page_number}]\n"
                + page_text
            )

    return text


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embedding_matrix(chunks):

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    return embeddings


# ============================================================
# GENERATE AI ANSWER
# ============================================================

def generate_answer(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_chunks(
    question,
    chunks,
    embedding_matrix,
    k=3
):

    question_embedding = embedding_model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    similarities = cosine_similarity(
        question_embedding,
        embedding_matrix
    )[0]

    top_indices = np.argsort(
        similarities
    )[::-1][:k]

    retrieved_chunks = [
        chunks[index]
        for index in top_indices
    ]

    scores = [
        similarities[index]
        for index in top_indices
    ]

    return retrieved_chunks, scores


# ============================================================
# ASK DOCUMENT
# ============================================================

def ask_document(
    question,
    chunks,
    embedding_matrix,
    k=3
):

    retrieved_chunks, scores = retrieve_chunks(
        question,
        chunks,
        embedding_matrix,
        k
    )

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the question using ONLY the information
provided in the context.

If the answer cannot be found in the context,
say:

"I could not find the answer in the document."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

    answer = generate_answer(
        prompt
    )

    return answer, retrieved_chunks, scores


# ============================================================
# SESSION STATE
# ============================================================

if "chunks" not in st.session_state:

    st.session_state.chunks = None


if "embedding_matrix" not in st.session_state:

    st.session_state.embedding_matrix = None


if "messages" not in st.session_state:

    st.session_state.messages = []


if "document_name" not in st.session_state:

    st.session_state.document_name = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📁 Document Upload")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    process_button = st.button(
        "🚀 Process Document",
        use_container_width=True
    )

    st.divider()

    st.info(
        "Upload a PDF, process it, "
        "then ask questions about its content."
    )


# ============================================================
# PROCESS DOCUMENT
# ============================================================

if process_button:

    if uploaded_file is None:

        st.warning(
            "⚠️ Please upload a PDF first."
        )

    else:

        with st.spinner(
            "📖 Reading and processing document..."
        ):

            try:

                # Extract text
                text = extract_text(
                    uploaded_file
                )

                # Check text
                if not text.strip():

                    st.error(
                        "❌ No readable text found in this PDF."
                    )

                else:

                    # Create chunks
                    chunks = create_chunks(
                        text
                    )

                    # Create embeddings
                    embedding_matrix = (
                        create_embedding_matrix(
                            chunks
                        )
                    )

                    # Save to session
                    st.session_state.chunks = chunks

                    st.session_state.embedding_matrix = (
                        embedding_matrix
                    )

                    st.session_state.document_name = (
                        uploaded_file.name
                    )

                    # Clear previous chat
                    st.session_state.messages = []

                    st.success(
                        f"✅ Document processed successfully!"
                    )

                    st.info(
                        f"📄 File: {uploaded_file.name}"
                    )

                    st.info(
                        f"🧩 Chunks created: {len(chunks)}"
                    )

            except Exception as e:

                st.error(
                    f"❌ Error while processing PDF: {e}"
                )


# ============================================================
# DOCUMENT STATUS
# ============================================================

if (
    st.session_state.chunks is not None
    and
    st.session_state.embedding_matrix is not None
):

    st.success(
        f"📄 Active Document: "
        f"{st.session_state.document_name}"
    )

    st.write(
        f"🧩 Total chunks: "
        f"{len(st.session_state.chunks)}"
    )


# ============================================================
# CHAT SECTION
# ============================================================

st.subheader("💬 Ask Questions")

if (
    st.session_state.chunks is not None
    and
    st.session_state.embedding_matrix is not None
):

    # Display previous messages

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

            if (
                message["role"] == "assistant"
                and
                "sources" in message
            ):

                with st.expander(
                    "📚 View Retrieved Sources"
                ):

                    for i, source in enumerate(
                        message["sources"],
                        start=1
                    ):

                        st.markdown(
                            f"### Source {i}"
                        )

                        st.write(
                            source
                        )

                        if (
                            "scores" in message
                            and
                            i <= len(
                                message["scores"]
                            )
                        ):

                            st.caption(
                                f"Similarity Score: "
                                f"{message['scores'][i-1]:.4f}"
                            )


    # Chat input

    question = st.chat_input(
        "Ask something about your PDF..."
    )

    if question:

        # User message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.write(question)


        # AI response
        with st.chat_message("assistant"):

            with st.spinner(
                "🤖 Searching document and generating answer..."
            ):

                try:

                    answer, sources, scores = (
                        ask_document(
                            question,
                            st.session_state.chunks,
                            st.session_state.embedding_matrix,
                            k=3
                        )
                    )

                    st.write(answer)

                    with st.expander(
                        "📚 View Retrieved Sources"
                    ):

                        for i, source in enumerate(
                            sources,
                            start=1
                        ):

                            st.markdown(
                                f"### Source {i}"
                            )

                            st.write(
                                source
                            )

                            st.caption(
                                f"Similarity Score: "
                                f"{scores[i-1]:.4f}"
                            )

                    # Save AI response

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "scores": scores
                        }
                    )

                except Exception as e:

                    st.error(
                        f"❌ Error generating answer: {e}"
                    )

else:

    st.info(
        "👈 Upload a PDF from the sidebar "
        "and click **Process Document** to start."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with Python • RAG • NLP • "
    "Sentence Transformers • Hugging Face • Streamlit"
)
