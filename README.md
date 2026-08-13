# 📄 AI Document Q&A System using RAG

An AI-powered Document Question Answering system that allows users to upload a PDF and ask questions about its content. The system uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from the document and generate answers using an AI language model.

## 🚀 Features

* 📤 Upload PDF documents
* 📖 Extract text from PDF
* ✂️ Split documents into meaningful chunks
* 🧠 Generate text embeddings
* 🔎 Semantic similarity search using FAISS
* 🤖 Generate answers using FLAN-T5
* 💬 Interactive question-answering interface
* 📚 Display retrieved document sources
* 🌐 Streamlit web application

## 🛠️ Tech Stack

* Python
* Google Colab
* PyPDF
* Sentence Transformers
* FAISS
* Hugging Face Transformers
* FLAN-T5
* LangChain Text Splitters
* Streamlit
* NumPy

## 🧠 Architecture

```text
PDF Document
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Text Embeddings
     ↓
FAISS Vector Database
     ↓
User Question
     ↓
Question Embedding
     ↓
Similarity Search
     ↓
Relevant Document Chunks
     ↓
RAG Prompt
     ↓
FLAN-T5 LLM
     ↓
AI Generated Answer
```

## ⚙️ How It Works

1. User uploads a PDF document.
2. The system extracts text from the PDF.
3. The extracted text is divided into smaller chunks.
4. Each chunk is converted into a vector embedding.
5. Embeddings are stored in a FAISS vector index.
6. User asks a question about the document.
7. The question is converted into an embedding.
8. FAISS retrieves the most relevant document chunks.
9. Retrieved information is provided as context to the language model.
10. The model generates the final answer.

## 💻 Run Locally

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AI-Document-QA-RAG.git
cd AI-Document-QA-RAG
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 📁 Project Structure

```text
AI-Document-QA-RAG/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── AI_Document_QA_RAG.ipynb
```

## 🎯 Use Cases

* Research paper Q&A
* Company document analysis
* Study material assistant
* Technical documentation search
* Resume/document analysis
* Business report Q&A

## 🔮 Future Improvements

* Support multiple PDF documents
* Add conversation memory
* Add page-number citations
* Improve answer accuracy
* Add multilingual support
* Use advanced LLMs
* Add document summarization
* Deploy with Streamlit Community Cloud

## 👨‍💻 Project Type

**Generative AI | RAG | NLP | Document Intelligence**

## 📌 Skills Demonstrated

**Python • NLP • Generative AI • RAG • Embeddings • Vector Database • FAISS • Hugging Face • LangChain • Streamlit**

## ⭐ Project Goal

The goal of this project is to demonstrate a practical **Retrieval-Augmented Generation (RAG) pipeline** that can understand and answer questions from user-provided PDF documents.

