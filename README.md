
# Gemini + Chroma RAG Restaurant Agent

A smart restaurant recommendation chatbot powered by Google's Gemini model and Chroma vector search.

## Features
- Semantic search using sentence-transformers
- Embedded categories (family, friend, dating, etc.)
- ChromaDB vector storage
- Gemini-powered natural language responses
- Streamlit chatbot UI

## Setup

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Create `.env` file from `.env.example` and add your Gemini API key.

3. Run the chatbot:

```
PYTHONPATH=$(pwd) streamlit run app/ui.py
```