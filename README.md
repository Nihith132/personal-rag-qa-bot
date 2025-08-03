# Personal RAG Q&A Bot

A simple command-line RAG (Retrieval-Augmented Generation) application that answers questions based on a local text file. This project leverages local sentence-transformer embeddings for privacy and the high-speed Groq API for language model inference.

---

## Features

-   **Bring Your Own Data:** Answers questions using a custom knowledge base (`my_notes.txt`).
-   **Local & Private Embeddings:** Uses a `sentence-transformers` model to create text embeddings locally, so your data is never sent to a third-party embedding service.
-   **High-Speed LLM:** Powered by Meta's Llama 3 model running on the blazingly fast Groq API.
-   **Built with LangChain:** Orchestrates the entire RAG pipeline using the LangChain framework.

---

## Tech Stack

-   **Language:** Python
-   **Core Framework:** LangChain
-   **LLM Provider:** Groq (Llama 3)
-   **Embeddings:** Sentence-Transformers (Hugging Face)
-   **Vector Store:** FAISS (Facebook AI Similarity Search)

---

## Setup and Installation

Follow these steps to run the project locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
