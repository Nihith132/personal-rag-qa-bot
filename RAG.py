import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

def main():
    print("--- Personal RAG Q&A Bot (Local Embeddings + Groq LLM) ---")

    # 1. Load the document
    with open("my_notes.txt", "r") as f:
        document_text = f.read()
    # 2. Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(document_text)
    print(f"Split the document into {len(texts)} chunks.")

    # 3. Create embeddings and the vector store using a local model
    print("Loading local embedding model... (This may take a moment on first run)")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = FAISS.from_texts(texts, embeddings)
    print("Created vector store.")

    # 4. Set up the RAG chain with Groq and Llama 3
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    print("\nRAG chain is ready. Ask your questions! (Type 'exit' to quit)")

    # 5. Ask questions in a loop
    while True:
        query = input("\n enter the input: ")
        if query.lower() in ["exit", "quit"]:
            break
        try:
            response = qa_chain.invoke(query)
            print("\nAssistant:", response['result'])
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
