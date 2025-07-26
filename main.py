import os
import shutil
from typing import List
from dotenv import load_dotenv, find_dotenv
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Langsmith

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "AI-COURSE-RAG"

# Load models once
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={"device": "cpu"})
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)


def load_documents(file_paths: List[str]):
    all_docs = []
    for file_path in file_paths:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            print(f"Skipping unsupported file type: {file_path}")
            continue
        all_docs.extend(loader.load())

    if not all_docs:
        return

    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)

    # Handle vector store
    if os.path.exists(DB_FAISS_PATH):
        try:
            vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_documents(chunks)
        except Exception:
            shutil.rmtree(DB_FAISS_PATH)
            vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(DB_FAISS_PATH)


def delete_vector_store():
    """Deletes the vector store directory if it exists."""
    if os.path.exists(DB_FAISS_PATH):
        shutil.rmtree(DB_FAISS_PATH)


def response(query: str, history: List[dict] = []):
    if not os.path.exists(DB_FAISS_PATH):
        return "The document store is empty. Please upload documents first."

    # Create a new prompt with history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question based on the context provided and the chat history. If you don't know the answer, just say that you don't know. Don't try to make up an answer.\n\nContext: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # Convert history to LangChain messages
    chat_history = []
    for msg in history:
        if msg.get("role") == "user":
            chat_history.append(HumanMessage(content=msg.get("content")))
        elif msg.get("role") == "assistant":
            chat_history.append(AIMessage(content=msg.get("content")))
            
    # Loading vector store
    try:
        vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return "The vector store seems to be corrupted. Please upload the documents again to fix it."

    # Creating retriever
    retriever = vectorstore.as_retriever()

    # Creating stuff doc chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Creating retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Get response
    result = retrieval_chain.invoke({
        "input": query,
        "chat_history": chat_history
    })

    return result["answer"]
    
    
    

    
    





