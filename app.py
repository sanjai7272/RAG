import streamlit as st
import requests
import time
import os

# --- API Endpoints ---
API_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{API_URL}/upload-files"
QUERY_ENDPOINT = f"{API_URL}/rag-query"
DELETE_ENDPOINT = f"{API_URL}/delete-store"

# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="centered",
)

# --- Sidebar ---
with st.sidebar:
    st.header("About")
    st.info(
        """
        This application allows you to chat with your documents.
        
        **How to use:**
        1. Upload one or more PDF or TXT files.
        2. Wait for the files to be processed.
        3. Ask questions in the chat interface!
        """
    )
    
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload .pdf or .txt files", type=["pdf", "txt"], accept_multiple_files=True
    )
    
    if uploaded_files:
        with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
            files_to_upload = [
                ("files", (file.name, file, file.type)) for file in uploaded_files
            ]
            try:
                response = requests.post(UPLOAD_ENDPOINT, files=files_to_upload)
                if response.status_code == 200:
                    st.success("Files uploaded and processed successfully!")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the API: {e}")
                
    st.header("Manage Data")
    if st.button("Clear Vector Store"):
        with st.spinner("Clearing all processed documents..."):
            try:
                response = requests.post(DELETE_ENDPOINT)
                if response.status_code == 200:
                    st.success("Vector store has been cleared successfully!")
                    # Optional: Clear chat history as well
                    st.session_state.messages = [
                        {"role": "assistant", "content": "I've cleared my memory. Please upload new documents for me to read."}
                    ]
                    st.rerun()
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the API: {e}")


# --- Main Chat Interface ---
st.title("💬 RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload some documents and I'll help you with your questions about them."}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to simulate streaming
def stream_response(text, delay=0.02):
    for word in text.split():
        yield word + " "
        time.sleep(delay)

# React to user input
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get the last 5 messages for history
                history = st.session_state.messages[-6:-1]
                
                response = requests.post(
                    QUERY_ENDPOINT, 
                    json={"query": prompt, "history": history}
                )
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "Sorry, I couldn't find an answer.")
                    st.write_stream(stream_response(answer))
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_message = f"Error: {response.status_code} - {response.text}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
            except requests.exceptions.RequestException as e:
                error_message = f"Failed to connect to the API: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message}) 