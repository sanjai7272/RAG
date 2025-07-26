import os
import tempfile
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from main import load_documents, response, delete_vector_store
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Create fastapi app 
app = FastAPI(
    title="AI Course RAG API",
    description="API for AI Course RAG",
    version="1.0.0",
)

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FILE UPLOAD

# Creating a class for the response 
class FileUploadResponse(BaseModel):
    message: str
    
# Creating a route for the API 
@app.post("/upload-files", response_model=FileUploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    tmp_paths = []
    try:
        for file in files:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                tmp.write(await file.read())
                tmp_paths.append(tmp.name)
        
        # Load the documents
        if tmp_paths:
            load_documents(tmp_paths)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        # Clean up the temporary files
        for path in tmp_paths:
            if os.path.exists(path):
                os.remove(path)
    
    return {"message": f"{len(files)} file(s) uploaded and processed successfully"}

# RAG QUERY

# Creating a class for the query 
class RAGQuery(BaseModel):
    query: str
    history: List[dict] = []

# Creating a class for the response 
class RAGQueryResponse(BaseModel):
    answer: str
    
@app.post("/rag-query",response_model=RAGQueryResponse)
async def rag_query(query: RAGQuery):
    answer = response(query.query, query.history)
    return {"answer": answer}

@app.post("/delete-store", response_model=FileUploadResponse)
async def delete_store():
    try:
        delete_vector_store()
        return {"message": "Vector store cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear vector store: {e}")

# Run the API 
if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)
