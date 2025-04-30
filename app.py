# app.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI
import pinecone
from typing import List, Dict, Any
import uuid

app = FastAPI()

# Add CORS middleware to allow requests from Node-RED
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your Node-RED domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize Pinecone for vector storage
pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT")
)
index_name = "documents-index"
index = pinecone.Index(index_name)

class QueryRequest(BaseModel):
    query: str
    max_results: int = 3

class RAGResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []
    request_id: str

@app.post("/api/query", response_model=RAGResponse)
async def process_query(request: QueryRequest):
    try:
        # 1. Convert query to embedding
        query_embedding = client.embeddings.create(
            model="text-embedding-ada-002",
            input=request.query
        ).data[0].embedding
        
        # 2. Search in Pinecone
        search_results = index.query(
            vector=query_embedding,
            top_k=request.max_results,
            include_metadata=True
        )
        
        # 3. Format context from search results
        contexts = []
        sources = []
        for match in search_results.matches:
            if match.score < 0.7:  # Filter out irrelevant matches
                continue
            contexts.append(match.metadata["text"])
            sources.append({
                "title": match.metadata.get("title", "Unknown"),
                "url": match.metadata.get("url", ""),
                "relevance": match.score
            })
        
        context_text = "\n\n---\n\n".join(contexts)
        
        # 4. Generate response with OpenAI
        prompt = f"""
        Answer the following question based on the provided context. If the context doesn't contain 
        relevant information, say "I don't have enough information to answer this question."
        
        Context:
        {context_text}
        
        Question: {request.query}
        """
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        
        request_id = str(uuid.uuid4())
        
        return RAGResponse(
            response=response.choices[0].message.content,
            sources=sources,
            request_id=request_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)