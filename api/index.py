from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI
import pinecone
from typing import List, Dict, Any
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Debug environment variables (don't log full API keys in production)
logger.info(f"OPENAI_API_KEY present: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No'}")
logger.info(f"PINECONE_API_KEY present: {'Yes' if os.environ.get('PINECONE_API_KEY') else 'No'}")
logger.info(f"PINECONE_ENVIRONMENT: {os.environ.get('PINECONE_ENVIRONMENT')}")

# Test endpoint
@app.get("/api/test")
async def test_endpoint():
    return {"status": "ok", "message": "API is running"}

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"OpenAI initialization failed: {str(e)}")
    client = None

# Initialize Pinecone
pinecone_initialized = False
index = None
try:
    if os.environ.get("PINECONE_API_KEY") and os.environ.get("PINECONE_ENVIRONMENT"):
        pinecone.init(
            api_key=os.environ.get("PINECONE_API_KEY"),
            environment=os.environ.get("PINECONE_ENVIRONMENT")
        )
        index_name = "documents-index"
        # Check if index exists
        indexes = pinecone.list_indexes()
        logger.info(f"Available Pinecone indexes: {indexes}")
        if index_name in indexes:
            index = pinecone.Index(index_name)
            pinecone_initialized = True
            logger.info(f"Pinecone index '{index_name}' initialized successfully")
        else:
            logger.error(f"Pinecone index '{index_name}' not found")
    else:
        logger.error("Pinecone API key or environment not set")
except Exception as e:
    logger.error(f"Pinecone initialization error: {str(e)}")

class QueryRequest(BaseModel):
    query: str
    max_results: int = 3

class RAGResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []
    request_id: str

# Simple endpoint that doesn't use Pinecone or OpenAI
@app.post("/api/simple-query")
async def simple_query(request: QueryRequest):
    return RAGResponse(
        response=f"Received query: {request.query}",
        sources=[],
        request_id=str(uuid.uuid4())
    )

@app.post("/api/query")
async def process_query(request: QueryRequest):
    request_id = str(uuid.uuid4())
    logger.info(f"Request received: {request.query} with ID {request_id}")
    
    # Check if services are initialized
    if not client:
        logger.error("OpenAI client not initialized")
        return RAGResponse(
            response="Error: OpenAI service not available",
            sources=[],
            request_id=request_id
        )
    
    if not pinecone_initialized or not index:
        logger.error("Pinecone not initialized")
        return RAGResponse(
            response="Error: Pinecone service not available",
            sources=[],
            request_id=request_id
        )
    
    try:
        # Convert query to embedding
        logger.info("Creating embedding")
        embedding_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=request.query
        )
        query_embedding = embedding_response.data[0].embedding
        
        # Search in Pinecone
        logger.info("Querying Pinecone")
        search_results = index.query(
            vector=query_embedding,
            top_k=request.max_results,
            include_metadata=True
        )
        
        # Format context from search results
        contexts = []
        sources = []
        for match in search_results.matches:
            if match.score < 0.7:  # Filter out irrelevant matches
                continue
            contexts.append(match.metadata.get("text", ""))
            sources.append({
                "title": match.metadata.get("title", "Unknown"),
                "url": match.metadata.get("url", ""),
                "relevance": match.score
            })
        
        context_text = "\n\n---\n\n".join(contexts)
        
        # Generate response with OpenAI
        logger.info("Generating response with OpenAI")
        prompt = f"""
        Answer the following question based on the provided context. If the context doesn't contain 
        relevant information, say "I don't have enough information to answer this question."
        
        Context:
        {context_text}
        
        Question: {request.query}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return RAGResponse(
            response=response.choices[0].message.content,
            sources=sources,
            request_id=request_id
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return RAGResponse(
            response=f"Error processing your query: {str(e)}",
            sources=[],
            request_id=request_id
        )

# Root route for Vercel
@app.get("/")
def read_root():
    return {"message": "FastAPI RAG API is running"}