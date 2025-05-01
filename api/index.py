from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import openai
from pydantic import BaseModel
import os
from typing import List, Dict, Any
import uuid
import logging
from dotenv import load_dotenv
 
load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
app = FastAPI()
 
API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = API_KEY
 
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
 
# Test endpoint
@app.get("/api/test")
async def test_endpoint():
    return {"status": "ok", "message": "API is running"}
 
class QueryRequest(BaseModel):
    query: str
    max_results: int = 3  # Keeping for backward compatibility, but not used anymore
 
class Response(BaseModel):
    response: str
    request_id: str
 
# Simple endpoint for OpenAI only
@app.post("/api/simple-query")
async def simple_query(request: QueryRequest):
    return Response(
        response=f"Received query: {request.query}",
        request_id=str(uuid.uuid4())
    )
 
@app.post("/api/query")
async def process_query(request: QueryRequest):
    request_id = str(uuid.uuid4())
    logger.info(f"Request received: {request.query} with ID {request_id}")
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": request.query}
            ]
        )
        response_text = response.choices[0].message.content
        
        return Response(
            response=response_text,
            request_id=request_id
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return Response(
            response=f"Error processing your query: {str(e)}",
            request_id=request_id
        )
 
# Root route for Vercel
@app.get("/")
def read_root():
    return {"message": "FastAPI OpenAI API is running"}
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
 