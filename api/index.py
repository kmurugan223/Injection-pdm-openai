from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import os
from openai import OpenAI
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="OpenAI API with FastAPI")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check for OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.warning("OPENAI_API_KEY environment variable is not set")

# Initialize OpenAI client conditionally
try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    client = None

# Pydantic models for request validation
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = Field(default=150)
    temperature: Optional[float] = Field(default=0.7)
    
class ChatRequest(BaseModel):
    messages: List[dict]
    max_tokens: Optional[int] = Field(default=150)
    temperature: Optional[float] = Field(default=0.7)

# Dependency to check if OpenAI client is available
def get_openai_client():
    if client is None:
        raise HTTPException(
            status_code=503, 
            detail="OpenAI client is not available. Please check if the API key is properly set."
        )
    return client

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the OpenAI API with FastAPI"}

# Health check endpoint (useful for Vercel)
@app.get("/health")
def health_check():
    # Basic check that doesn't require OpenAI client
    return {"status": "ok"}

# Text completion endpoint
@app.post("/completion")
async def get_completion(request: CompletionRequest, openai_client: OpenAI = Depends(get_openai_client)):
    try:
        response = openai_client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return {"result": response.choices[0].text}
    except Exception as e:
        logger.error(f"Error in completion endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Chat completion endpoint
@app.post("/chat")
async def get_chat_completion(request: ChatRequest, openai_client: OpenAI = Depends(get_openai_client)):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return {"result": response.choices[0].message.content}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": f"An unexpected error occurred: {str(exc)}"},
    )