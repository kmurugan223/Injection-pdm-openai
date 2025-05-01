from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import logging
from openai import OpenAI

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

# Get OpenAI API key and add debugging
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY environment variable is not set")
    print("ERROR: OPENAI_API_KEY environment variable is not set")
elif len(api_key) < 20:
    logger.error(f"API key is suspiciously short: {len(api_key)} characters")
    print(f"ERROR: API key is suspiciously short: {len(api_key)} characters")
else:
    logger.info(f"API key found with length {len(api_key)}")
    print(f"INFO: API key found with length {len(api_key)}, starting with: {api_key[:5]}...")

# Initialize OpenAI client with explicit api_key parameter
try:
    client = OpenAI(api_key=api_key)
    logger.info("OpenAI client initialized successfully")
    print("SUCCESS: OpenAI client initialized successfully")
except Exception as e:
    error_msg = f"Failed to initialize OpenAI client: {str(e)}"
    logger.error(error_msg)
    print(f"ERROR: {error_msg}")
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
        logger.error("OpenAI client is not available when endpoint was called")
        raise HTTPException(
            status_code=503, 
            detail="OpenAI client is not available. Please check if the API key is properly set."
        )
    return client

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the OpenAI API with FastAPI", "openai_client_available": client is not None}

# Health check endpoint (useful for Vercel)
@app.get("/health")
def health_check():
    # Provide more detailed health information
    return {
        "status": "ok",
        "openai_client_available": client is not None,
        "api_key_configured": api_key is not None
    }

# Environment info endpoint (for debugging)
@app.get("/debug")
def debug_info():
    env_vars = {k: v[:3] + "..." if k.lower().endswith("key") and v else v for k, v in os.environ.items() if k.startswith("OPENAI")}
    return {
        "environment_variables": env_vars,
        "openai_client_available": client is not None,
        "api_key_length": len(api_key) if api_key else 0
    }

# Text completion endpoint
@app.post("/completion")
async def get_completion(request: CompletionRequest, openai_client: OpenAI = Depends(get_openai_client)):
    try:
        logger.info(f"Processing completion request with prompt: {request.prompt[:20]}...")
        response = openai_client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return {"result": response.choices[0].text}
    except Exception as e:
        error_msg = f"Error in completion endpoint: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# Chat completion endpoint
@app.post("/chat")
async def get_chat_completion(request: ChatRequest, openai_client: OpenAI = Depends(get_openai_client)):
    try:
        logger.info(f"Processing chat request with {len(request.messages)} messages")
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return {"result": response.choices[0].message.content}
    except Exception as e:
        error_msg = f"Error in chat endpoint: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = f"Unhandled exception: {str(exc)}"
    logger.error(error_msg)
    return JSONResponse(
        status_code=500,
        content={"message": error_msg},
    )