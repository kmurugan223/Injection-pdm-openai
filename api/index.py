from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import os
from openai import OpenAI
from fastapi.routing import APIRouter

# Initialize FastAPI app
app = FastAPI(title="OpenAI API with FastAPI")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Pydantic models for request validation
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = Field(default=150)
    temperature: Optional[float] = Field(default=0.7)
    
class ChatRequest(BaseModel):
    messages: List[dict]
    max_tokens: Optional[int] = Field(default=150)
    temperature: Optional[float] = Field(default=0.7)

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the OpenAI API with FastAPI"}

# Health check endpoint (useful for Vercel)
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Text completion endpoint
@app.post("/completion")
async def get_completion(request: CompletionRequest):
    try:
        response = client.completions.create(
            model="gpt-4o-mini",
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return {"result": response.choices[0].text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chat completion endpoint
@app.post("/chat")
async def get_chat_completion(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return {"result": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": f"An unexpected error occurred: {str(exc)}"},
    )