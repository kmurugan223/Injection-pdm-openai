from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "API is working"}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Test endpoint that doesn't use OpenAI
@app.post("/echo")
async def echo(data: dict):
    return {"echo": data}