from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your main app
from app import app as app_instance

# For Vercel serverless functions
app = app_instance