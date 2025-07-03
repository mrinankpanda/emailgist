# Import necessary FastAPI and supporting libraries
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import time
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

summarizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global summarizer
    models_to_try = [
        "sshleifer/distilbart-cnn-12-6",  
        "sshleifer/distilbart-cnn-6-6",  
        "t5-small",                      
        "facebook/bart-large-cnn"
    ]

    for models in models_to_try:
        try:
            logger.info(f"Trying to load model: {models}")
            summarizer = pipeline (
                "summarization",
                model = models,
                device = -1
            )
            logger.info(f"Model loaded successfully: {models}")
            break
        except Exception as e:
            logger.warning(f"Failed to load {models}: {e}")
            continue

    if summarizer is None:
        logger.error("All models failed to load!")
    else:
        logger.info("AI Summarization is ready!")
    
    yield
    
    logger.info("Shutting down...")

# Initialize the FastAPI app with metadata
app = FastAPI(
    title="emailgist API",
    description="AI-powered email summarization and highlighting service",
    version="1.0.0",
    lifespan = lifespan
)

# Enable CORS to allow requests from frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://emailgist-lime.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body schema for email summarization
class EmailRequest(BaseModel):
    email_content: str # Raw email text which will be used to generate the summary

# Define response schema for summarized output
class SummaryResponse(BaseModel):
    summary: str                # The generated summary
    highlights: List[str]       # Key extracted terms or highlights
    processing_time: float      # Time taken to process the request

# Root endpoint: Simple health check / welcome message
@app.get("/")
async def read_root():
    """Health check endpoint"""
    return {"message": "This emailgist API is up and running.", "status": "healthy"}

# Main API endpoint for summarizing emails
@app.post("/summarize", response_model=SummaryResponse)
async def summarize_email(request: EmailRequest):
    """
    Summarize email content and extract the highlights.
    
    For now, this wil return dummy data to test the full pipleine.
    Will be replaced with actual AI summarization later.
    """
    start_time = time.time()
    # Validate that email content is not too short
    if not request.email_content or len(request.email_content.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail = "The content of your email must be at least 50 characters long."
        )
    
    if summarizer is None:
        raise HTTPException(
            status_code = 503,
            detail = "AI Model is not available. Please try again later."
        )
    
    try:
        text = request.email_content.strip()
        max_length = 4000
        if len(text) > max_length:
            text = text[:max_length]

        summary_result = await generate_summary(text)

        highlights = extract_highlights(request.email_content)

        processing_time = time.time() - start_time

        #Return summary and highlights in structured response
        return SummaryResponse (
            summary = summary_result,
            highlights = highlights,
            processing_time = round (processing_time, 2)
        )
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        raise HTTPException (
            status_code = 500,
            detail = f"Error processing email: {str(e)}"
        )

async def generate_summary(text: str) -> str:
    """Generate summary using the AI model"""
    try: 
        loop = asyncio.get_event_loop()

        def run_summarization():
            result = summarizer(
                text,
                max_length = 250,
                min_length = 30,
                do_sample = False
            )
            return result[0]['summary_text']
        summary = await loop.run_in_executor(None, run_summarization)
        return f"📧 {summary}"
    except Exception as e:
        logger.error(f"There was an error with summarization: {e}")
        return "There was an error with summarization."


def extract_highlights(text: str) -> List[str]:
    """Extract the key highlights from the email"""
    import re

    highlights = []

    date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s*\d{4}|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    dates = re.findall(date_pattern, text, re.IGNORECASE)
    highlights.extend(dates[:3]) # Limit to 3 dates

    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    highlights.extend(emails[:1])

    name_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
    names = re.findall(name_pattern, text)
    highlights.extend(names[:3])  # Limit to 3 names
    
    money_pattern = r'\$[\d,]+(?:\.\d{2})?'
    amounts = re.findall(money_pattern, text)
    highlights.extend(amounts[:2])  # Limit to 2 amounts
    
    highlights = list(dict.fromkeys(highlights))[:8]
    
    if not highlights:
        highlights = ["Email processed", "No specific highlights detected"]
    
    return highlights

@app.get("/health")
async def health_check():
    """Detailed health check for monitoring"""
    return {
        "status": "healthy",
        "service": "emailgist API",
        "version": "1.0.0",
        "timestamp": time.time()
    }

# Entry point for running the app directly (useful for local dev)
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host = "0.0.0.0",
        port = 8000,
        reload = True,
        log_level="info"
    )