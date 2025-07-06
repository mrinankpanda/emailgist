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
import re

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
    allow_origins=["http://localhost:5173", "https://emailgist-lime.vercel.app"],
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


def email_preprocessing(text: str) -> str:
    """Email cleanup for better summarization"""

    # This removes any of the headers ex. From:, To:, etc.
    text = re.sub(r'^(From|To|Subject|Date|Sent|Cc|Bcc):\s*.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # This removes any forwarding indicators
    text = re.sub(r'^(FW:|RE:|FWD:)', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # This removes the email signatures
    text = re.sub(r'\n\s*--\s*\n.*$', '', text, flags=re.DOTALL)
    text = re.sub(r'\n\s*Best regards?.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\n\s*Sincerely.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\n\s*Thanks?.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # This removes any excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    # This removes URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    return text.strip()

def extract_highlights(text: str) -> List[str]:
    """Extract the key highlights from the email"""
    highlights = []

    # Filtering to find action words and phrases that indicate important content
    action_patterns = [
        r'\b(?:deadline|due|urgent|asap|immediately|schedule|meeting|call|review|approve|complete|finish|submit|send|deliver)\b',
        r'\b(?:action item|next step|follow up|please|need to|must|should|required|important)\b',
        r'\b(?:meeting|conference|call|appointment|interview)\s+(?:on|at|scheduled|planned)\b'
    ]

    # Do the search for action-oriented phrases
    for pattern in action_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            highlights.append(f"{match.title()}")
    
    # Filtering to find any date patterns
    date_patterns = [
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s*\d{4}\b',
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\b(?:this|next)\s+(?:week|month|quarter|year)\b'
    ]

    for pattern in date_patterns:
        dates = re.findall(pattern, text, re.IGNORECASE)
        for date in dates:
            highlights.append(date)

    # Filtering and search for times
    time_pattern = r'\b(?:1[0-2]|[1-9]):[0-5][0-9]\s*(?:AM|PM|am|pm)\b'
    times = re.findall(time_pattern, text)
    for time in times:
        highlights.append(time)

    # Filtering and search for money
    money_patterns = [
        r'\$[\d,]+(?:\.\d{2})?',
        r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|€|euros?|£|pounds?)\b',
        r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%\b'  # Percentages
    ]

    for pattern in money_patterns:
        amounts = re.findall(pattern, text, re.IGNORECASE)
        for amount in amounts:
            highlights.append(amount)

    # Filtering and searching for emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    for email in emails:
        highlights.append(email)

    phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'     
    phones = re.findall(phone_pattern, text)
    for phone in phones:
        highlights.append(phone)   

    # Filtering and searching for names
    name_pattern = name_pattern = r'\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b'
    names = re.findall(name_pattern, text)
    # Filtering out some false positives that come up when searching for names
    common_false_positives = {'Best Regards', 'Thank You', 'Please Let', 'Kind Regards', 'Sincerely'}
    names = [name for name in names if name not in common_false_positives]
    for name in names:
        highlights.append(name)
    
    # Filtering and searching for companies
    company_pattern = r'\b[A-Z][a-z]+\s+(?:Inc|LLC|Corp|Corporation|Company|Group|Team|Department)\b'
    companies = re.findall(company_pattern, text)
    for company in companies:
        highlights.append(company)

    seen = set()
    unique_highlights = []
    for highlight in highlights:
        if highlight not in seen:
            seen.add(highlight)
            unique_highlights.append(highlight)

    unique_highlights = unique_highlights [:8]

    return unique_highlights

async def generate_summary(text: str) -> str:
    """Generate summary using the AI model"""
    try: 
        loop = asyncio.get_event_loop()

        def run_summary_process():
            input_length = len(text.split())
            max_length = min(150, max(50, input_length // 4))
            min_length = max(30, max_length // 3)

            result = summarizer(
                text,
                max_length = max_length,
                min_length = min_length,
                do_sample = True,
                temperature = 0.7,
                num_beams = 4,
                early_stopping = True,
                no_repeat_ngram_size = 3,
                length_penalty = 1.0
            )
            return result[0]['summary_text']
        
        summary = await loop.run_in_executor(None, run_summary_process)
        summary = summary.strip()

        if summary.lower().startswith(('the email', 'this email', 'the message')):
            sentences = summary.split('.')
            if len(sentences) > 1:
                summary = '.'.join(sentences[1:]).strip()
        
        return f"{summary}"
    
    except Exception as e:
        logger.error(f"There was an error with the summarization process: {e}")
        return "There was an error with the summarization process."

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
        text = email_preprocessing(text)
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