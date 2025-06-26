# Import necessary FastAPI and supporting libraries
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import time
import uvicorn

# Initialize the FastAPI app with metadata
app = FastAPI(
    title="emailgist API",
    description="AI-powered email summarization and highlighting service",
    version="1.0.0",
)

# Enable CORS to allow requests from frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://emailgist.com"],
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
    if not request.email_content or len(request.email_content.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail = "The content of your email must be at least 10 characters long."
        )
    
    # Dummy response - simulates the AI processing time
    await simulate_processing_delay()

    # Generate dummy summary based on the length of the email for now
    email_length = len(request.email_content)
    if (email_length > 1000):
        summary = ("📧 Long email detected! Key points: Multiple topics discussed, "
            "several action items identified, important deadlines mentioned. "
            "This appears to be a comprehensive business communication requiring immediate attention.")
    elif email_length > 500:
        summary = (
            "📧 Medium-length email summarized: Main topic covered with relevant details, "
            "some action items present, moderate priority level indicated."
        )
    else:
        summary = (
            "📧 Brief email summary: Concise communication with clear purpose and "
            "minimal action items required."
        )
    
    # Dummy highlights (normally extracted via NLP model)
    dummy_highlights = [
        "John Smith",
        "Project Alpha", 
        "December 15th",
        "Budget approval",
        "Client meeting"
    ]

    # Calculate the processing time
    processing_time = time.time() - start_time

    #Return summary and highlights in structured response
    return SummaryResponse (
        summary = summary,
        highlights = dummy_highlights,
        processing_time = round (processing_time, 2)
    )

# Simulate processing time to mimic the actual AI model delay
async def simulate_processing_delay():
    """Simulate AI Processing Time"""
    import asyncio
    await asyncio(0.5) # Simulate 500ms delay


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