import sys
import asyncio
import platform
import traceback

# --- WINDOWS PLAYWRIGHT FIX ---
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import time

import config
from config import logger, MY_SECRET
from agent import solve_quiz_task

app = FastAPI(title="Quiz Solver Agent")

class QuizTask(BaseModel):
    email: str
    secret: str
    url: str

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global error handler for unhandled exceptions"""
    logger.error(f"Unhandled error: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

@app.post("/solve")
async def start_quiz_solver(task: QuizTask, background_tasks: BackgroundTasks, request: Request):
    """Main endpoint to receive quiz tasks from the evaluation server."""
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Received quiz task from {client_ip} for URL: {task.url}")

    # Validate secret
    if task.secret != MY_SECRET:
        logger.warning(f"Invalid secret from {client_ip}. Access denied.")
        raise HTTPException(status_code=403, detail="Invalid secret")

    # Set deadline (170s to have buffer before 3-minute limit)
    deadline = time.time() + 170
    
    logger.info("Secret verified. Starting agent in background.")
    background_tasks.add_task(
        solve_quiz_task,
        email=task.email,
        secret=task.secret,
        url=task.url,
        deadline=deadline
    )
    
    return {"status": "Task accepted. Processing in background."}

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "LLM Analysis Quiz Agent is ready.",
        "version": "2.0"
    }

@app.get("/health")
def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "api_key_configured": bool(config.GOOGLE_API_KEY),
        "secret_configured": bool(MY_SECRET)
    }

if __name__ == "__main__":
    logger.info("Starting FastAPI server on http://0.0.0.0:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860)