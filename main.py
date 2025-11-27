import sys
import asyncio
import platform

# --- WINDOWS PLAYWRIGHT FIX ---
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel
import uvicorn
import time

import config
from config import logger, MY_SECRET
from agent import solve_quiz_task

app = FastAPI()

class QuizTask(BaseModel):
    email: str
    secret: str
    url: str

@app.post("/solve")
async def start_quiz_solver(task: QuizTask, background_tasks: BackgroundTasks, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Received quiz task from {client_ip} for URL: {task.url}")

    if task.secret != MY_SECRET:
        logger.warning(f"Invalid secret from {client_ip}. Access denied.")
        raise HTTPException(status_code=403, detail="Invalid secret")

    # --- FIX: Enforce 3-minute limit (170s to provide buffer) ---
    deadline = time.time() + 170
    
    logger.info("Secret verified. Accepting task and starting agent in background.")
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
    return {"message": "LLM Analysis Quiz Agent is running."}

if __name__ == "__main__":
    logger.info("Starting FastAPI server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)