import uvicorn
import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Wiki Proxy: Video Games")

# --- THE TASK ---
# Target: Sum of sales for the Top 3 best-selling games of all time.
# 1. Minecraft: ~300 million
# 2. Grand Theft Auto V: ~205 million
# 3. Tetris (EA): ~100 million
# Expected Total: ~605 million (Accepting a range due to live updates)

class Submission(BaseModel):
    email: str
    secret: str
    url: str
    answer: Optional[str | int | float] = None

@app.get("/wiki/games", response_class=HTMLResponse)
def get_wiki_page(request: Request):
    url = "https://en.wikipedia.org/wiki/List_of_best-selling_video_games"
    print(f"Fetching live data from: {url}...")
    
    # 1. User-Agent Header (Critical for Wikipedia)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers)
    
    # 2. Inject Task Instructions
    base_url = str(request.base_url).rstrip("/")
    submit_url = f"{base_url}/submit"
    
    instructions = f"""
    <div style="background:#e0f7fa; padding:20px; border:3px solid #006064; font-size:1.3em; margin-bottom:20px;">
        <strong>AGENT TASK:</strong><br>
        Locate the main table of "List of best-selling video games".<br>
        Calculate the <b>sum of sales (in millions)</b> for the <b>Top 3</b> games listed.<br>
        <br>
        <em>Hint:</em> You need to clean the data. If a cell says "300,000,000", treat it as 300 million. 
        If it says "205 million", treat it as 205.<br>
        Submit your answer (just the number) to: <code>{submit_url}</code>
    </div>
    <hr>
    """
    
    # Inject at the top of the body
    modified_html = response.text.replace('<body', '<body' + instructions, 1)
    
    return modified_html

@app.post("/submit")
def submit_answer(submission: Submission):
    print(f"Received submission: {submission.answer}")
    
    try:
        # Clean output (remove text like "million")
        raw_ans = str(submission.answer).lower().replace("million", "").replace(",", "").strip()
        val = float(raw_ans)
    except:
        return {"correct": False, "reason": f"Could not parse number from '{submission.answer}'"}

    # Target is roughly 605. We allow a range because Wiki edits happen daily.
    if 590 <= val <= 620:
        return {"correct": True, "message": f"Correct! The sum is approximately {val} million."}
    else:
        return {
            "correct": False, 
            "reason": f"Incorrect. You submitted {val}. Expected range: 590-620."
        }

if __name__ == "__main__":
    # Runs on Port 8005
    uvicorn.run(app, host="0.0.0.0", port=8005)