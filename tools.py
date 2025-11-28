import os
import requests
import json
import base64
import io
import re
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pdfplumber
import subprocess
import sys
import google.generativeai as genai  # Added for Audio Transcription

from config import logger

# --- Configuration ---
TRUNCATE_LIMIT = 2000
NETWORK_TIMEOUT = 30

# --- Tool 1: Web Scraper (Playwright) ---
def scrape_website(url: str) -> str:
    """
    Fetches a URL, executes JavaScript via Playwright, and returns the full HTML content.
    Essential for getting dynamic content (like secret codes) that 'requests' misses.
    """
    logger.info(f"Tool: scrape_website - URL: {url}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=NETWORK_TIMEOUT * 1000) 
            
            # Wait a moment for dynamic content (like the 'Loading...' text) to settle
            try:
                page.wait_for_load_state('networkidle', timeout=2000)
            except:
                pass # Continue even if network isn't perfectly idle
            
            content = page.content()
            browser.close()
            
            soup = BeautifulSoup(content, 'lxml')
            return f"Full HTML content from {url}:\n{str(soup)}"
            
    except Exception as e:
        logger.error(f"Tool: scrape_website - Error: {e}")
        return f"Error scraping {url}: {e}"

# --- Tool 2: Smart File Downloader (Updated with Gemini Audio) ---
def download_and_read_file(url: str) -> str:
    """
    Downloads a file and saves it. 
    - JS/Logic -> temp_utils.js
    - CSV/Text/Log -> temp_data.csv
    - Audio -> Transcribes via Gemini API
    - Images -> OCR/Description via Gemini Vision
    """
    logger.info(f"Tool: download_and_read_file - URL: {url}")
    try:
        response = requests.get(url, timeout=NETWORK_TIMEOUT)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        
        # --- 1. Handle Audio Files (Gemini Transcription) ---
        if 'audio' in content_type or url.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg')):
            logger.info("Audio detected. Attempting transcription via Gemini...")
            try:
                api_key = os.getenv("GOOGLE_API_KEY_TOOLS")
                if not api_key:
                    return "Error: GOOGLE_API_KEY_TOOLS not found in environment variables."
                
                genai.configure(api_key=api_key)

                # Save binary audio to a temp file
                temp_audio_filename = "temp_audio_input.mp3"
                with open(temp_audio_filename, 'wb') as f:
                    f.write(response.content)
                
                # Upload to Gemini (File API)
                audio_file = genai.upload_file(path=temp_audio_filename)
                
                # Generate Content (Transcribe)
                # Using Gemini 1.5 Flash for speed/cost efficiency
                model = genai.GenerativeModel(model_name="gemini-2.5-flash")
                result = model.generate_content(
                    ["Transcribe this audio file verbatim.", audio_file]
                )
                
                return f"Audio Transcription (via Gemini):\n{result.text}"

            except Exception as e:
                logger.error(f"Gemini Transcription Error: {e}")
                return f"Error transcribing audio: {e}"

        # --- 2. Handle Images (OCR/Vision) [NEW INTEGRATION] ---
        elif 'image' in content_type or url.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
            logger.info("Image detected. Analyzing via Gemini Vision...")
            try:
                # Reuse the same API key variable as the original code
                api_key = os.getenv("GOOGLE_API_KEY_TOOLS")
                if not api_key:
                    return "Error: GOOGLE_API_KEY_TOOLS not found in environment variables."

                genai.configure(api_key=api_key)

                temp_filename = "temp_media_input.png"
                with open(temp_filename, 'wb') as f: 
                    f.write(response.content)
                
                media_file = genai.upload_file(path=temp_filename)
                model = genai.GenerativeModel(model_name="gemini-2.5-flash")
                
                prompt = (
                    "Analyze this image. 1. Transcribe any text or numbers visible exactly. "
                    "2. If it is a chart/graph, describe the x-axis, y-axis, and data points. "
                    "3. If it is a secret code or captcha, state it clearly."
                )
                result = model.generate_content([prompt, media_file])
                return f"Image Analysis (OCR & Description):\n{result.text}"
            except Exception as e:
                logger.error(f"Gemini Vision Error: {e}")
                return f"Error analyzing image: {e}"

        # --- 3. Consolidated File Naming (Existing Logic) ---
        filename = 'temp_downloaded_file' # Fallback
        
        if 'javascript' in content_type or url.lower().endswith('.js'):
            filename = 'temp_utils.js'
        elif 'json' in content_type or url.lower().endswith('.json'):
            filename = 'temp_data.json'
        elif ('csv' in content_type or 'text' in content_type or 
              url.lower().endswith(('.csv', '.txt', '.log'))):
            filename = 'temp_data.csv'
        # Fix for Databases/Zips
        elif url.lower().endswith(('.db', '.sqlite', '.sqlite3')):
            filename = 'temp_data.db'
        elif url.lower().endswith('.zip'):
            filename = 'temp_data.zip'  
        
        # --- 4. Handle Text-based files ---
        if any(x in content_type for x in ['text', 'javascript', 'json', 'csv']):
            text_content = response.text
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            preview = "\n".join(text_content.splitlines()[:10])
            return f"File saved as '{filename}'. Preview:\n{preview}"
            
        # --- 5. Handle PDF ---
        elif 'application/pdf' in content_type:
            with io.BytesIO(response.content) as f:
                with pdfplumber.open(f) as pdf:
                    text_content = "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())
            
            if not text_content:
                return "PDF downloaded, but no text extraction possible."
            
            return f"PDF text extracted. Content preview:\n{text_content[:1500]}..."

        # Handle Video - Ignore
        elif 'video' in content_type:
            return "File is video. Ignored."
        
        else:
            with open(filename, 'wb') as f:
                f.write(response.content)
            return f"Binary file saved as '{filename}'. Content not displayed."

    except Exception as e:
        logger.error(f"Tool: download_and_read_file - Error: {e}")
        return f"Error downloading {url}: {e}"

# --- Tool 3: File Reader ---
def read_file(filename: str) -> str:
    """Reads full content of a local file."""
    logger.info(f"Tool: read_file - Filename: {filename}")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if len(content) > TRUNCATE_LIMIT * 5: # Give a large buffer (10k chars)
            logger.warning(f"File {filename} is very large, truncating for observation.")
            return f"Full file content (truncated):\n{content[:TRUNCATE_LIMIT * 5]}\n...[TRUNCATED]"
        else:
            return f"Full file content of {filename}:\n{content}"
            
    except Exception as e:
        return f"Error reading {filename}: {e}"

# --- Tool 4: File Writer ---
def write_to_file(filename: str, content: str, mode: str = 'w') -> str:
    """Writes text to a local file. Essential for creating Python scripts."""
    logger.info(f"Tool: write_to_file - Filename: {filename}")
    try:
        with open(filename, mode, encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {filename}."
    except Exception as e:
        return f"Error writing to {filename}: {e}"

# --- Tool 5: Code Executor (Dual Mode) ---
def run_python_code(code_string: str = None, filename: str = None) -> str:
    """
    Executes Python code.
    - Mode A (File): Runs a .py file via subprocess (Safe, supports imports).
    - Mode B (String): Runs a string via exec() (Fast, limits imports).
    """
    # Mode A: File Execution (Preferred for Q3/Q4)
    if filename:
        logger.info(f"Tool: run_python_code - Executing File: {filename}")
        try:
            # We use subprocess to run the file, ensuring it uses the same python
            result = subprocess.run(
                [sys.executable, filename],
                capture_output=True,
                text=True,
                timeout=15,
                encoding='utf-8'
            )
            output = result.stdout + result.stderr
            if not output: return "Script executed successfully (No Output)."
            return f"Output:\n{output}"
        except Exception as e:
            return f"Error executing file: {e}"

    # Mode B: String Execution (Legacy/Simple tasks)
    elif code_string:
        logger.info(f"Tool: run_python_code - Executing String")
        buffer = io.StringIO()
        allowed_globals = {
            "pd": pd, "plt": plt, "io": io, "base64": base64, 
            "json": json, "re": re, "open": open,
            "print": lambda *args: buffer.write(" ".join(map(str, args)) + "\n")
        }
        try:
            with redirect_stdout(buffer):
                exec(code_string, allowed_globals)
            return f"Output:\n{buffer.getvalue()}"
        except Exception as e:
            return f"Error executing string: {e}"
    
    return "Error: No code or filename provided."

# --- Tool 6: Answer Submitter (SECURED) ---
def submit_answer(submit_url: str, answer_payload: dict, email: str, secret: str, task_url: str) -> str:
    logger.info(f"Tool: submit_answer - URL: {submit_url}")
    try:
        # --- CRITICAL FIX ---
        # Remove 'secret', 'email', 'url' from agent's payload to prevent overwriting
        safe_payload = {k: v for k, v in answer_payload.items() if k not in ['email', 'secret', 'url']}
        
        final_payload = {
            "email": email,
            "secret": secret, # <--- THIS MUST BE THE ARGUMENT FROM ENV
            "url": task_url,
            **safe_payload    # <--- Agent's answer is merged here
        }
        
        response = requests.post(submit_url, json=final_payload, timeout=NETWORK_TIMEOUT)
        return f"Submission response: {response.text}"
    except Exception as e:
        return f"Error submitting answer: {e}"