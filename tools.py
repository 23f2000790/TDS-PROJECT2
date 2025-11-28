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
import google.generativeai as genai

from config import logger

# --- Configuration ---
TRUNCATE_LIMIT = 2000
NETWORK_TIMEOUT = 30

# --- Tool 1: Web Scraper ---
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

# --- Tool 2: Smart File Downloader (Audio + Vision) ---
def download_and_read_file(url: str) -> str:
    """
    Downloads a file and saves it with a FIXED filename.
    - Images -> OCR/Vision
    - Audio -> Transcription
    - CSV/Text/Log -> temp_data.csv
    - DB/Binary -> temp_data.db / temp_file (FIX ADDED)
    """
    logger.info(f"Tool: download_and_read_file - URL: {url}")
    try:
        response = requests.get(url, timeout=NETWORK_TIMEOUT)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        
        # Setup Gemini if needed
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key: genai.configure(api_key=api_key)

        # --- 1. Handle Audio ---
        if 'audio' in content_type or url.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg')):
            if not api_key: return "Error: GOOGLE_API_KEY not found. Cannot transcribe audio."
            logger.info("Audio detected. Attempting transcription via Gemini...")
            try:
                temp_audio_filename = "temp_audio_input.mp3"
                with open(temp_audio_filename, 'wb') as f: f.write(response.content)
                
                audio_file = genai.upload_file(path=temp_audio_filename)
                model = genai.GenerativeModel(model_name="gemini-2.5-flash")
                result = model.generate_content(["Transcribe this audio file verbatim.", audio_file])
                return f"Audio Transcription:\n{result.text}"
            except Exception as e:
                return f"Error transcribing audio: {e}"

        # --- 2. Handle Images ---
        elif 'image' in content_type or url.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
            if not api_key: return "Error: GOOGLE_API_KEY not found. Cannot analyze image."
            logger.info("Image detected. Analyzing via Gemini Vision...")
            try:
                temp_img_filename = "temp_image_input.png"
                with open(temp_img_filename, 'wb') as f: f.write(response.content)
                
                img_file = genai.upload_file(path=temp_img_filename)
                model = genai.GenerativeModel(model_name="gemini-2.5-flash")
                prompt = (
                    "Analyze this image details. "
                    "1. If it contains text, transcribe it exactly. "
                    "2. If it is a chart, describe the data trends and values. "
                    "3. If it is a captcha or secret code, state the code clearly."
                )
                result = model.generate_content([prompt, img_file])
                return f"Image Analysis Result:\n{result.text}"
            except Exception as e:
                return f"Error analyzing image: {e}"

        # --- 3. Consolidated File Naming ---
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
            return f"PDF text extracted. Content preview:\n{text_content[:1500]}..."

        # --- 6. Handle Generic/Binary Files (The Fix) ---
        else:
            # Save the raw bytes for any other type (DB, Zip, etc.)
            with open(filename, 'wb') as f:
                f.write(response.content)
            return f"Binary file saved as '{filename}'. Content not displayed."

    except Exception as e:
        return f"Error downloading {url}: {e}"

# --- Tool 3: Read File ---
def read_file(filename: str) -> str:
    logger.info(f"Tool: read_file - Filename: {filename}")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        if len(content) > TRUNCATE_LIMIT * 5:
            return f"Content (truncated):\n{content[:TRUNCATE_LIMIT * 5]}..."
        return f"Content of {filename}:\n{content}"
    except Exception as e:
        return f"Error reading {filename}: {e}"

# --- Tool 4: Write File ---
def write_to_file(filename: str, content: str, mode: str = 'w') -> str:
    logger.info(f"Tool: write_to_file - Filename: {filename}")
    try:
        with open(filename, mode, encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {filename}."
    except Exception as e:
        return f"Error writing to {filename}: {e}"

# --- Tool 5: Run Python ---
def run_python_code(code_string: str = None, filename: str = None) -> str:
    if filename:
        logger.info(f"Tool: run_python_code - Executing File: {filename}")
        try:
            result = subprocess.run(
                [sys.executable, filename],
                capture_output=True, text=True, timeout=15, encoding='utf-8'
            )
            output = result.stdout + result.stderr
            if not output: return "Script executed successfully (No Output)."
            return f"Output:\n{output}"
        except Exception as e:
            return f"Error executing file: {e}"
    elif code_string:
        logger.info(f"Tool: run_python_code - Executing String")
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                exec(code_string, {'pd': pd, 'plt': plt, 'io': io, 'base64': base64, 'json': json, 're': re, 'open': open, 'print': lambda *args: buffer.write(" ".join(map(str, args)) + "\n")})
            return f"Output:\n{buffer.getvalue()}"
        except Exception as e:
            return f"Error executing string: {e}"
    return "Error: No code provided."

# --- Tool 6: Submit Answer ---
def submit_answer(submit_url: str, answer_payload: dict, email: str, secret: str, task_url: str) -> str:
    logger.info(f"Tool: submit_answer - URL: {submit_url}")
    try:
        # Sanitize payload
        safe_payload = {k: v for k, v in answer_payload.items() if k not in ['email', 'secret', 'url']}
        
        # Auto-convert numeric strings
        if 'answer' in safe_payload:
            ans = safe_payload['answer']
            if isinstance(ans, str) and ans.replace('.', '', 1).isdigit():
                if '.' in ans: safe_payload['answer'] = float(ans)
                else: safe_payload['answer'] = int(ans)

        final_payload = {
            "email": email,
            "secret": secret,
            "url": task_url,
            **safe_payload 
        }
        
        response = requests.post(submit_url, json=final_payload, timeout=NETWORK_TIMEOUT)
        return f"Submission response: {response.text}"
    except Exception as e:
        return f"Error submitting answer: {e}"