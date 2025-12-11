import os
import requests
import json
import base64
import io
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pdfplumber
import subprocess
import sys
import zipfile
from PIL import Image
from collections import Counter
from dateutil import parser as dateparser
import google.generativeai as genai

from config import logger

# --- Configuration ---
TRUNCATE_LIMIT = 8000
NETWORK_TIMEOUT = 45

# =============================================================================
# TOOL 1: Web Scraper (Playwright) - Enhanced
# =============================================================================
def scrape_website(url: str) -> str:
    """
    Fetches a URL, executes JavaScript via Playwright, and returns:
    - Full HTML content
    - Rendered visible text (for easy reading)
    - All links found on page
    """
    logger.info(f"Tool: scrape_website - URL: {url}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=NETWORK_TIMEOUT * 1000)
            
            # Wait for dynamic content
            try:
                page.wait_for_load_state('networkidle', timeout=5000)
            except:
                pass
            
            content = page.content()
            
            # Get visible text
            try:
                body_text = page.inner_text('body')
            except:
                body_text = ""
            
            browser.close()
            
            soup = BeautifulSoup(content, 'lxml')
            if not body_text:
                body_text = soup.get_text(separator='\n', strip=True)
            
            # Extract all links
            links = []
            for a in soup.find_all('a', href=True):
                href = a.get('href')
                text = a.get_text(strip=True)
                links.append({"href": href, "text": text})
            
            # Extract all data from page
            result = f"=== RENDERED TEXT ===\n{body_text[:6000]}\n\n"
            result += f"=== LINKS FOUND ===\n{json.dumps(links[:50], indent=2)}\n\n"
            result += f"=== FULL HTML ===\n{str(soup)[:8000]}"
            
            return result
            
    except Exception as e:
        logger.error(f"Tool: scrape_website - Error: {e}")
        return f"Error scraping {url}: {e}"


# =============================================================================
# TOOL 2: Smart File Downloader - Massively Enhanced
# =============================================================================
def download_and_read_file(url: str, save_as: str = None) -> str:
    """
    Downloads a file and intelligently processes it based on type:
    - Audio -> Gemini transcription
    - Image -> OCR/Vision analysis OR pixel analysis
    - PDF -> Text extraction with table detection
    - CSV/JSON/Text -> Save and preview
    - ZIP -> Extract and list contents
    
    Args:
        url: The URL to download from
        save_as: Optional custom filename to save the file as (useful for comparing multiple files)
    """
    logger.info(f"Tool: download_and_read_file - URL: {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*'
        }
        response = requests.get(url, timeout=NETWORK_TIMEOUT, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        
        # --- AUDIO FILES ---
        if 'audio' in content_type or url.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus')):
            logger.info("Audio detected. Transcribing via Gemini...")
            try:
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY_TOOLS")
                if not api_key:
                    return "Error: GOOGLE_API_KEY not found."
                
                genai.configure(api_key=api_key)
                
                # Determine extension
                ext = '.opus' if '.opus' in url.lower() else '.mp3'
                temp_audio = f"temp_audio{ext}"
                with open(temp_audio, 'wb') as f:
                    f.write(response.content)
                
                audio_file = genai.upload_file(path=temp_audio)
                model = genai.GenerativeModel(model_name="gemini-2.5-flash")
                result = model.generate_content([
                    "Transcribe this audio file EXACTLY and VERBATIM. Include any numbers, codes, or passphrases spoken. Return ONLY the transcription, nothing else.",
                    audio_file
                ])
                
                return f"AUDIO TRANSCRIPTION:\n{result.text}"
            except Exception as e:
                logger.error(f"Audio transcription error: {e}")
                return f"Error transcribing audio: {e}"

        # --- IMAGE FILES ---
        elif 'image' in content_type or url.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp')):
            logger.info("Image detected. Saving and analyzing...")
            
            # Save image - use custom name if provided, otherwise derive from URL or use default
            if save_as:
                img_filename = save_as
            else:
                # Try to get filename from URL
                url_filename = url.split('/')[-1].split('?')[0]
                if url_filename and '.' in url_filename:
                    img_filename = url_filename
                else:
                    img_filename = "temp_image.png"
            
            with open(img_filename, 'wb') as f:
                f.write(response.content)
            
            # Also do quick pixel analysis for heatmap-type questions
            try:
                img = Image.open(io.BytesIO(response.content)).convert('RGB')
                pixels = list(img.getdata())
                color_counts = Counter(pixels)
                top_colors = color_counts.most_common(10)
                
                color_info = "TOP 10 MOST FREQUENT COLORS (RGB):\n"
                for color, count in top_colors:
                    hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
                    color_info += f"  {hex_color} (RGB: {color}) - {count} pixels\n"
                
                most_common = top_colors[0][0]
                dominant_hex = '#{:02x}{:02x}{:02x}'.format(*most_common)
                color_info += f"\nMOST FREQUENT/DOMINANT COLOR: {dominant_hex}"
                
                # Vision analysis via Gemini
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY_TOOLS")
                if api_key:
                    try:
                        genai.configure(api_key=api_key)
                        media_file = genai.upload_file(path=img_filename)
                        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
                        vision_result = model.generate_content([
                            "Analyze this image. If it contains text, transcribe it exactly. If it's a chart/graph, describe the data. If it's a heatmap or color grid, identify the dominant color.",
                            media_file
                        ])
                        vision_text = vision_result.text
                    except Exception as ve:
                        vision_text = f"Vision analysis error: {ve}"
                else:
                    vision_text = "Vision analysis unavailable (no API key)"
                
                return f"Image saved as '{img_filename}'.\n\n{color_info}\n\nVISION ANALYSIS:\n{vision_text}"
            except Exception as e:
                return f"Image saved as '{img_filename}'. Analysis error: {e}"

        # --- PDF FILES ---
        elif 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            logger.info("PDF detected. Extracting text and tables...")
            try:
                pdf_content = io.BytesIO(response.content)
                all_text = []
                all_tables = []
                
                with pdfplumber.open(pdf_content) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        all_text.append(f"--- Page {i+1} ---\n{text}")
                        
                        # Extract tables
                        tables = page.extract_tables()
                        for j, table in enumerate(tables):
                            all_tables.append(f"Table {j+1} on Page {i+1}:\n{json.dumps(table, indent=2)}")
                
                result = "PDF TEXT CONTENT:\n" + "\n".join(all_text)
                if all_tables:
                    result += "\n\nPDF TABLES:\n" + "\n".join(all_tables)
                
                return result[:TRUNCATE_LIMIT]
            except Exception as e:
                return f"Error reading PDF: {e}"

        # --- ZIP FILES ---
        elif 'zip' in content_type or url.lower().endswith('.zip'):
            logger.info("ZIP detected. Extracting contents...")
            try:
                zip_buffer = io.BytesIO(response.content)
                extract_dir = "temp_extracted"
                os.makedirs(extract_dir, exist_ok=True)
                
                with zipfile.ZipFile(zip_buffer, 'r') as zf:
                    file_list = zf.namelist()
                    zf.extractall(extract_dir)
                
                result = f"ZIP extracted to '{extract_dir}'. Files:\n"
                for fname in file_list[:20]:
                    result += f"  - {fname}\n"
                
                # Try to read JSON/CSV files inside
                for fname in file_list:
                    if fname.endswith('.json') or fname.endswith('.csv') or fname.endswith('.log') or fname.endswith('.txt'):
                        fpath = os.path.join(extract_dir, fname)
                        if os.path.isfile(fpath):
                            try:
                                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()[:3000]
                                result += f"\n--- Content of {fname} ---\n{content}\n"
                            except:
                                pass
                
                return result
            except Exception as e:
                return f"Error extracting ZIP: {e}"

        # --- JSON FILES ---
        elif 'json' in content_type or url.lower().endswith('.json'):
            # Use save_as if provided, otherwise default
            if save_as:
                filename = save_as
            else:
                filename = 'temp_data.json'
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            try:
                data = json.loads(response.text)
                pretty = json.dumps(data, indent=2)
                return f"JSON saved as '{filename}'.\nContent:\n{pretty[:TRUNCATE_LIMIT]}"
            except:
                return f"JSON saved as '{filename}'.\nRaw content:\n{response.text[:TRUNCATE_LIMIT]}"

        # --- CSV/TEXT FILES ---
        elif 'csv' in content_type or 'text' in content_type or url.lower().endswith(('.csv', '.txt', '.log')):
            # Use save_as if provided, otherwise default based on type
            if save_as:
                filename = save_as
            elif url.lower().endswith('.csv') or 'csv' in content_type:
                filename = 'temp_data.csv'
            else:
                filename = 'temp_data.txt'
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            return f"File saved as '{filename}'.\nContent preview:\n{response.text[:TRUNCATE_LIMIT]}"

        # --- JAVASCRIPT FILES ---
        elif 'javascript' in content_type or url.lower().endswith('.js'):
            filename = 'temp_utils.js'
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(response.text)
            return f"JavaScript saved as '{filename}'.\nContent:\n{response.text[:TRUNCATE_LIMIT]}"

        # --- BINARY/OTHER ---
        else:
            filename = 'temp_downloaded_file'
            if '.' in url.split('/')[-1]:
                ext = '.' + url.split('/')[-1].split('.')[-1]
                filename = f'temp_file{ext}'
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            return f"Binary file saved as '{filename}' ({len(response.content)} bytes)."

    except Exception as e:
        logger.error(f"Tool: download_and_read_file - Error: {e}")
        return f"Error downloading {url}: {e}"


# =============================================================================
# TOOL 3: File Reader - Enhanced
# =============================================================================
def read_file(filename: str) -> str:
    """Reads full content of a local file."""
    logger.info(f"Tool: read_file - Filename: {filename}")
    try:
        # Handle binary files
        if filename.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.zip')):
            return f"File {filename} is binary. Use appropriate tools to process it."
        
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if len(content) > TRUNCATE_LIMIT:
            return f"File content (truncated):\n{content[:TRUNCATE_LIMIT]}\n...[TRUNCATED]"
        return f"Full content of {filename}:\n{content}"
            
    except Exception as e:
        return f"Error reading {filename}: {e}"


# =============================================================================
# TOOL 4: File Writer
# =============================================================================
def write_to_file(filename: str, content: str, mode: str = 'w') -> str:
    """Writes text to a local file."""
    logger.info(f"Tool: write_to_file - Filename: {filename}")
    try:
        with open(filename, mode, encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {filename}."
    except Exception as e:
        return f"Error writing to {filename}: {e}"


# =============================================================================
# TOOL 5: Code Executor - Massively Enhanced
# =============================================================================
def run_python_code(code_string: str = None, filename: str = None) -> str:
    """
    Executes Python code with full library support.
    Mode A (filename): Run a .py file via subprocess
    Mode B (code_string): Run code string via exec()
    
    Pre-imported libraries: pandas, numpy, json, re, requests, PIL, dateutil, collections
    """
    # Mode A: File Execution
    if filename:
        logger.info(f"Tool: run_python_code - Executing File: {filename}")
        try:
            result = subprocess.run(
                [sys.executable, filename],
                capture_output=True,
                text=True,
                timeout=60,
                encoding='utf-8',
                cwd=os.getcwd()
            )
            output = result.stdout + result.stderr
            if not output.strip():
                return "Script executed successfully (no output)."
            return f"Output:\n{output[:TRUNCATE_LIMIT]}"
        except subprocess.TimeoutExpired:
            return "Error: Script execution timed out (60s limit)."
        except Exception as e:
            return f"Error executing file: {e}"

    # Mode B: String Execution
    elif code_string:
        logger.info(f"Tool: run_python_code - Executing String")
        buffer = io.StringIO()
        
        # Rich execution environment
        exec_globals = {
            "__builtins__": __builtins__,
            "pd": pd,
            "np": np,
            "plt": plt,
            "io": io,
            "base64": base64,
            "json": json,
            "re": re,
            "os": os,
            "requests": requests,
            "Image": Image,
            "Counter": Counter,
            "dateparser": dateparser,
            "open": open,
            "print": lambda *args, **kwargs: buffer.write(" ".join(map(str, args)) + "\n"),
            "math": __import__('math'),
            "datetime": __import__('datetime'),
            "collections": __import__('collections'),
        }
        
        try:
            with redirect_stdout(buffer):
                exec(code_string, exec_globals)
            output = buffer.getvalue()
            if not output.strip():
                return "Code executed successfully (no output)."
            return f"Output:\n{output[:TRUNCATE_LIMIT]}"
        except Exception as e:
            return f"Error executing code: {e}"
    
    return "Error: No code or filename provided."


# =============================================================================
# TOOL 6: Answer Submitter - Robust
# =============================================================================
def submit_answer(submit_url: str, answer_payload, email: str, secret: str, task_url: str) -> str:
    """
    Submits an answer to the quiz server.
    Automatically handles:
    - Non-dict answer_payload (wraps in {"answer": ...})
    - FILE: prefix for reading answer from file
    - Type conversion for numeric answers
    - Proper payload construction
    """
    logger.info(f"Tool: submit_answer - URL: {submit_url}")
    try:
        # CRITICAL: Handle non-dict answer_payload
        # If agent passes int/str/float directly, wrap it properly
        if not isinstance(answer_payload, dict):
            logger.info(f"Converting non-dict answer_payload: {answer_payload} -> {{'answer': {answer_payload}}}")
            answer_payload = {"answer": answer_payload}
        
        # Handle FILE: prefix
        if 'answer' in answer_payload and isinstance(answer_payload['answer'], str):
            val = answer_payload['answer']
            if val.startswith("FILE:"):
                filename = val.replace("FILE:", "").strip()
                logger.info(f"Reading answer from file: {filename}")
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        answer_payload['answer'] = f.read().strip()
                except Exception as e:
                    return f"Error reading answer file {filename}: {e}"
        
        # Clean payload - remove any keys that shouldn't be there
        safe_payload = {k: v for k, v in answer_payload.items() if k not in ['email', 'secret', 'url']}
        
        # Smart type conversion for answer
        if 'answer' in safe_payload:
            ans = safe_payload['answer']
            if isinstance(ans, str):
                stripped = ans.strip()
                # Try to convert to number if it looks like one
                if re.match(r'^-?\d+$', stripped):
                    safe_payload['answer'] = int(stripped)
                elif re.match(r'^-?\d+\.\d+$', stripped):
                    safe_payload['answer'] = float(stripped)
        
        # Build final payload
        final_payload = {
            "email": email,
            "secret": secret,
            "url": task_url,
            **safe_payload
        }
        
        logger.info(f"Submitting payload: {json.dumps(final_payload, default=str)[:500]}")
        
        response = requests.post(submit_url, json=final_payload, timeout=NETWORK_TIMEOUT)
        return response.text
        
    except Exception as e:
        return f"Error submitting answer: {e}"


# =============================================================================
# TOOL 7: GitHub API - With Large Response Handling
# =============================================================================
def github_api(endpoint: str, params: dict = None, save_to_file: str = None) -> str:
    """
    Makes authenticated GitHub API requests.
    endpoint: e.g., "/repos/{owner}/{repo}/git/trees/{sha}"
    params: query parameters like {"recursive": "1"}
    save_to_file: Optional filename to save full response (for large data like trees)
    
    IMPORTANT: For tree endpoints with many files, use save_to_file to get complete data,
    then use run_python_code to analyze the file. Otherwise, the response will be truncated.
    """
    logger.info(f"Tool: github_api - Endpoint: {endpoint}, save_to_file: {save_to_file}")
    try:
        base_url = "https://api.github.com"
        url = base_url + endpoint
        
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'QuizAgent/1.0'
        }
        
        # Add token if available
        token = os.getenv("GITHUB_TOKEN")
        if token:
            headers['Authorization'] = f'token {token}'
        
        response = requests.get(url, headers=headers, params=params, timeout=NETWORK_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        full_json = json.dumps(data, indent=2)
        
        # If save_to_file is specified, save the COMPLETE response
        if save_to_file:
            with open(save_to_file, 'w', encoding='utf-8') as f:
                f.write(full_json)
            # Return summary info + file location
            tree_count = len(data.get('tree', [])) if 'tree' in data else 0
            return f"GitHub API Response saved to: {save_to_file}\n" \
                   f"Total items in tree: {tree_count}\n" \
                   f"File size: {len(full_json)} bytes\n" \
                   f"Use run_python_code to analyze the complete data from the saved file."
        
        # If response is large, warn about truncation
        if len(full_json) > TRUNCATE_LIMIT:
            tree_count = len(data.get('tree', [])) if 'tree' in data else 0
            return f"WARNING: Response truncated! Full response is {len(full_json)} bytes.\n" \
                   f"Total items in tree: {tree_count}\n" \
                   f"To get complete data, call github_api again with save_to_file parameter.\n\n" \
                   f"Partial response:\n{full_json[:TRUNCATE_LIMIT]}"
        
        return f"GitHub API Response:\n{full_json}"
        
    except Exception as e:
        return f"GitHub API Error: {e}"


# =============================================================================
# TOOL 8: Image Comparison - NEW
# =============================================================================
def compare_images(image1_path: str, image2_path: str) -> str:
    """
    Compares two images and counts differing pixels.
    Returns the count of pixels that differ in RGB values.
    Supports both local file paths and URLs.
    """
    logger.info(f"Tool: compare_images - {image1_path} vs {image2_path}")
    try:
        # Helper function to load image from path or URL
        def load_image(path_or_url, temp_name):
            if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
                response = requests.get(path_or_url, timeout=NETWORK_TIMEOUT)
                response.raise_for_status()
                # Save to temp file
                with open(temp_name, 'wb') as f:
                    f.write(response.content)
                return Image.open(temp_name).convert('RGB')
            else:
                return Image.open(path_or_url).convert('RGB')
        
        img1 = load_image(image1_path, "temp_compare_img1.png")
        img2 = load_image(image2_path, "temp_compare_img2.png")
        
        if img1.size != img2.size:
            return f"Error: Images have different sizes. {img1.size} vs {img2.size}"
        
        pixels1 = list(img1.getdata())
        pixels2 = list(img2.getdata())
        
        diff_count = sum(1 for p1, p2 in zip(pixels1, pixels2) if p1 != p2)
        
        return f"Image comparison result:\n- Image 1: {image1_path}\n- Image 2: {image2_path}\n- Differing pixels: {diff_count}\n- Total pixels: {len(pixels1)}"
        
    except Exception as e:
        return f"Error comparing images: {e}"


# =============================================================================
# TOOL 9: Extract Dominant Color - NEW
# =============================================================================
def get_dominant_color(image_path: str) -> str:
    """
    Analyzes an image and returns the most frequent color as hex.
    Supports both local file paths and URLs.
    """
    logger.info(f"Tool: get_dominant_color - {image_path}")
    try:
        # Handle URLs
        if image_path.startswith('http://') or image_path.startswith('https://'):
            response = requests.get(image_path, timeout=NETWORK_TIMEOUT)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert('RGB')
        else:
            img = Image.open(image_path).convert('RGB')
        
        pixels = list(img.getdata())
        color_counts = Counter(pixels)
        
        top_colors = color_counts.most_common(10)
        result = "Color analysis:\n"
        for i, (color, count) in enumerate(top_colors):
            hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
            result += f"{i+1}. {hex_color} (RGB: {color}) - {count} pixels ({100*count/len(pixels):.2f}%)\n"
        
        most_common = top_colors[0][0]
        dominant_hex = '#{:02x}{:02x}{:02x}'.format(*most_common)
        result += f"\nMOST FREQUENT COLOR: {dominant_hex}"
        
        return result
        
    except Exception as e:
        return f"Error analyzing image: {e}"


# =============================================================================
# TOOL 10: PDF Table Extractor - NEW  
# =============================================================================
def extract_pdf_tables(pdf_path_or_url: str) -> str:
    """
    Extracts tables from a PDF file, returns as JSON.
    """
    logger.info(f"Tool: extract_pdf_tables - {pdf_path_or_url}")
    try:
        # Download if URL
        if pdf_path_or_url.startswith('http'):
            response = requests.get(pdf_path_or_url, timeout=NETWORK_TIMEOUT)
            pdf_content = io.BytesIO(response.content)
        else:
            pdf_content = pdf_path_or_url
        
        all_tables = []
        with pdfplumber.open(pdf_content) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for j, table in enumerate(tables):
                    # Convert to list of dicts if has header
                    if table and len(table) > 1:
                        headers = [str(h).strip() if h else f"col{k}" for k, h in enumerate(table[0])]
                        rows = []
                        for row in table[1:]:
                            row_dict = {}
                            for k, val in enumerate(row):
                                if k < len(headers):
                                    row_dict[headers[k]] = val
                            rows.append(row_dict)
                        all_tables.append({
                            "page": i+1,
                            "table": j+1,
                            "headers": headers,
                            "rows": rows
                        })
        
        return f"PDF Tables Extracted:\n{json.dumps(all_tables, indent=2)}"
        
    except Exception as e:
        return f"Error extracting PDF tables: {e}"


# =============================================================================
# TOOL 11: HTTP API Request - NEW (for custom API calls with headers)
# =============================================================================
def http_request(url: str, method: str = "GET", headers: dict = None, params: dict = None, body: dict = None) -> str:
    """
    Makes HTTP requests with custom headers (useful for API calls).
    """
    logger.info(f"Tool: http_request - {method} {url}")
    try:
        default_headers = {
            'User-Agent': 'QuizAgent/1.0'
        }
        if headers:
            default_headers.update(headers)
        
        if method.upper() == "GET":
            response = requests.get(url, headers=default_headers, params=params, timeout=NETWORK_TIMEOUT)
        elif method.upper() == "POST":
            response = requests.post(url, headers=default_headers, params=params, json=body, timeout=NETWORK_TIMEOUT)
        else:
            return f"Unsupported method: {method}"
        
        return f"HTTP {method} Response (Status {response.status_code}):\n{response.text[:TRUNCATE_LIMIT]}"
        
    except Exception as e:
        return f"HTTP request error: {e}"


# =============================================================================
# TOOL 12: Calculate Expression - NEW (for rate limiting, shards, etc.)
# =============================================================================
def calculate(expression: str, variables: dict = None) -> str:
    """
    Safely evaluates a mathematical expression.
    variables: dict of variable names to values
    """
    logger.info(f"Tool: calculate - {expression}")
    try:
        import math
        
        safe_dict = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'len': len, 'int': int, 'float': float,
            'ceil': math.ceil, 'floor': math.floor, 'sqrt': math.sqrt,
            'pow': pow, 'math': math
        }
        
        if variables:
            safe_dict.update(variables)
        
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"Result: {result}"
        
    except Exception as e:
        return f"Calculation error: {e}"
