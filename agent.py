"""
Enhanced Quiz Agent - Designed to solve all 25 question types
"""
import json
import re
import requests
from urllib.parse import urljoin
import time
import google.generativeai as genai

from config import logger, GOOGLE_API_KEY
import tools

# =============================================================================
# SYSTEM PROMPT - General Purpose Problem Solver
# =============================================================================
SYSTEM_PROMPT = """
You are an autonomous problem-solving agent. You receive tasks, analyze them, and solve them step-by-step using available tools.

## RESPONSE FORMAT (CRITICAL)
Respond with ONLY a valid JSON object:
{
  "thinking": "Your reasoning about the current observation and what it means",
  "plan": "Your next steps to solve the problem",
  "action": {"tool": "tool_name", "parameters": {...}}
}

## AVAILABLE TOOLS

1. `scrape_website(url)` - Fetch and render a webpage (executes JavaScript). Returns visible text, links, and HTML.

2. `download_and_read_file(url, save_as=None)` - Download and process any file:
   - Audio files -> Returns transcription
   - Images -> Returns color analysis and OCR/description. Use `save_as` to save with custom filename.
   - PDFs -> Returns extracted text and tables
   - ZIP files -> Extracts and shows contents
   - CSV/JSON/Text -> Saves locally and shows preview
   - **IMPORTANT**: Use `save_as` parameter to save files with specific names (e.g., for comparing multiple images)

3. `read_file(filename)` - Read a local file's contents.

4. `write_to_file(filename, content, mode='w')` - Write content to a file.

5. `run_python_code(code_string=None, filename=None)` - Execute Python code.
   - For simple code: use `code_string`
   - For complex code: first write to file, then execute with `filename`
   - Available libraries: pandas, numpy, json, re, math, datetime, PIL, requests, collections

6. `submit_answer(submit_url, answer_payload)` - Submit your answer.
   - Format: `{"answer": <your_answer>}`
   - The system auto-adds email, secret, and task URL

7. `github_api(endpoint, params, save_to_file=None)` - Make GitHub API requests.
   - Example: `github_api("/repos/owner/repo/contents/path", {"ref": "main"})`
   - **IMPORTANT for large trees**: Use `save_to_file="tree_data.json"` to save full response, then process with run_python_code
   - Example: `github_api("/repos/owner/repo/git/trees/sha", {"recursive": 1}, save_to_file="tree.json")`

8. `compare_images(image1_path, image2_path)` - Compare two images pixel by pixel.
   - **Supports both local files AND URLs** - can directly compare two URLs
   - Returns: count of differing pixels

9. `get_dominant_color(image_path)` - Find the most frequent color in an image.

10. `http_request(url, method, headers, params, body)` - Make custom HTTP requests with specific headers.

## PROBLEM-SOLVING METHODOLOGY

### Phase 1: GATHER ALL RESOURCES FIRST (CRITICAL!)
**Before analyzing or making ANY assumptions, download ALL linked files:**
- Look at the `links` section from scrape_website output
- Download EVERY file: .csv, .json, .opus, .mp3, .wav, .zip, .pdf, .png, .jpg, etc.
- Audio files → Get transcription (may contain key values like cutoffs, passphrases, codes)
- CSV/JSON files → Get the actual data content
- ZIP files → Extract and see what's inside
- Images → Get descriptions/OCR
- **DO NOT assume what a file contains based on its name or the page text!**
- The page description might show placeholder values, but the REAL values are IN the files

### Phase 2: UNDERSTAND (Only after gathering)
- NOW read the task with full context from all downloaded files
- What values did you extract from audio/files?
- What is the actual data in the CSV/JSON?
- What format should the answer be in?

### Phase 3: PROCESS
- Use the ACTUAL values from downloaded files, not assumptions
- For data tasks: Write Python code to process/analyze
- For calculations: Use values extracted from files
- For transformations: Follow exact specifications given

### Phase 4: VALIDATE
- Does your answer match the required format?
- Did you use values from the FILES, not from page text?
- For numeric answers: check rounding, units
- For JSON answers: ensure valid syntax

### Phase 5: SUBMIT
- Extract the submission URL from the task page
- Submit with the correct answer format

## KEY PRINCIPLES

1. **GATHER FIRST, THINK LATER.** Download ALL files before making any assumptions about the task.

2. **Files contain the truth.** The page text might show example values, but the ACTUAL values are in the files (audio, CSV, JSON, etc.).

3. **Use Python for complex logic.** Write scripts for data processing, calculations, aggregations.

4. **Check the answer format.** Is it a number? String? JSON object? Array? Match it exactly.

5. **The submission URL is on the page.** Look for `/submit` or similar endpoint.

6. **If wrong, re-read the task AND the file contents.** Did you use values from files or from page text?

7. **Personalized tasks use your email.** Email: 23f2000790@ds.study.iitm.ac.in (length: 30 chars)

## COMMON PATTERNS

- **"Download X and compute Y"** -> download_and_read_file -> run_python_code -> submit_answer
- **"What is the value of X in the file"** -> download_and_read_file -> extract answer -> submit_answer  
- **"Write code/command to do X"** -> Construct the code/command string -> submit as answer
- **"Transform data from X to Y"** -> download -> write Python script -> run -> submit result
- **API tasks** -> Read API docs/params from task -> make request -> process response -> submit
- **Compare two images** -> Use compare_images(url1, url2) directly with URLs

## HANDLING ERRORS

- **Wrong answer:** Re-read the question. Check format. Check calculations.
- **Parse error:** Your JSON might be malformed. Validate it.
- **Tool error:** Check parameters. URLs must be absolute or properly relative.
- **Timeout:** Simplify your approach. Break into smaller steps.

## CRITICAL TIPS

1. **CSV Date Parsing:** Ambiguous dates like "02/01/24" could be Feb 1 (MM/DD/YY) or Jan 2 (DD/MM/YY).
   - Compare with other dates in the same file to infer the format
   - If another date is "30/01/24", that's clearly DD/MM/YY (no month 30)
   - Use pandas with `dayfirst=True` for DD/MM/YY or `dayfirst=False` for MM/DD/YY
   - When in doubt, try both interpretations and use the one that's consistent with other dates

2. **Large API Responses:** For GitHub tree APIs with many files, use `save_to_file` parameter to save complete data.
   - Example: `github_api("/repos/owner/repo/git/trees/sha", {"recursive": 1}, save_to_file="tree.json")`
   - Then use `run_python_code` to read and process the saved JSON file

3. **Image Comparison:** Use `compare_images(url1, url2)` directly - it handles URLs. Don't waste time downloading manually.

4. **Answer Format:** If the task says "POST a JSON array" or "submit as JSON", the answer IS the JSON structure, not a string. Example: `{"answer": [{"id": 1}, {"id": 2}]}` NOT `{"answer": "[{...}]"}`

5. **Stringified JSON:** Some tasks explicitly ask for "stringified JSON" - then use a string: `{"answer": "{\\"key\\": \\"value\\"}"}`

6. **Rate Limiting Calculations:** Be careful with variable interpretation:
   - `retry_every` typically means "every N items/pages", NOT "every N seconds" (unless explicitly stated as time)
   - Calculate base time from the SLOWEST/most restrictive limit
   - Count retries: total_items / retry_every (use integer division)
   - Total time = ceil(base_time) + (num_retries × retry_delay_seconds / 60)
   - Apply ceiling only at the END, not to intermediate values

7. **Don't Retry the Same Thing:** If an answer is wrong, the approach is wrong. Try a DIFFERENT interpretation, not the same code again.

8. **Tool Call JSON Plans:** When asked to create a JSON plan/array of tool calls:
   - Read the tool schemas in the provided JSON file VERY carefully
   - Use EXACT parameter names from the schema (case-sensitive)
   - For parameters that reference output from previous tools, DON'T use template syntax like `{{variable}}`
   - Instead, use the exact reference format shown in the schema (e.g., `"text": "$fetch_issue.output"` or simply omit if the tool chains automatically)
   - Check if parameters should be strings or integers (e.g., `"id": 42` vs `"id": "42"`)
   - The `query` parameter for search tools should match the task description closely

9. **File Naming:** When you download a file with `save_as` parameter, remember the exact filename you used.
   - Downloaded files go to the working directory
   - If you use `save_as="orders.csv"`, the file will be `orders.csv` NOT `temp_data.csv`
   - Always use the same filename in subsequent `read_file` or `run_python_code` operations

10. **Start/Initial Questions:** Some quizzes start with a warm-up question that accepts any non-empty answer.
    - Try "start" or any simple string - don't submit empty strings
    - Empty string submissions often fail with "Missing field answer"

11. **ALWAYS Download ALL Files First:** This is the #1 cause of wrong answers!
    - See a .csv link? Download it BEFORE assuming what's in it
    - See an .opus/.mp3/.wav link? Transcribe it BEFORE assuming any values
    - See a .json link? Download it BEFORE reasoning about the schema
    - See a .zip link? Extract it BEFORE guessing the contents
    - The page text often shows EXAMPLE or PLACEHOLDER values - the REAL data is in the files!
    - Example: Page says "Cutoff: 48266" but the audio file says "The cutoff is 12345" - use 12345!

Your entire response must be valid JSON. No text outside the JSON object.
"""

# =============================================================================
# Helper Functions
# =============================================================================

def extract_submit_url(html_content: str) -> str:
    """Extract submission URL from page content."""
    # Try various patterns
    patterns = [
        r'https?://[^\s"\'<>]*/submit',
        r'/submit'
    ]
    for pattern in patterns:
        match = re.search(pattern, html_content)
        if match:
            return match.group(0)
    return None


def parse_json_response(response_text: str) -> dict:
    """Robustly parse JSON from LLM response."""
    try:
        return json.loads(response_text)
    except:
        pass
    
    # Try to extract JSON from response
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    
    return None


def parse_submission_response(response_text: str) -> dict:
    """Parse the quiz server response."""
    try:
        return json.loads(response_text)
    except:
        pass
    
    # Extract JSON from HTML or mixed content
    json_match = re.search(r'\{[^{}]*"correct"[^{}]*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    
    return {"correct": False, "reason": "Could not parse response", "url": None}


# =============================================================================
# Main Agent Loop
# =============================================================================

def solve_quiz_task(email: str, secret: str, url: str, deadline: float = None):
    """
    The main autonomous agent loop.
    """
    logger.info(f"=== Starting New Task === Email: {email}, URL: {url}")
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT
    )
    
    history = []
    max_steps = 25
    wrong_answer_count = 0  # Track failed attempts
    last_next_url = None  # Track if we have a skip option
    
    # Derive base URL and default submit URL
    from urllib.parse import urlparse
    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    default_submit_url = f"{base_url}/submit"
    fallback_submit_url = default_submit_url  # Start with default
    
    # Initial scrape
    logger.info(f"--- Step 1/{max_steps} --- Initial scrape")
    try:
        observation = tools.scrape_website(url)
        found_url = extract_submit_url(observation)
        if found_url:
            if found_url.startswith('/'):
                found_url = urljoin(url, found_url)
            fallback_submit_url = found_url
            logger.info(f"Found submit URL: {fallback_submit_url}")
        else:
            logger.info(f"Using default submit URL: {fallback_submit_url}")
    except Exception as e:
        logger.error(f"Initial scrape failed: {e}")
        observation = f"Error scraping {url}: {e}"

    for step in range(1, max_steps):
        # Check time/step limits
        time_remaining = deadline - time.time() if deadline else 180
        is_time_panic = deadline and time_remaining < 15
        is_step_panic = step >= max_steps - 2
        
        if is_time_panic or is_step_panic:
            logger.warning(f"!!! PANIC MODE - {'Time' if is_time_panic else 'Steps'} running out !!!")
            target_url = fallback_submit_url or urljoin(url, "/submit")
            
            try:
                panic_response = tools.submit_answer(
                    submit_url=target_url,
                    answer_payload={"answer": "0"},
                    email=email,
                    secret=secret,
                    task_url=url
                )
                resp_dict = parse_submission_response(panic_response)
                new_url = resp_dict.get("url")
                
                if new_url:
                    logger.info(f"Panic submission got new URL: {new_url}")
                    solve_quiz_task(email, secret, new_url, deadline=time.time() + 170)
                else:
                    logger.info("No new URL from panic submission")
            except Exception as e:
                logger.error(f"Panic mode failed: {e}")
            break

        time_str = f"{round(time_remaining, 1)}s" if deadline else "N/A"
        logger.info(f"--- Step {step+1}/{max_steps} --- Time: {time_str}")
        
        # Add observation to history
        history.append({
            "role": "user",
            "parts": [f"Observation:\n{observation}\n\nProvide your JSON response:"]
        })
        
        # Get LLM response
        try:
            logger.info("Calling Gemini...")
            response = model.generate_content(
                history,
                generation_config={"response_mime_type": "application/json"}
            )
            response_text = response.text
            
            llm_response = parse_json_response(response_text)
            if not llm_response:
                raise ValueError("Could not parse JSON from LLM response")
            
            thinking = llm_response.get("thinking", llm_response.get("analysis", ""))
            plan = llm_response.get("plan", "")
            action_obj = llm_response.get("action")
            
            logger.info(f"Thinking: {thinking[:200]}...")
            logger.info(f"Plan: {plan[:200]}...")
            logger.info(f"Action: {json.dumps(action_obj)[:300]}")
            
            history.append({"role": "model", "parts": [json.dumps(llm_response)]})
            
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            observation = f"Error: LLM call failed - {e}. Try again."
            continue

        # Execute action
        try:
            if not isinstance(action_obj, dict):
                raise ValueError(f"Action must be dict, got {type(action_obj)}")
            
            tool_name = action_obj.get("tool") or action_obj.get("type") or action_obj.get("name")
            params = action_obj.get("parameters") or action_obj.get("args") or action_obj.get("params") or {}
            
            # Resolve relative URLs
            if tool_name in ["scrape_website", "download_and_read_file", "http_request"]:
                if "url" in params and params["url"] and not params["url"].startswith("http"):
                    params["url"] = urljoin(url, params["url"])
            
            if tool_name == "submit_answer":
                # ALWAYS use /submit endpoint - either from page scrape or default
                # The agent sometimes incorrectly uses the question URL as submit URL
                if fallback_submit_url:
                    submit_url = fallback_submit_url
                else:
                    submit_url = urljoin(url, "/submit")
                params["submit_url"] = submit_url

            # Execute tools
            logger.info(f"Executing: {tool_name}")
            
            if tool_name == "scrape_website":
                observation = tools.scrape_website(url=params["url"])
                found = extract_submit_url(observation)
                if found:
                    fallback_submit_url = urljoin(url, found) if found.startswith('/') else found
                    
            elif tool_name == "download_and_read_file":
                observation = tools.download_and_read_file(
                    url=params["url"],
                    save_as=params.get("save_as")
                )
                
            elif tool_name == "read_file":
                observation = tools.read_file(filename=params["filename"])
                
            elif tool_name == "write_to_file":
                observation = tools.write_to_file(
                    filename=params["filename"],
                    content=params["content"],
                    mode=params.get("mode", "w")
                )
                
            elif tool_name == "run_python_code":
                observation = tools.run_python_code(
                    code_string=params.get("code_string"),
                    filename=params.get("filename")
                )
                
            elif tool_name == "github_api":
                observation = tools.github_api(
                    endpoint=params["endpoint"],
                    params=params.get("params"),
                    save_to_file=params.get("save_to_file")
                )
                
            elif tool_name == "compare_images":
                observation = tools.compare_images(
                    image1_path=params["image1_path"],
                    image2_path=params["image2_path"]
                )
                
            elif tool_name == "get_dominant_color":
                observation = tools.get_dominant_color(image_path=params["image_path"])
                
            elif tool_name == "http_request":
                observation = tools.http_request(
                    url=params["url"],
                    method=params.get("method", "GET"),
                    headers=params.get("headers"),
                    params=params.get("params"),
                    body=params.get("body")
                )
                
            elif tool_name == "calculate":
                observation = tools.calculate(
                    expression=params["expression"],
                    variables=params.get("variables")
                )
                
            elif tool_name == "submit_answer":
                observation = tools.submit_answer(
                    submit_url=params["submit_url"],
                    answer_payload=params["answer_payload"],
                    email=email,
                    secret=secret,
                    task_url=url
                )
                
                logger.info(f"Submission response: {observation[:500]}")
                
                resp_dict = parse_submission_response(observation)
                is_correct = resp_dict.get("correct", False)
                new_url = resp_dict.get("url")
                reason = resp_dict.get("reason", "")
                
                if is_correct:
                    logger.info("[OK] Answer CORRECT!")
                    if new_url:
                        logger.info(f"[NEXT] Moving to: {new_url}")
                        solve_quiz_task(email, secret, new_url, deadline=time.time() + 170)
                    else:
                        logger.info("[DONE] Quiz complete!")
                    return
                else:
                    wrong_answer_count += 1
                    logger.warning(f"[WRONG] Answer INCORRECT ({wrong_answer_count}x): {reason}")
                    
                    # Track skip URL if available (server provides next URL even on wrong answer)
                    if new_url:
                        last_next_url = new_url
                        logger.info(f"[SKIP URL] Available: {last_next_url}")
                    
                    # If stuck too long (3 wrong attempts), skip to next question
                    if wrong_answer_count >= 3 and last_next_url:
                        logger.warning(f"[STUCK] {wrong_answer_count} wrong attempts. Skipping to: {last_next_url}")
                        solve_quiz_task(email, secret, last_next_url, deadline=time.time() + 170)
                        return
                    
                    observation = f"INCORRECT! Server reason: {reason}. Attempt {wrong_answer_count}/3."
                    if wrong_answer_count >= 2:
                        observation += " WARNING: Next wrong answer will SKIP this question. Try a COMPLETELY DIFFERENT approach."
                    continue
            else:
                observation = f"Unknown tool: {tool_name}"
                
        except Exception as e:
            logger.error(f"Action execution error: {e}")
            observation = f"Error executing {tool_name}: {e}"

    logger.info(f"=== Task Finished === URL: {url}")


# =============================================================================
# Entry point for testing
# =============================================================================
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test with demo
    solve_quiz_task(
        email="23f2000790@ds.study.iitm.ac.in",
        secret=os.getenv("MY_SECRET"),
        url="https://tds-llm-analysis.s-anand.net/demo",
        deadline=time.time() + 170
    )
