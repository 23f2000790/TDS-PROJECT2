import json
import re
import requests
from urllib.parse import urljoin
import time
import google.generativeai as genai # Import SDK

from config import logger, GOOGLE_API_KEY
import tools

# --- 1. ORIGINAL SYSTEM PROMPT (RESTORED & IMPROVED) ---
SYSTEM_PROMPT = """
You are an expert-level autonomous data analyst agent.
Your goal is to solve a multi-step data quiz by creating and executing a plan.
PLANNING IS KEY. YOU ARE A PROFESSIONAL EXPERT OUT HERE.

You operate in a strict "Reason-Act" loop.
At each step, I will provide you with an "Observation".
If a question asks for email, then the email is `23f2000790@ds.study.iitm.ac.in`. Remember this might not be an answer for any question, this could be one of the step to get to the answer, USE THIS ONLY WHEN NO INFO ABOUT THE EMAIL IS PROVIDED IN THE QUESTION.
You MUST respond with a single, valid JSON block.
NOTE: You must never assume any urls for quiz/submission by yourself. There will be hints, decode.
NOTE: The submission server might not be the same as the quiz url.

**CRITICAL: JSON-ONLY RESPONSE**
You MUST NOT write any text or explanations before or after the JSON block.
Your entire response must be ONLY the JSON object.

**CRITICAL: RESPONSE STRUCTURE**
Your JSON response MUST contain three keys:
1.  `"analysis"`: Your step-by-step analysis of the 'Observation'. What does it mean? What are the clues (files, links, hidden text)? What are the red herrings?
2.  `"plan"`: Your high-level, multi-step plan to solve the *entire* task from this point forward.
3.  `"action"`: The *single* tool call that executes the *first step* of your plan.

**YOUR CORE HEURISTICS (How to Think):**

BEFORE STARTING THE SOLUTION, you MUST identify the submission endpoint. The endpoint will be tricked for you in many ways, either encoded or half displayed. You have to keep your senses active.

1.  **YOUR JOB AT SUBMISSION:** The `submit_answer` tool is "smart." It automatically adds the correct `email`, `secret`, and `url`. Your job is to provide *only* the answer. Your `answer_payload` MUST *only* contain the `{"answer": ...}` key.

2.  **CLUES ARE NOT ANSWERS :** An "action" (like a link or file) is **almost always the correct path.** Your `plan` MUST be to process the "data" to find the *next "action"*.

3.  **TEST BEFORE SERVING CODES :** For any question, if you are using `run_python_code`, you must test/decode/validate the logic before submission.

4.  **LOGIC FILES REQUIRE FULL CONTENT.** For *logic files* (`.js`, `.txt`), you must use the `download_and_read_file` tool. Immediately following this, you must use `read_file(filename='...')` to see the full content before planning further. **Do not plan based on the preview alone.**

5.  **JAVASCRIPT IS THE TRUTH:** The logic found in a `.js` file is *always* the "truth" for calculations. It is more important than any static text.

6.  **HTML IS THE GOAL :** The *HTML page* contains the *goal*.

BONUS HEURISTICS (Advanced):
IF A PAGE HAS ANY FILES first of all, you must download them before arrriving to solutions. They might have important info regarding submission endpoints, or the question itself.
**HANDLE FILES FIRST:** If a page has an AUDIO, IMAGE, CSV, or TXT file, you MUST use the `download_and_read_file` tool to get it.
* **For Audio:** The tool automatically generates a transcript.
* **For Images:** The tool automatically performs OCR and Visual Analysis (extracting text/data).
* **NEVER** try to use Python libraries like `pytesseract` or `PIL` for images; use the tool instead.

7.  **PYTHON MUST BE JSON-SAFE :** When using `run_python_code` with `code_string`, your code *must* be a single-line JSON string, using `\n` for newlines.

8.  **USE YOUR MEMORY:** Use `write_to_file` to save key facts (like `GOAL=sum` or `CUTOFF=123`) for later Python steps.

9.  **JS vs. HTML MISMATCH (Retry Logic):** If the static HTML value contradicts the logic in a `.js` file, follow the logic in this order:
    * **First Attempt:** Follow **Heuristic #5 (JAVASCRIPT IS THE TRUTH)**.
    * **If Submission Fails:** Immediately retry the problem using the **static HTML value** instead.

10. **EFFICIENCY: ONE-SHOT CALCULATION.** Your goal is to be efficient. For final data tasks (like Q2 or Q3), your calculation **must be done in a single `run_python_code` action** whenever possible.

11. **AVOID JSON ERRORS (File Execution):** To run complex code, use a 2-step plan:
    * **Step 1:** Use `write_to_file` to save your complex Python script (e.g., `script.py`).
    * **Step 2:** Use `run_python_code(filename='script.py')` to execute it.
    * This **avoids all "unterminated f-string literal" errors.**

12. **FILE NAMING IS FIXED.** When you use `download_and_read_file` for a `.csv` or `.json`, the file name is **always** saved as **`temp_data.csv`** (or `.json`). Your subsequent Python code **MUST** read from this fixed filename.

13. **`submit_answer` REQUIRES BOTH PARAMETERS.** The `submit_answer` tool *always* needs both `submit_url` and `answer_payload`. Your action *must* include both, e.g., `{"tool": "submit_answer", "parameters": {"submit_url": "...", "answer_payload": {...}}}`.

14. **FILE NAME PRE-FLIGHT CHECK.** Before any action reading `temp_data.csv`, confirm the file name is *correct* (`temp_data.csv`). **Do not use names from old tasks** (`temp_log.txt`).

15. **VISUALIZATION BOILERPLATE.** When creating a chart for Base64 submission (using `matplotlib.pyplot`), you **MUST** include `import matplotlib; matplotlib.use('Agg')` or `plt.switch_backend('Agg')` at the beginning of your script to ensure compatibility with the non-GUI execution environment. **DO NOT** use `plt.show()`.

16. **VISUALIZATION MUST USE FILE EXECUTION.** Due to the complexity of charting and encoding, all visualization tasks (Q4) **MUST** be executed using the **`write_to_file` $\rightarrow$ `run_python_code(filename='...')` pipeline** (Heuristic #11). The script **MUST** include the boilerplate (Heuristic #15).

17. **SUBMISSION URL TRAP:** If the instructions say "submit to /submit", use that EXACT path relative to the domain. Do NOT submit to the question URL itself. If your submission fails with a parsing error, you likely posted to the wrong URL (e.g., an HTML page instead of an API endpoint).

18. If you get INCORRECT answers multiple times for a question, take few seconds and think
    - what did the question ask for?
    - what is the reason given for incorrect answer?
    - does every think makes sense logically?
Moreover, TRUST your calculations and codes, if you get constant INCORRECT, it might be possible that the submission endpoint might be messed up.
If your answer fails, your first assumption must be that the submission endpoint might be wrong, letme check again.

19. The instructions for the submission URL/Endpoint will be made clear in the quiz page, you must give importance to them and smartly craft the submission server point. 

Remember, sometimes revisiting the question, the expected answer wheter its a code or a word or whatevenr, you might get a fresh perspective and identify mistakes you might have overlooked earlier.

**Available Tools:**
1.  `scrape_website(url: str)`
2.  `download_and_read_file(url: str)`
3.  `read_file(filename: str)`
4.  `write_to_file(filename: str, content: str, mode: str = 'w')`
5.  `run_python_code(code_string: str = None, filename: str = None)`
6.  `submit_answer(submit_url: str, answer_payload: dict)`
"""

def extract_submit_url(html_content: str) -> str:
    """Helper to find the submission URL in the HTML early on for Panic Mode."""
    match = re.search(r'https?://[^\s"\'<>]*/submit', html_content)
    if match:
        return match.group(0)
    return None

def solve_quiz_task(email: str, secret: str, url: str, deadline: float = None):
    """
    The main autonomous agent loop.
    """
    logger.info(f"--- Starting New Task --- Email: {email}, URL: {url}")
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Use gemini-1.5-flash or gemini-2.0-flash-exp as per availability
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash", # Or your available model
        system_instruction=SYSTEM_PROMPT
    )
    
    history = [] 
    
    max_steps = 20 # Increased slightly for recovery
    fallback_submit_url = None 

    logger.info(f"--- Step 1 / {max_steps} --- (Performing initial scrape)")
    try:
        observation = tools.scrape_website(url) 
        found_url = extract_submit_url(observation)
        if found_url:
            fallback_submit_url = found_url
            logger.info(f"Identified submission URL for emergency use: {fallback_submit_url}")
    except Exception as e:
        logger.error(f"Initial scrape failed: {e}")
        observation = f"Error: The initial scrape of {url} failed. {e}"


    for step in range(1, max_steps): 
        
        # --- PANIC MODE ---
        time_remaining = deadline - time.time() if deadline else 100
        is_time_panic = (deadline and time_remaining < 10)
        is_step_panic = (step >= max_steps - 1)
        
        if is_time_panic or is_step_panic:
            panic_reason = "Time is running out" if is_time_panic else "Max steps reached"
            logger.warning(f"!!! PANIC MODE ACTIVATED ({panic_reason}) !!!")
            logger.warning("Sending dummy answer to force next question URL.")
            
            target_url = fallback_submit_url
            if not target_url:
                # Default fallback if not found
                target_url = urljoin(url, "/submit")
            
            dummy_payload = {"answer": "0"} 
            
            try:
                panic_response = tools.submit_answer(
                    submit_url=target_url,
                    answer_payload=dummy_payload,
                    email=email,
                    secret=secret,
                    task_url=url
                )
                json_part = panic_response.split("Submission response:", 1)[-1].strip()
                resp_dict = json.loads(json_part)
                new_url = resp_dict.get("url")
                
                if new_url:
                    logger.info(f"Panic Mode Successful! Moving to next URL: {new_url}")
                    new_deadline = time.time() + 170
                    solve_quiz_task(email, secret, new_url, deadline=new_deadline)
                else:
                    logger.error("Panic Mode: Server did not return a new URL.")
            except:
                logger.error("Panic Mode: Failed.")
            
            break 
        # ------------------------

        time_left_str = f"{round(time_remaining, 1)}s" if deadline else "N/A"
        logger.info(f"--- Step {step+1} / {max_steps} --- (Time left: {time_left_str})")
        
        history.append({"role": "user", "parts": [f"Observation:\n{observation}\n\nProvide your JSON response:"]})
        
        try:
            logger.info("Agent is reasoning (calling Google Gemini SDK)...")
            
            response = model.generate_content(
                history,
                generation_config={"response_mime_type": "application/json"}
            )
            response_text = response.text
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("LLM did not return valid JSON.")

            json_str = json_match.group(0)
            llm_response = json.loads(json_str)
            
            analysis = llm_response.get("analysis", "No analysis provided.")
            plan = llm_response.get("plan", "No plan provided.")
            action_obj = llm_response.get("action") 
            
            logger.info(f"Agent Analysis: {analysis}")
            logger.info(f"Agent Plan: {plan}")
            logger.info(f"Agent Action: {action_obj}") 

            history.append({"role": "model", "parts": [json_str]})
            
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            observation = f"Error: Your last response was not valid or API call failed. Please try again. Error: {e}"
            continue 

        try:
            if not isinstance(action_obj, dict):
                raise ValueError(f"Action must be a dict, but got {type(action_obj)}")

            tool_name = action_obj.get("tool") or action_obj.get("type")
            if "parameters" in action_obj: args = action_obj.get("parameters", {})
            elif "args" in action_obj: args = action_obj.get("args", {})
            elif "kwargs" in action_obj: args = action_obj.get("kwargs", {})
            else: args = action_obj
            
            if tool_name in ["scrape_website", "download_and_read_file"]:
                call_url = args.get('url')
                if call_url and not call_url.startswith('http'):
                    call_url = urljoin(url, call_url) 
                    args['url'] = call_url 

            elif tool_name == "submit_answer":
                submit_url = args.get('submit_url')
                if submit_url and not submit_url.startswith('http'):
                    submit_url = urljoin(url, submit_url)
                    args['submit_url'] = submit_url
                fallback_submit_url = submit_url

            # --- Execute Tools ---
            if tool_name == "scrape_website":
                observation = tools.scrape_website(url=args['url'])
                found_url = extract_submit_url(observation)
                if found_url: fallback_submit_url = found_url
            
            elif tool_name == "download_and_read_file":
                observation = tools.download_and_read_file(url=args['url'])
            
            elif tool_name == "read_file":
                observation = tools.read_file(filename=args['filename'])

            elif tool_name == "write_to_file":
                mode = args.get('mode', 'w')
                observation = tools.write_to_file(filename=args['filename'], content=args['content'], mode=mode)

            elif tool_name == "run_python_code":
                observation = tools.run_python_code(
                    code_string=args.get('code_string'),
                    filename=args.get('filename')
                )
            
            elif tool_name == "submit_answer":
                observation = tools.submit_answer(
                    submit_url=args['submit_url'],
                    answer_payload=args['answer_payload'],
                    email=email,
                    secret=secret,
                    task_url=url
                )
                
                logger.info("Submission complete. Analyzing response...")
                
                # --- FIX: Handle Non-JSON Responses Gracefully ---
                try:
                    json_response_part = observation.split("Submission response:", 1)[-1].strip()
                    submit_response_json = json.loads(json_response_part)
                    
                    is_correct = submit_response_json.get("correct", False)
                    new_url = submit_response_json.get("url")
                    reason = submit_response_json.get("reason", "No reason provided.")

                    if is_correct:
                        logger.info("Answer was CORRECT.")
                        if new_url:
                            logger.info(f"New URL found! Starting next task: {new_url}")
                            new_deadline = time.time() + 170
                            solve_quiz_task(email, secret, new_url, deadline=new_deadline)
                        else:
                            logger.info("No new URL found. Quiz is complete.")
                        break 

                    else:
                        logger.warning(f"Answer was INCORRECT. Reason: {reason}")
                        
                        # Retry Logic: Only continue if we are NOT in panic mode
                        observation = f"Submission Incorrect. Server Reason: {reason}. Analyze why it failed and try again."
                        continue 

                except json.JSONDecodeError:
                    # This catches the "Expecting value..." error
                    logger.error("Could not parse submission response. Server likely returned HTML.")
                    observation = (
                        "Error: The server response was not valid JSON. "
                        "This usually means you submitted to the WRONG URL (e.g., the problem page instead of /submit). "
                        "Check the 'submit_url' parameter."
                    )
                except Exception as e:
                    logger.error(f"General error parsing response: {e}")
                    observation = f"Error parsing submission response: {e}"

            else:
                observation = f"Error: Unknown tool '{tool_name}'."
        
        except Exception as e:
            logger.error(f"Error executing action '{action_obj}': {e}")
            observation = f"Error: Failed to execute action. {e}"

    logger.info(f"--- Task Finished --- URL: {url}")