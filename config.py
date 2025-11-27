import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# --- Load Secrets ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MY_SECRET = os.getenv("MY_SECRET")

# --- Configure API ---
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found or not set in .env file.")
if not MY_SECRET:
    raise ValueError("MY_SECRET not found or not set in .env file.")

# Configure the Gemini Library
genai.configure(api_key=GOOGLE_API_KEY)

# --- Setup Professional Logger ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - (%(module)s.%(funcName)s) - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("test_run.log"), # Separate log file for testing
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Configuration loaded. Logging is active. Using Google Gemini API directly.")