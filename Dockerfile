# Use Python 3.10
FROM python:3.10-slim

# 1. Install system tools required for Playwright and general usage
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup a new user 'user' (Required for Hugging Face Spaces security)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 3. Set working directory to the user's app directory
WORKDIR $HOME/app

# 4. Install Python dependencies
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 5. Install Playwright Browsers and System Dependencies
RUN playwright install chromium
RUN playwright install-deps

# 6. Copy the rest of your application code
COPY --chown=user . .

# 7. Start the FastAPI server on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]