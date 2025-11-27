# Use Python 3.10
FROM python:3.10-slim

# 1. Install basic tools
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies (including Playwright) AS ROOT
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

# 3. Install Playwright System Dependencies AS ROOT
#    (This installs the missing libraries like libglib, libnss3, etc.)
RUN playwright install-deps

# 4. Setup the non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 5. Set working directory
WORKDIR $HOME/app

# 6. Install the Browser Binary AS USER
#    (This ensures the browser is downloaded to /home/user/.cache, accessible by the app)
RUN playwright install chromium

# 7. Copy the application code
#    (We use --chown because we are already the 'user')
COPY --chown=user . .

# 8. Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]