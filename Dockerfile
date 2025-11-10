FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app

WORKDIR ${APP_HOME}

# --- System dependencies (minimal) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# --- Copy project files (artifacts & venv excluded via .dockerignore) ---
COPY . .

# --- Install Python dependencies ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Expose ports ---
EXPOSE 8082
EXPOSE 9101

# --- Run app ---
CMD ["python", "webapp/app.py"]
