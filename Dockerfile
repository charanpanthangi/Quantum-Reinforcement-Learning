# Lightweight Python image
FROM python:3.11-slim

# Install system dependencies (if any are needed later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirement file and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repository
COPY . .

# Default command runs a small demo training
CMD ["python", "app/main.py", "--episodes", "300", "--n-arms", "2"]
