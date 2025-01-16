# syntax=docker/dockerfile:1.2
FROM python:3.11.9-slim

WORKDIR /app

# Install build dependencies using apt-get
RUN apt-get update && apt-get install -y \
    make \
    gcc \
    musl-dev \
    libffi-dev \
    libpq-dev \
    bash \
    g++ \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Copy the application code
COPY . /app/

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that your FastAPI app will run on
EXPOSE 8000

# Set the environment variable for the FastAPI app
ENV PYTHONPATH=/app

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8000"]
