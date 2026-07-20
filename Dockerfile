# Use an official lightweight Python runtime as parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in container
WORKDIR /app

# Install minimal OS dependencies required by OpenCV (even headless OpenCV requires glib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt /app/

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-privileged system group and user
RUN groupadd -g 1000 appuser && useradd -u 1000 -g appuser -m -s /bin/bash appuser

# Create directories for persistent face database and dataset and grant access
RUN mkdir -p /app/data /app/dataset && chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Copy python core modules, static assets, and tests with correct ownership
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser static/ /app/static/
COPY --chown=appuser:appuser tests/ /app/tests/

# Expose the port Flask runs on
EXPOSE 5000

# Run the Flask application
CMD ["python", "src/app.py"]
