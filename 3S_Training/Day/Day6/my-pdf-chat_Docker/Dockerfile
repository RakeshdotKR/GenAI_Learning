# Use a lightweight Python base image.
FROM python:3.12-slim

# Set the application directory inside the container.
WORKDIR /app

# Copy dependency information first.
COPY requirements.txt .

# Install the Python dependencies.
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit application.
COPY app.py .

# Document the Streamlit port.
EXPOSE 8501

# Check whether the Streamlit server is responding.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
