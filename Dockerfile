FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "src/api/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
