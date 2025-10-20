# Use an official image that includes both Python + Node
FROM nikolaik/python-nodejs:python3.10-nodejs20

# Set working directory inside container
WORKDIR /app

# Copy everything from your repo into the image
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Node dependencies (for Puppeteer)
RUN cd scripts && npm install

# Install Chromium for Puppeteer (needed on Railway)
RUN apt-get update && apt-get install -y chromium

# Expose FastAPI port
EXPOSE 8000

# Start your FastAPI app
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
