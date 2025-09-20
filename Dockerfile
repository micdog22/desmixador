FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg build-essential git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY README.md LICENSE .
RUN mkdir -p inputs outputs reports
EXPOSE 8000 8501
CMD ["python", "src/autostems.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
