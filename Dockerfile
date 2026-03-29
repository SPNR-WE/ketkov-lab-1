FROM python/python:3.12-slim
WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
CMD ["sleep", "infinity"]
