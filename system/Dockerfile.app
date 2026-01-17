FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1         PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends         bash         curl         libgomp1       && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY system/requirements-system.txt /app/system/requirements-system.txt
RUN pip install --no-cache-dir -r /app/requirements.txt -r /app/system/requirements-system.txt

COPY system /app/system

EXPOSE 8000
