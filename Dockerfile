FROM python:3.11-slim

RUN mkdir -p /disaster-tweet-classification

WORKDIR /disaster-tweet-classification

RUN pip install --no-cache-dir -U pip

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "api.py"]