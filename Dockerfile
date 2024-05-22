FROM python:3.11-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY app.py .

ENV FLASK_APP=app

response = requests.get("http://127.0.0.1:5000/predict_get?sl=6.3&pl=2.6")

subprocess.call(["flask", "run", "--host", "0.0.0.0", "--port", "8000"])

# EXPOSE 8000
# CMD ["flask", "run", "--host", "0.0.0.0", "--port", "8000"]

