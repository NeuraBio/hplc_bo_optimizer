
FROM python:3.11-slim

RUN pip install poetry

WORKDIR /app
COPY . /app

RUN poetry install

# Replace this with your web entrypoint later
CMD ["poetry", "run", "streamlit", "run", "app.py"]
