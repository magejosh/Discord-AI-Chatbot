FROM python:bullseye

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install aiohttp==3.9.0b0
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
