FROM python:3.11-bullseye

WORKDIR /app

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-warn-script-location --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --upgrade pip
RUN pip install aiohttp==3.9.0b0
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
