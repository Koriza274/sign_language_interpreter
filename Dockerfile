FROM python:3.10-slim


RUN apt-get update && \
    apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir opencv-python-headless

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY package_folder package_folder
COPY models models

CMD uvicorn package_folder.api_file:app --host 0.0.0.0 --port $PORT
