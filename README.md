# Sign Language Recognition with Deep Learning

This project is the final project of the Data Science course by LeWagon, Batch 1705 (Diana, Robert, Jean-Michel, Gabriel & Boris). It aims to recognize sign language gestures, focusing on American Sign Language (ASL). The model uses machine learning libraries such as TensorFlow and OpenCV to detect and classify hand gestures and letters, which are then processed through a web interface powered by FastAPI and Streamlit.

This is the back-end repo. See the front-end repo [here](https://github.com/Koriza274/sign_language_interpreter_front-end). This readme file covers the whole project.

## Table of Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Running the Application](#running-the-application)
5. [Docker Deployment](#docker-deployment)
6. [License](#license)

---

### Features

- Web-based interface using Streamlit
- Scalable API using FastAPI
- Deployment ready with Docker support
- Includes a Jupyter notebook (letter_detection_images.ipynb) that trains the model and is adaptable for further improvements

---

### Requirements

Ensure you have the following installed:

Python 3.8 or higher

The libraries listed in requirements.txt

Additional system packages from packages.txt (for OpenCV and GUI support)

Alternatively, you can use the make install command to install all the necessary dependencies and create a virtual environment automatically. After running the command, remember to activate the virtual environment manually.

---

### Installation

#### 1. Clone the Repository
For back-end:
```bash
git clone https://github.com/Koriza274/sign_language_interpreter.git
cd sign_language_interpreter
```
For front-end:
```bash
git clone https://github.com/Koriza274/sign_language_interpreter_front-end.git
cd sign_language_interpreter
```

#### 2. Install Python Dependencies

Use the provided `requirements.txt` to install necessary Python packages.

```bash
make install
```


#### 3. System Dependencies

If you are on Linux, install additional packages required for OpenCV and other libraries.

```bash
sudo apt-get update
sudo apt-get install -y $(cat packages.txt)
```

---

### Running the Application

#### 1. API Service

Start the FastAPI backend service to manage gesture recognition.

```bash
uvicorn api:app --reload
```

#### 2. Frontend with Streamlit

Run the Streamlit app to launch the web-based interface.

```bash
streamlit run front_ASL_1.py
```

#### 3. Configuration

The application uses environment variables. You may configure these in a `.env` file. Refer to `params.py` for possible parameters that may be required.

---

### Docker Deployment

For easier deployment, you can use Docker to containerize the application.

1. Build the Docker image:

    ```bash
    docker build -t asl_recognition .
    ```

2. Run the Docker container:

    ```bash
    docker run -p 8000:8000 asl_recognition
    ```

This will expose the FastAPI backend on port 8000. To access the Streamlit interface, you might need to modify the `Dockerfile` to run both FastAPI and Streamlit, or run separate containers if required.

---

### License

This project is licensed under the MIT License. See the LICENSE file for details.

---
