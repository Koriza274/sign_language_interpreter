import streamlit as st
#import cv2 as cv
#import mediapipe as mp
import numpy as np
import time
from PIL import Image
import requests
#import copy
#import os
import io
import time
from io import BytesIO
import base64
from dotenv import load_dotenv
#from params import *
#from streamlit_webrtc import webrtc_streamer
from front_ASL_layout import display_image_columns


load_dotenv()
#api_url = st.secrets["API_URL"]
api_url = "https://my-api-app-3-564221756825.us-central1.run.app"


st.sidebar.title("Project Information")
st.sidebar.write("""
**Project Name**: Sign Language Recognition

**Goal**: Develop a model to recognize sign language using computer vision and machine learning techniques.

**Course**: LeWagon #1705 (Diana, Robert, Jean-Michel, Gabriel & Boris).

**Technologies Used**:
- Python for scripting and data processing
- Mediapipe and OpenCV for image processing and hand landmark detection
- TensorFlow for machine learning model development
- Streamlit for the web interface
""")
LAST_CLEAR_TIME = 0  # Keep track of the last time the cache was cleared
CACHE_CLEAR_INTERVAL = 60  # Interval in seconds (clear cache every 120 seconds)

def should_clear_cache():
    global LAST_CLEAR_TIME
    current_time = time.time()
    if current_time - LAST_CLEAR_TIME > CACHE_CLEAR_INTERVAL:
        LAST_CLEAR_TIME = current_time
        return True
    return False

def base64_image(base64_str):
            img_data = base64.b64decode(base64_str)
            img = Image.open(BytesIO(img_data))
            return img


def get_predictions_with_progress(uploaded_file):

    if should_clear_cache():
        st.cache_data.clear()
        #st.info("Cache cleared periodically.")


    # Initialize progress bar at 0%
    progress_bar = st.progress(0)

    # Prepare image data for API
    if isinstance(uploaded_file, np.ndarray):
        # Convert the numpy array directly to a PIL image
        image = Image.fromarray(uploaded_file)
    else:
        # If it's a file-like object, use Image.open
        image = Image.open(uploaded_file)

    # Get original dimensions
    original_width, original_height = image.size

    # Calculate new dimensions while keeping the aspect ratio
    if original_width > original_height:
        new_width = 512
        new_height = int((512 / original_width) * original_height)
    else:
        new_height = 512
        new_width = int((512 / original_height) * original_width)

    # Resize image with the new dimensions
    image = image.resize((new_width, new_height))

    # Convert the image to a bytes object to send over the API
    image_data = io.BytesIO()
    image.save(image_data, format='PNG')  # Save the image in PNG format
    image_data.seek(0)  # Rewind the buffer to the start

    # Set up the API URL and prepare files for the request
    url = f"{api_url}/upload"
    files = {'file': image_data}

    # Update progress bar to 10% after image preparation
    progress_bar.progress(10)

    # Send the request to the API and wait for the response
    response = requests.post(url, files=files)

    # Simulate loading progress in increments, to show the API processing
    for i in range(20, 100, 20):  # Increment progress in steps of 20%
        time.sleep(0.1)  # Short delay to simulate loading time
        progress_bar.progress(i)

    # Retrieve and return the prediction from the API response
    prediction = response.json()['message']
    confidence = response.json()['confidence']
    processed_image = base64_image(response.json()["image"])
    hand_region = base64_image(response.json()["hand"])
    # Set progress to 100% once loading is complete
    progress_bar.progress(100)

    # Clear the progress bar from the screen
    progress_bar.empty()

    return prediction, confidence, processed_image, hand_region

st.title("Show hands!")

st.write('Take a picture with the computer camera, or upload a file.')

# Initialize MediaPipe hands model
#mp_hands = mp.solutions.hands
#hands = mp_hands.Hands(static_image_mode=True,
                       #max_num_hands=2,
                       #min_detection_confidence=0.5,
                       #min_tracking_confidence=0.5)


camera_image = st.camera_input("Take a picture")

hand_region = None

# Display and save the captured image
if camera_image is not None:
    # Display the captured image
    prediction, confidence, processed_image, hand_region = get_predictions_with_progress(camera_image)


# File uploader to upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    prediction, confidence, processed_image, hand_region = get_predictions_with_progress(uploaded_file)

if hand_region is not None:

    display_image_columns(processed_image, hand_region, (prediction, confidence))
else:
    st.write("No hand detected in the image.")



def display_url():
    print(api_url)

if __name__  == '__main__':
    display_url()
