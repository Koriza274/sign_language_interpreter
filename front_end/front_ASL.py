import streamlit as st
import cv2 as cv
import mediapipe as mp
import numpy as np
import time
from PIL import Image
import requests
import copy
import os
import io
import time
from dotenv import load_dotenv
from params import *
from streamlit_webrtc import webrtc_streamer
from front_ASL_layout import display_image_columns


load_dotenv()
api_url = st.secrets["API_URL"]

#API_URL = "https://dmapi-564221756825.europe-west1.run.app"

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
    predictions = response.json()['message']
    confidence = response.json()['confidence']

    # Set progress to 100% once loading is complete
    progress_bar.progress(100)

    # Clear the progress bar from the screen
    progress_bar.empty()

    return predictions, confidence

def calc_bounding_rect(image, landmarks):

    padding = 20

    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x-padding, y-padding, x + w + (padding), y + h + (padding)]

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

def extract_hand(source_image):

    # Save the file to a temporary path
    temp_path = os.path.join("temp_dir", source_image.name)
    os.makedirs("temp_dir", exist_ok=True)  # Create directory if it doesn't exist

    # Write the uploaded file to the specified location
    with open(temp_path, "wb") as f:
        f.write(source_image.getbuffer())

    image = cv.imread(temp_path)
    debug_image = copy.deepcopy(image)

    image = cv.flip(image, 1)

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    if results.multi_hand_landmarks is not None:

        i = 0

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                    results.multi_handedness):
            # Bounding box calculation
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            x_min, y_min, x_max, y_max = brect

            # Crop the hand region from the image based on bounding rectangle
            #+1 pixel to remove the outer border of the picture
            hand_region = image[y_min+1:y_max, x_min+1:x_max]

            # Drawing part
            image = draw_bounding_rect(True, image, brect)

            #delete debug_image
            os.remove(temp_path)

            return image, hand_region

    #delete debug_image
    os.remove(temp_path)

    return None, None

st.title("Show hands!")

st.write('Take a picture with the computer camera, or upload a file.')

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)


camera_image = st.camera_input("Take a picture")

hand_region = None

# Display and save the captured image
if camera_image is not None:
    # Display the captured image
    processed_image, hand_region = extract_hand(camera_image)

# File uploader to upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    processed_image, hand_region = extract_hand(uploaded_file)

if hand_region is not None:
    prediction, confidence = get_predictions_with_progress(hand_region)
    display_image_columns(processed_image, hand_region, (prediction, confidence))
else:
    st.write("No hand detected in the image.")



def display_url():
    print(api_url)

if __name__  == '__main__':
    display_url()
