import streamlit as st
import cv2 as cv
import mediapipe as mp
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import gc

#url_api = "https://dmapi-564221756825.europe-west1.run.app"
url_api = "https://sign-interpreter-app-373962339093.europe-west1.run.app"

# Maybe we can try to initialize MediaPipe Hands module outside of functions to avoid repeated instantiation
@st.cache_resource
def load_hand_model():
    return mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

hands = load_hand_model()

def calc_bounding_rect(image, landmarks):
    padding = 40
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.array([(min(int(landmark.x * image_width), image_width - 1),
                                min(int(landmark.y * image_height), image_height - 1))
                                for landmark in landmarks.landmark])
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x-padding, y-padding, x + w + padding, y + h + padding]

def draw_bounding_rect(image, brect):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def resize_image(image, max_dim=640):
    h, w = image.shape[:2]
    scale = max_dim / max(h, w)
    return cv.resize(image, (int(w * scale), int(h * scale)))

def extract_hand(source_image):
    image = np.array(Image.open(source_image).convert("RGB"))
    image = resize_image(image)  # Resize for memory optimization
    results = hands.process(image)

    if results.multi_hand_landmarks:
        debug_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            x_min, y_min, x_max, y_max = brect
            hand_region = image[max(0, y_min):y_max, max(0, x_min):x_max]
            debug_image = draw_bounding_rect(debug_image, brect)
            return debug_image, hand_region
    return None, None

def send_to_api(hand_region):
    is_success, buffer = cv.imencode(".jpg", hand_region)
    if is_success:
        img_bytes = BytesIO(buffer)
        response = requests.post(url_api +'/predict',
                                 files={"file": ("hand_image.jpg", img_bytes, "image/jpeg")})

        if response.status_code == 200:
            return response.json()
        else:
            st.error("API Error: " + response.json().get("detail", "Unknown error"))

st.title("Show hands!")
st.write("Take a picture with the computer camera, or upload a file.")

camera_image = st.camera_input("Take a picture")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

hand_region = None
if camera_image or uploaded_file:
    source_image = camera_image if camera_image else uploaded_file
    processed_image, hand_region = extract_hand(source_image)

    if hand_region is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(processed_image, caption="Original", use_column_width=True)
        with col2:
            st.image(hand_region, caption="Hand region")

        prediction = send_to_api(hand_region)
        if prediction:
            st.write(f"Prediction: {prediction['prediction']}")
            st.write(f"Confidence: {prediction['confidence']:.2f}")
    else:
        st.write("No hand detected in the image.")

# Free up memory manually
gc.collect()
