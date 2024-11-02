import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import gc

url_api = "https://sign-interpreter-app-373962339093.europe-west1.run.app"

def send_to_api(image_bytes):
    """Send the raw image bytes directly to the API."""
    response = requests.post(url_api + '/predict', files={"file": ("hand_image.jpg", image_bytes, "image/jpeg")})
    if response.status_code == 200:
        return response.json()
    else:
        st.error("API Error: " + response.json().get("detail", "Unknown error"))

st.title("Show hands!")
st.write("Take a picture with the computer camera, or upload a file.")

camera_image = st.camera_input("Take a picture")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if camera_image or uploaded_file:
    # Load the image and ensure it's in RGB format
    source_image = camera_image if camera_image else uploaded_file
    img = Image.open(source_image).convert("RGB")

    # Resize image for display only (e.g., width=300)
    display_img = img.copy()
    display_img.thumbnail((300, 300))
    st.image(display_img, caption="Uploaded Image (Resized for Display)", use_column_width=False)

    # Convert original image to bytes for API
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    # Send the image bytes to the API
    prediction = send_to_api(img_bytes.getvalue())

    if prediction:
        st.write(f"Prediction: {prediction['prediction']}")
        st.write(f"Confidence: {prediction['confidence']:.2f}")
    else:
        st.write("No hand detected in the image.")

# Free up memory manually
gc.collect()
