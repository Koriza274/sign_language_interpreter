import streamlit as st
import cv2 as cv
import mediapipe as mp
import numpy as np
import copy
import os
#from streamlit_webrtc import webrtc_streamer

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

            return image, hand_region

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

    col1, col2 = st.columns(2)

    with col1:
        st.image(processed_image, caption="Original", use_column_width=True)

    with col2:
        st.image(hand_region, caption="Hand region")

else:
    st.write("No hand detected in the image.")
