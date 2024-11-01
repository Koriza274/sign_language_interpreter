import os
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))

def adjust_brightness_contrast(image, brightness=40, contrast=1.0):
    """
    Adjust the brightness and contrast of an image.
    """
    img = image.astype(np.float32)
    img = img * contrast + brightness
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# Function to normalize landmarks relative to bounding box center and scale
def normalize_landmarks(landmarks):
    # Extract x and y coordinates
    x_coords = landmarks[::3]
    y_coords = landmarks[1::3]

    # Compute bounding box
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    # Center the landmarks around (0, 0)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    centered_landmarks = []
    for i in range(0, len(landmarks), 3):
        centered_landmarks.append(landmarks[i] - center_x)  # x - center_x
        centered_landmarks.append(landmarks[i + 1] - center_y)  # y - center_y
        centered_landmarks.append(landmarks[i + 2])  # z stays the same

    # Scale by diagonal of the bounding box to normalize for aspect ratio
    bbox_diagonal = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
    normalized_landmarks = [coord / bbox_diagonal for coord in centered_landmarks]

    return normalized_landmarks

def predict_asl_letter(image):
    """
    Predict the American Sign Language (ASL) letter from an image.
    """

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.1)
    # Load the model from the keras file
    model_path = os.path.join(ROOT_PATH, 'models', 'production_model', 'asl_sign_language_model.keras')
    label_path = os.path.join(ROOT_PATH, 'models', 'production_model', 'label_classes_large.npy')

    model = tf.keras.models.load_model(model_path)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(label_path)

    # Check if the image is in BGR by comparing with a converted version
    if not np.array_equal(image[0, 0], cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[0, 0]):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = adjust_brightness_contrast(image, brightness=20, contrast=1.0)

    results = hands.process(image)
    if results.multi_hand_landmarks:
        landmarks = []
        wrist = results.multi_hand_landmarks[0].landmark[0]
        for lm in results.multi_hand_landmarks[0].landmark:
            # Normalize each landmark relative to the wrist
            normalized_x = lm.x - wrist.x
            normalized_y = lm.y - wrist.y
            normalized_z = lm.z - wrist.z
            landmarks.extend([normalized_x, normalized_y, normalized_z])
            #landmarks.extend([lm.x, lm.y, lm.z])
        # Normalize landmarks
        #normalized_landmarks = normalize_landmarks(landmarks)

        # Normalize and reshape the landmarks for model input
        landmarks = np.array(landmarks).reshape(1, -1, 1)
        prediction = model.predict(landmarks)
        predicted_label_index = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
        confidence = prediction[0][predicted_label_index] * 100

        return predicted_label, confidence
    else:
        return None, 0.0


if __name__ == '__main__':
    #image = cv2.imread("../raw_data/test_set_pics/C/test_C_3.jpg")
    image = cv2.imread("../raw_data/asl_alphabet_dataset/asl_alphabet_test/C_test.jpg")
    label, confidence = predict_asl_letter(image)
    if label:
        print(f"Predicted ASL Letter: {label} with {confidence:.2f}% confidence")
    else:
        print("No hand detected in the image.")
