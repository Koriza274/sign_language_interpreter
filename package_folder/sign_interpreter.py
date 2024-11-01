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

def predict_asl_letter(image):
    """
    Predict the American Sign Language (ASL) letter from an image.
    """

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.2)
    # Load the model from the keras file
    model_path = os.path.join(ROOT_PATH, 'models', 'production_model', 'asl_sign_language_model_large_normalized.keras')
    label_path = os.path.join(ROOT_PATH, 'models', 'production_model', 'label_classes.npy')

    model = tf.keras.models.load_model(model_path)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(label_path)

    # Check if the image is in BGR by comparing with a converted version
    if not np.array_equal(image[0, 0], cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[0, 0]):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
    #image = cv2.imread("../raw_data/test_set_pics/X/test_X_2.jpg")
    image = cv2.imread("../raw_data/asl_alphabet_dataset/asl_alphabet_test/space_test.jpg")
    label, confidence = predict_asl_letter(image)
    if label:
        print(f"Predicted ASL Letter: {label} with {confidence:.2f}% confidence")
    else:
        print("No hand detected in the image.")
