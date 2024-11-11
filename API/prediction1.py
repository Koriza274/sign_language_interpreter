import os
import numpy as np
import cv2
import copy
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

def predict_asl_letter(image_in):
    """
    Predict the American Sign Language (ASL) letter from an image.
    """
    img = cv2.imread(image_in)
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
    # Load the model from the keras file

    model_path = os.path.join(ROOT_PATH,"API", 'models','asl_new_model.keras')
    label_path = os.path.join(ROOT_PATH,"API", 'models', 'labels_v_large.npy')

    model = tf.keras.models.load_model(model_path)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(label_path)

    # Check if the image is in BGR by comparing with a converted version
    #if not np.array_equal(image[0, 0], cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[0, 0]):
       # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    brightness_values = range(-60, 60, 5)  # e.g., from -50 to 50 in steps of 20
    #contrast_values = [1.0, 0.5, 0.75, 1.25, 1.5]
    landmarks_found = False
    for brightness in brightness_values:
        image_adjusted = adjust_brightness_contrast(image, brightness=brightness, contrast=1.0)
        results = hands.process(image_adjusted)
        if results.multi_hand_landmarks:
            landmarks_found = True
            break

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


def calc_bounding_rect(image, landmarks):

    padding = 20

    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x-padding, y-padding, x + w + (padding), y + h + (padding)]

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

def extract_hand(path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    image = cv2.imread(path)
    debug_image = copy.deepcopy(image)

    image = cv2.flip(image, 1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    mp_drawing.draw_landmarks(
            image,
            results.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS
        )

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






if __name__ == '__main__':
    #image = cv2.imread("../raw_data/test_set_pics/C/test_C_3.jpg")
    #image = cv2.imread("../raw_data/asl_alphabet_dataset/asl_alphabet_test/C_test.jpg")
    image = cv2.imread("../raw_data/L_cropped.jpg")
    label, confidence = predict_asl_letter(image)
    if label:
        print(f"Predicted ASL Letter: {label} with {confidence:.2f}% confidence")
    else:
        print("No hand detected in the image.")
