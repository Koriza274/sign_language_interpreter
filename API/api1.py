from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
from PIL import Image
import os
import base64
from io import BytesIO
import cv2
from API.prediction1 import predict_asl_letter, extract_hand
#uvicorn API.api1:app --reload --port 8000
# FastAPI instance
app = FastAPI()


def image_to_base64(image: np.ndarray) -> str:
    if isinstance(image, np.ndarray):
        # If it's a numpy array, convert it to a PIL Image (RGB format)
        #image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = Image.fromarray(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# Root endpoint
@app.get("/")
def root():
    return {'greeting': "Ready for ASL letter prediction!!?"}

# Prediction endpoint
@app.post("/upload")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file and process it as an image
    try:
        #giving the directory for the uploaded file
        os.makedirs("API/uploads", exist_ok=True)
        file_location = f"API/uploads/{file.filename}"

        #creating the file:
        contents = file.file.read()
        with open(file_location, 'wb') as f:
            f.write(contents)

    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        try:
            #prediction using directory:
            label, confidence = predict_asl_letter(file_location)
        except Exception:
            raise HTTPException(status_code=500, detail='Prediction failed')
        # Read the image and convert to Base64 for the response
        try:
            image, hand = extract_hand(file_location)
            image_base64 = image_to_base64(image)
            hand_image_base64 = image_to_base64(hand)

        except Exception:
            raise HTTPException(status_code=500, detail="Failed to read the image")

        try:
            #removing the created file
            os.remove(file_location)
        except Exception:
            print(f"Error deleting file: {e}")

        file.file.close()
    return { "message": f"This is {str(label)}",
            "confidence":f"{confidence}%",
            "image": image_base64,
            "hand":hand_image_base64

        }
