from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import cv2
from package_folder.sign_interpreter import predict_asl_letter

# FastAPI instance
app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    return {'greeting': "Ready for ASL letter prediction!"}

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file and process it as an image
    try:
        image_data = await file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image format")

        # Call the prediction function from the package
        label, confidence = predict_asl_letter(image)

        if label is None:
            raise HTTPException(status_code=400, detail="No hand detected in the image.")

        # Return the prediction result
        return {
            "prediction": str(label),
            "confidence": float(confidence)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
