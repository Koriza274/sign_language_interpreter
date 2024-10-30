from fastapi import FastAPI
from tensorflow.keras.models import load_model
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# model_path = "/app/basic_models/asl_sign_language_model.keras"
model_path = "/app/basic_models/asl_sign_language_model_tf_2.18.keras"
model = load_model(model_path)

class PredictionRequest(BaseModel):
    data: list

@app.post("/predict")
async def predict(request: PredictionRequest):
    input_data = np.array([request.data])
    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}