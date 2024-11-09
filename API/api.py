from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
from PIL import Image
import os
import io
import cv2
from API.prediction import predict_asl_letter

# FastAPI instance
app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    return {'greeting': "Ready for ASL letter prediction!"}

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
        finally:
            try:
                #removing the created file
                os.remove(file_location)
            except Exception:
                print(f"Error deleting file: {e}")
            file.file.close()
    return {
            "message": f"This is {str(label)}","confidence":f"{confidence}%"
        }
