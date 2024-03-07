from fastapi import FastAPI, Form, UploadFile
from typing import Annotated
from tensorflow import keras
import io
from PIL import Image
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = keras.models.load_model('cp3')

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/uploadfile")
async def create_upload_file(file: UploadFile):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents))
    image = image.convert('RGB')  # Chuyển đổi thành ảnh RGB
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image)
    prediction = np.argmax(prediction)
    
    if prediction == 0:
        return {
            "predict": "Alaska"
        }
    elif prediction ==  1:
        return {
            "predict": "Corgi"
        }
    elif prediction == 2:
        return {
            "predict": "Poodle"
        }
    elif prediction == 3:
        return {
            "predict": "Shiba"
        }
    else:
        return {
            "predict": "Không thuộc 4 lớp"
        }