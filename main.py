from fastapi import FastAPI, Request
import joblib
import numpy as np
from pydantic import BaseModel

# Load model saat aplikasi mulai
model = joblib.load('model.pkl')

# Definisikan app
app = FastAPI()

# Definisikan struktur input data
class InputData(BaseModel):
    nilai_ujian: float

# Endpoint prediksi
@app.post("/predict")
async def predict(data: InputData):
    nilai = np.array([[data.nilai_ujian]])
    pred = model.predict(nilai)[0]
    return {
        "nilai_ujian": data.nilai_ujian,
        "hasil": "Lulus" if pred == 1 else "Tidak Lulus"
    }
