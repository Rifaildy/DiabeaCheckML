# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from tensorflow.keras.models import load_model
from pydantic import Field

# Definisikan path ke artefak model
MODEL_DIR = 'model_artifacts'
MODEL_PATH = os.path.join(MODEL_DIR, 'diabetes_mlp_model.h5')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

# Muat model dan scaler saat aplikasi dimulai
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model MLP dan Scaler berhasil dimuat.")
except FileNotFoundError as e:
    print(f"Error: Artefak model tidak ditemukan. Pastikan file tersedia. Detail: {e}")
    model = None
    scaler = None

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Definisikan input schema menggunakan Pydantic
class PredictionInput(BaseModel):
    Age: int = Field(..., ge=0, le=120, example=1)
    BMI: float = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BloodPressure: int = Field(..., ge=0)


# Endpoint prediksi
@app.post("/predict/")
async def predict_diabetes(data: PredictionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model atau scaler gagal dimuat saat startup.")

    # Susun input dalam array
    input_array = np.array([[
        data.Age,
        data.BMI,
        data.Glucose,
        data.Insulin,
        data.BloodPressure
    ]])

    # Scaling data
    input_scaled = scaler.transform(input_array)

    # Prediksi dengan model MLP
    probability = float(model.predict(input_scaled)[0][0])
    prediction = int(probability > 0.5)
    label = "Diabetes" if prediction == 1 else "Tidak Diabetes"

    # Kembalikan hasil dalam format JSON
    return {
        "prediction": label,
        "raw_output": prediction,
        "probability": round(probability, 4)
    }

# Endpoint root
@app.get("/")
async def read_root():
    return {"message": "DiabeaCheck API, Pendeteksi Resiko Diabetes menggunakan Machine Learning Berjalan dengan Baik."}
