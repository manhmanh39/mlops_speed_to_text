import os
import torch
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from eval_wav2vec2 import _load_model_and_processor, _load_local_weights, transcribe_wav2vec, vietnamese_number_converter

app = FastAPI(title="Vietnamese Name ASR API", description="MLOps Final Project - Group 7")

# Cấu hình đường dẫn (Khớp với Docker volumes của bạn)
MODEL_ID = "nguyenvulebinh/wav2vec2-large-vi-vlsp2020"
MODEL_DIR = "/app/models/wav2vec2-finetuned"
LOCAL_WEIGHTS = "/app/models/wav2vec2-finetuned/model.safetensors"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model 1 lần duy nhất khi khởi động server để tối ưu hiệu năng
model = None
processor = None

@app.on_event("startup")
async def load_model():
    global model, processor
    try:
        model, processor = _load_model_and_processor(MODEL_ID, MODEL_DIR)
        _load_local_weights(model, LOCAL_WEIGHTS)
        model.to(DEVICE)
        model.eval()
        print(f"--- Model loaded successfully on {DEVICE} ---")
    except Exception as e:
        print(f"--- Error loading model: {e} ---")

class PredictionResponse(BaseModel):
    filename: str
    transcription: str
    post_processed: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    # Lưu file tạm để xử lý
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 1. Transcribe (Dùng hàm core của bạn)
        raw_text = transcribe_wav2vec(temp_path, processor, model, DEVICE)
        
        # 2. Post-process (Dùng hàm xử lý số của bạn)
        clean_text = vietnamese_number_converter(raw_text)

        return {
            "filename": file.filename,
            "transcription": raw_text,
            "post_processed": clean_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL_ID}