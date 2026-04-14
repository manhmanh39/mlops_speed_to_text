import os
import shutil
import urllib.request
import zipfile

import torch
from fastapi import FastAPI, HTTPException, UploadFile
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from eval_wav2vec2 import (
    _load_local_weights,
    _load_model_and_processor,
    build_ctcdecoder,  # <-- IMPORT THÊM DECODER
    transcribe_wav2vec,
    vietnamese_number_converter,
)

app = FastAPI(title="Vietnamese Name ASR API", description="MLOps Final Project")

# --- KHỞI TẠO MONITORING ---
# Đặt ngay sau khi khởi tạo app để bắt đầu đo lường request
instrumentator = Instrumentator().instrument(app)

MODEL_ID = "nguyenvulebinh/wav2vec2-large-vi-vlsp2020"
MODEL_DIR = "/app/models/wav2vec2-finetuned"
LOCAL_WEIGHTS = "/app/models/wav2vec2-finetuned/model.safetensors"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = None
processor = None
decoder = None # <-- THÊM DECODER GLOBAL

@app.on_event("startup")
async def load_model():
    global model, processor, decoder
    try:
        model, processor = _load_model_and_processor(MODEL_ID, MODEL_DIR)
        _load_local_weights(model, LOCAL_WEIGHTS)
        model.to(DEVICE)
        model.eval()

        # --- NẠP KENLM DECODER ---
        lm_dir = "/app/models/lm"
        os.makedirs(lm_dir, exist_ok=True)
        lm_path = os.path.join(lm_dir, "vi_lm_4grams.bin")

        # Tải nếu chưa có (nên mount volume để không tải lại mỗi lần Pod restart)
        if not os.path.exists(lm_path):
            print("Downloading KenLM...")
            zip_path = os.path.join(lm_dir, "lm.zip")
            urllib.request.urlretrieve("https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h/resolve/main/vi_lm_4grams.bin.zip", zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(lm_dir)
            os.remove(zip_path)

        # Xây dựng vocab list từ tokenizer
        vocab_dict = processor.tokenizer.get_vocab()
        sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
        vocab_list = [item[0] for item in sorted_vocab]
        if len(vocab_list) > model.config.vocab_size:
            vocab_list = vocab_list[:model.config.vocab_size]
        if processor.tokenizer.pad_token in vocab_list:
            vocab_list[vocab_list.index(processor.tokenizer.pad_token)] = ""
        word_delim = processor.tokenizer.word_delimiter_token
        if word_delim in vocab_list:
            vocab_list[vocab_list.index(word_delim)] = " "

        decoder = build_ctcdecoder(labels=vocab_list, kenlm_model_path=lm_path, alpha=0.5, beta=1.5)
        print(f"--- Model and KenLM loaded successfully on {DEVICE} ---")
    except Exception as e:
        print(f"--- Error loading model: {e} ---")

@app.on_event("startup")
async def startup_event():
    await load_model()
    # Expose endpoint /metrics sau khi app khởi động
    instrumentator.expose(app)

class PredictionResponse(BaseModel):
    filename: str
    transcription: str
    post_processed: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Truyền thêm decoder vào hàm
        raw_text = transcribe_wav2vec(temp_path, processor, model, DEVICE, decoder)
        clean_text = vietnamese_number_converter(raw_text)

        return {
            "filename": file.filename,
            "transcription": raw_text,
            "post_processed": clean_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/health")
async def health_check():
    # Sửa lại: Phải check cả decoder (KenLM) vì nó chiếm 800MB,
    # nếu nó chưa load xong mà K8s đã gửi traffic vào thì sẽ bị lỗi timeout.
    if model is not None and decoder is not None:
        return {"status": "healthy", "device": str(DEVICE)}
    raise HTTPException(status_code=503, detail="Model or KenLM is still loading")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
