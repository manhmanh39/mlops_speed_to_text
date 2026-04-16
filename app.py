import os
import shutil
import urllib.request
import zipfile

import mlflow
import mlflow.pytorch
import torch
from fastapi import FastAPI, HTTPException, UploadFile
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from eval_wav2vec2 import (
    _load_local_weights,
    _load_model_and_processor,
    build_ctcdecoder,
    transcribe_wav2vec,
    vietnamese_number_converter,
)

app = FastAPI(
    title="Vietnamese Name ASR API",
    description="MLOps Final Project - backed by MLflow model registry",
)

# --- MONITORING ---
instrumentator = Instrumentator().instrument(app)

MODEL_ID = "nguyenvulebinh/wav2vec2-large-vi-vlsp2020"
MODEL_DIR = "/app/models/wav2vec2-finetuned"
LOCAL_WEIGHTS = "/app/models/wav2vec2-finetuned/model.safetensors"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MLflow settings (override via environment variables)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "wav2vec2-vietnamese-api")

model = None
processor = None
decoder = None


def _setup_mlflow():
    """Configure MLflow for inference logging."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


@app.on_event("startup")
async def load_model():
    global model, processor, decoder
    try:
        _setup_mlflow()

        model, processor = _load_model_and_processor(MODEL_ID, MODEL_DIR)
        _load_local_weights(model, LOCAL_WEIGHTS)
        model.to(DEVICE)
        model.eval()

        # --- LOG MODEL LOAD EVENT TO MLFLOW ---
        with mlflow.start_run(run_name="api-startup"):
            mlflow.set_tag("event", "model_loaded")
            mlflow.set_tag("device", DEVICE)
            mlflow.log_param("model_id", MODEL_ID)
            mlflow.log_param("model_dir", MODEL_DIR)
            mlflow.log_param("local_weights", LOCAL_WEIGHTS)

        # --- NẠP KENLM DECODER ---
        lm_dir = "/app/models/lm"
        os.makedirs(lm_dir, exist_ok=True)
        lm_path = os.path.join(lm_dir, "vi_lm_4grams.bin")

        if not os.path.exists(lm_path):
            print("Downloading KenLM...")
            zip_path = os.path.join(lm_dir, "lm.zip")
            urllib.request.urlretrieve(
                "https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h"
                "/resolve/main/vi_lm_4grams.bin.zip",
                zip_path,
            )
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(lm_dir)
            os.remove(zip_path)

        vocab_dict = processor.tokenizer.get_vocab()
        sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
        vocab_list = [item[0] for item in sorted_vocab]
        if len(vocab_list) > model.config.vocab_size:
            vocab_list = vocab_list[: model.config.vocab_size]
        if processor.tokenizer.pad_token in vocab_list:
            vocab_list[vocab_list.index(processor.tokenizer.pad_token)] = ""
        word_delim = processor.tokenizer.word_delimiter_token
        if word_delim in vocab_list:
            vocab_list[vocab_list.index(word_delim)] = " "

        decoder = build_ctcdecoder(
            labels=vocab_list,
            kenlm_model_path=lm_path,
            alpha=0.5,
            beta=1.5,
        )
        print(f"--- Model and KenLM loaded successfully on {DEVICE} ---")
    except Exception as e:
        print(f"--- Error loading model: {e} ---")


@app.on_event("startup")
async def startup_event():
    await load_model()
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
        raw_text = transcribe_wav2vec(temp_path, processor, model, DEVICE, decoder)
        clean_text = vietnamese_number_converter(raw_text)

        # Log inference to MLflow (async-safe: use a short-lived run)
        try:
            with mlflow.start_run(run_name="inference", nested=True):
                mlflow.set_tag("filename", file.filename)
                mlflow.log_param("raw_transcription", raw_text[:250])
                mlflow.log_param("post_processed", clean_text[:250])
        except Exception:
            pass  # Never block the API response for MLflow logging

        return {
            "filename": file.filename,
            "transcription": raw_text,
            "post_processed": clean_text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/health")
async def health_check():
    if model is not None and decoder is not None:
        return {"status": "healthy", "device": str(DEVICE)}
    raise HTTPException(status_code=503, detail="Model or KenLM is still loading")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
