import json
import logging
import os
import shutil
import uuid

from fastapi import FastAPI, HTTPException, UploadFile
import librosa
import mlflow
from prometheus_client import Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from scipy.stats import ks_2samp
import torch

from models.eval_baseline import (
    MODEL_ID_DEFAULT,
    _load_local_weights,
    _load_model_and_processor,
    transcribe_wav2vec,
    vietnamese_number_converter,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus Metrics
DRIFT_SCORE = Gauge(
    "asr_ks_drift_score", "K-S statistic for audio duration drift", ["feature_name"]
)
TRANSCRIPTION_LENGTH = Histogram(
    "asr_trans_len_words", "Transcription word count", buckets=[1, 3, 5, 10]
)

# Global variables cho Drift Detection
recent_durations = []
WINDOW_SIZE = 20
TRAIN_DURATIONS = []

app = FastAPI(
    title="Vietnamese Name ASR API",
    description="MLOps Final Project - E3: Finetune Only (No Preprocessing)",
)

# Instrument app cho Prometheus
instrumentator = Instrumentator().instrument(app)

# Configuration
MODEL_ID = os.environ.get("MODEL_ID", MODEL_ID_DEFAULT)
MODEL_DIR = "/app/models/wav2vec2-finetuned"
LOCAL_WEIGHTS = "/app/models/wav2vec2-finetuned/model.safetensors"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "wav2vec2-vietnamese-api")

model = None
processor = None


def _setup_mlflow():
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
    except Exception as e:
        logger.warning(f"Could not connect to MLflow: {e}")


@app.on_event("startup")
async def load_baseline_data():
    global TRAIN_DURATIONS
    try:
        baseline_path = "/app/models/baseline_distribution.json"
        with open(baseline_path, "r") as f:
            baseline = json.load(f)
            TRAIN_DURATIONS = baseline["durations"]
        logger.info(f">>> Baseline loaded: {len(TRAIN_DURATIONS)} samples found <<<")
    except Exception as e:
        logger.error(f"Failed to load baseline data: {e}")


@app.on_event("startup")
async def load_model_logic():
    global model, processor
    try:
        _setup_mlflow()
        logger.info(f">>> Loading model (E3 - Finetune Only) from: {MODEL_DIR} <<<")

        # 1. Load Model & Processor
        model, processor = _load_model_and_processor(MODEL_ID, MODEL_DIR)

        # 2. Load fine-tuned weights
        if _load_local_weights(model, LOCAL_WEIGHTS):
            logger.info(">>> Fine-tuned weights loaded successfully! <<<")
        else:
            logger.warning(">>> Local weights not found, using Hub weights! <<<")

        model.to(DEVICE)
        model.eval()

        try:
            with mlflow.start_run(run_name="api-startup-e3"):
                mlflow.set_tag("mode", "e3_finetune_only")
                mlflow.log_param("device", DEVICE)
        except Exception:
            pass

        logger.info(f"--- API READY ON {DEVICE.upper()} (E3 MODE) ---")
    except Exception as e:
        logger.error(f"--- STARTUP ERROR: {e} ---")


@app.on_event("startup")
async def setup_instrumentation():
    instrumentator.expose(app)


class PredictionResponse(BaseModel):
    filename: str
    transcription: str
    post_processed: str


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    request_id = str(uuid.uuid4())
    # Lưu file tạm để inference trực tiếp
    raw_path = f"raw_{request_id}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # --- DRIFT DETECTION ---
        try:
            # Vẫn tính duration trên file raw để monitor data drift
            curr_duration = librosa.get_duration(path=raw_path)
            recent_durations.append(curr_duration)
            if len(recent_durations) > WINDOW_SIZE:
                recent_durations.pop(0)

            if len(TRAIN_DURATIONS) > 0 and len(recent_durations) >= 5:
                stat, _ = ks_2samp(TRAIN_DURATIONS, recent_durations)
                DRIFT_SCORE.labels(feature_name="audio_duration").set(stat)
        except Exception as ex:
            logger.warning(f"Drift calculation failed: {ex}")

        # --- INFERENCE (E3: NO PREPROCESSING) ---
        # Truyền trực tiếp raw_path vào hàm transcribe thay vì qua _preprocess_wav
        raw_text = transcribe_wav2vec(raw_path, processor, model, DEVICE)

        # Metrics
        TRANSCRIPTION_LENGTH.observe(len(raw_text.split()))
        clean_text = vietnamese_number_converter(raw_text)

        # MLflow logging
        try:
            with mlflow.start_run(run_name="inference_e3", nested=True):
                mlflow.log_param("filename", file.filename)
                mlflow.log_metric("text_length", len(raw_text))
        except Exception:
            pass

        return {
            "filename": file.filename,
            "transcription": raw_text,
            "post_processed": clean_text,
        }
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        # Cleanup
        if os.path.exists(raw_path):
            os.remove(raw_path)


@app.get("/health")
async def health_check():
    if model is not None:
        return {"status": "healthy", "mode": "E3_Finetune_Only", "device": DEVICE}
    return {"status": "loading"}, 503

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
