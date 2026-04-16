import logging
import os
import shutil
import uuid

import mlflow
import torch
from fastapi import FastAPI, HTTPException, UploadFile
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

# Import logic từ file eval - Đảm bảo transcribe_wav2vec dùng Greedy Search
from eval_wav2vec2 import (
    MODEL_ID_DEFAULT,
    _load_local_weights,
    _load_model_and_processor,
    _preprocess_wav,
    transcribe_wav2vec,
    vietnamese_number_converter,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vietnamese Name ASR API",
    description="MLOps Final Project - Greedy Search (No KenLM)",
)

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
        logger.warning(f"Không thể kết nối MLflow: {e}")


@app.on_event("startup")
async def load_model_logic():
    global model, processor
    try:
        _setup_mlflow()
        logger.info(f">>> Đang nạp model (Greedy) từ: {MODEL_DIR} <<<")

        # 1. Nạp Model & Processor theo logic eval
        model, processor = _load_model_and_processor(MODEL_ID, MODEL_DIR)

        # 2. Nạp trọng số finetuned
        if _load_local_weights(model, LOCAL_WEIGHTS):
            logger.info(">>> Đã nạp tạ finetuned thành công! <<<")
        else:
            logger.warning(">>> Không tìm thấy tạ local, dùng trọng số Hub! <<<")

        model.to(DEVICE)
        model.eval()

        # Log sự kiện startup lên MLflow
        try:
            with mlflow.start_run(run_name="api-startup"):
                mlflow.set_tag("mode", "greedy_inference")
                mlflow.log_param("device", DEVICE)
        except Exception:
            pass

        logger.info(f"--- API SẴN SÀNG TRÊN {DEVICE} ---")
    except Exception as e:
        logger.error(f"--- LỖI KHỞI ĐỘNG: {e} ---")


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
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .wav")

    request_id = str(uuid.uuid4())
    raw_path = f"raw_{request_id}_{file.filename}"
    norm_path = f"norm_{request_id}_{file.filename}"

    with open(raw_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 1. Tiền xử lý (Denoise + Normalize)
        final_audio = _preprocess_wav(raw_path, norm_path)

        # 2. Nhận diện (Sử dụng hàm từ eval_wav2vec2)
        raw_text = transcribe_wav2vec(final_audio, processor, model, DEVICE)

        # 3. Chuẩn hóa số tiếng Việt
        clean_text = vietnamese_number_converter(raw_text)

        # 4. Log kết quả inference
        try:
            with mlflow.start_run(run_name="inference_call", nested=True):
                mlflow.log_param("file", file.filename)
                mlflow.log_metric("text_len", len(raw_text))
        except Exception:
            pass

        return {
            "filename": file.filename,
            "transcription": raw_text,
            "post_processed": clean_text,
        }
    except Exception as e:
        logger.exception("Inference thất bại")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        for p in [raw_path, norm_path]:
            if os.path.exists(p):
                os.remove(p)


@app.get("/health")
async def health_check():
    if model is not None:
        return {"status": "healthy", "device": DEVICE}
    return {"status": "loading"}, 503


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
