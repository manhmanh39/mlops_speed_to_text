import asyncio
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Khởi tạo Mock App
app = FastAPI(title="Mock Vietnamese Name ASR API")

# Cấu hình CORS để Frontend không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    filename: str
    transcription: str
    post_processed: str

@app.get("/health")
async def health_check():
    # Trả về status "healthy" ngay lập tức để Streamlit hiện đèn xanh
    return {"status": "healthy", "device": "cpu (mock mode)"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile):
    """
    Hàm giả lập việc nhận file audio và dịch ra chữ.
    """
    # 1. Giả lập thời gian model đang suy nghĩ (Latency) mất khoảng 2.5 giây
    # Bạn có thể tăng giảm số này để test xem UI xoay xoay (spinner) có đẹp không
    await asyncio.sleep(2.5)
    
    # 2. Trả về kết quả Fake (Giả lập giống hệt cấu trúc JSON của app thật)
    return {
        "filename": file.filename,
        "transcription": "vũ thị yến một hai ba (fake inference)",
        "post_processed": "vũ thị yến 1 2 3 (fake inference)"
    }

if __name__ == "__main__":
    import uvicorn
    # Vẫn chạy ở port 8000 để đánh lừa frontend
    uvicorn.run(app, host="0.0.0.0", port=8000)