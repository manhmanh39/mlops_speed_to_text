import streamlit as st
import requests
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
import os

# --- CẤU HÌNH HỆ THỐNG ---
# Tự động lấy URL từ biến môi trường nếu chạy trên K8s, mặc định là localhost
API_URL = os.getenv("API_URL", "http://localhost:8000")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:30200")

st.set_page_config(
    page_title="Vietnamese Name ASR | NEU Group 7",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THIẾT KẾ GIAO DIỆN (CSS CUSTOM) ---
# Tạo phong cách Dashboard hiện đại, bo góc thẻ và font chữ Nunito
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700&display=swap');

    /* Font chữ tổng thể và Cải thiện độ tương phản (Fix lỗi chữ xám mờ) */
    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif !important;
        color: #f8fafc !important; /* Màu chữ trắng sáng */
    }
    
    /* Đảm bảo các thẻ tiêu đề hiển thị màu trắng rõ ràng */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Màu nền tối chuyên nghiệp cho Dashboard */
    .stApp {
        background-color: #1a1c23;
    }
    
    /* Tùy chỉnh Sidebar */
    [data-testid="stSidebar"] {
        background-color: #12141a;
        border-right: 1px solid #2d323e;
    }
    
    /* Tạo hiệu ứng thẻ (Card) cho các khu vực nội dung */
    div.stBlock {
        background-color: #24262d;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2d323e;
        margin-bottom: 20px;
    }

    /* Tùy chỉnh các nút bấm */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #3b82f6;
        color: white;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }

    /* Màu sắc hiển thị kết quả Transcription */
    code {
        color: #10b981 !important; /* Xanh lá hiện đại */
        background-color: #12141a !important;
        font-size: 1.1em !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR (THANH ĐIỀU HƯỚNG) ---
with st.sidebar:
    # Chèn Logo NEU - Thay thế use_container_width bằng width="stretch" theo API mới của Streamlit
    st.image("Logo-NEU.png", width="stretch")
    st.markdown("<h2 style='text-align: center; color: white; margin-top: -10px;'>MLOps Project</h2>", unsafe_allow_html=True)
    
    # Đã làm sáng màu chữ phụ đề từ #8898aa sang #cbd5e1 để dễ đọc hơn trên nền đen
    st.markdown("<p style='text-align: center; color: #cbd5e1;'>NEU - Group 7</p>", unsafe_allow_html=True)
    st.divider()

    # Quản lý trạng thái hệ thống (Health Check)
    st.subheader("📊 System Status")
    try:
        health_res = requests.get(f"{API_URL}/health", timeout=3)
        if health_res.status_code == 200:
            st.success("🟢 API: Online")
            data = health_res.json()
            st.info(f"**Device:** {data.get('device', 'CPU').upper()}")
        elif health_res.status_code == 503:
            st.warning("⏳ API: Initializing Model...")
        else:
            st.error("🔴 API: Offline")
    except:
        st.error("🔴 API: Connection Failed")

    st.divider()
    
    # Liên kết trực tiếp tới Grafana Monitoring
    st.subheader("📈 Monitoring")
    st.markdown("Truy cập Dashboard giám sát thời gian thực.")
    st.sidebar.link_button("👉 Open Grafana Dashboard", GRAFANA_URL, width="stretch")
    
    st.divider()
    st.caption("Pipeline: Wav2Vec2 + KenLM + Prometheus")

# --- NỘI DUNG CHÍNH (MAIN DASHBOARD) ---
st.title("🎙️ Vietnamese Speech-to-Text Dashboard")
st.markdown("Hệ thống nhận dạng tên tiếng Việt tích hợp Language Model và Giám sát MLOps.")

# 1. Khu vực Giả lập Traffic (Dành cho Demo Grafana)
with st.expander("🚀 Simulate Traffic (Load Testing Demo)", expanded=False):
    st.markdown("Gửi các request ảo để quan sát sự thay đổi của biểu đồ trên Grafana.")
    num_req = st.slider("Số lượng request:", 1, 100, 20)
    if st.button("Fire Requests"):
        progress = st.progress(0)
        for i in range(num_req):
            try:
                requests.get(f"{API_URL}/health", timeout=1)
            except: pass
            progress.progress((i + 1) / num_req)
        st.success(f"Đã gửi {num_req} requests thành công!")

# 2. Khu vực Input (Tải file âm thanh)
st.header("1. Input Signal")
uploaded_file = st.file_uploader("Chọn file âm thanh (.wav)", type=["wav"])

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format="audio/wav")
    
    # 3. Khu vực Phân tích tín hiệu (Signal Analysis)
    st.header("2. Signal Analysis")
    with st.spinner("Đang trích xuất đặc trưng âm học..."):
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        col_w, col_s = st.columns(2)
        
        with col_w:
            st.subheader("Waveform (Biên độ)")
            fig_w, ax_w = plt.subplots(figsize=(10, 4), facecolor='#1a1c23')
            librosa.display.waveshow(y, sr=sr, ax=ax_w, color="#3b82f6")
            ax_w.set_facecolor('#1a1c23')
            ax_w.tick_params(colors='white')
            st.pyplot(fig_w)
            
        with col_s:
            st.subheader("Mel-Spectrogram (Phổ)")
            fig_s, ax_s = plt.subplots(figsize=(10, 4), facecolor='#1a1c23')
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax_s, cmap='magma')
            ax_s.set_facecolor('#1a1c23')
            ax_s.tick_params(colors='white')
            st.pyplot(fig_s)

    # 4. Khu vực Inference (Kết quả từ Model)
    st.header("3. AI Inference Pipeline")
    if st.button("🚀 Run Transcription"):
        with st.spinner("Mô hình đang xử lý..."):
            start = time.time()
            files = {"file": (uploaded_file.name, audio_bytes, "audio/wav")}
            try:
                response = requests.post(f"{API_URL}/predict", files=files)
                latency = time.time() - start
                
                if response.status_code == 200:
                    res = response.json()
                    st.success(f"Hoàn thành trong {latency:.2f} giây")
                    
                    res_c1, res_c2 = st.columns(2)
                    with res_c1:
                        st.info("🧠 **Raw ASR (Wav2Vec2 + KenLM)**")
                        st.code(res.get("transcription"), language="text")
                    with res_c2:
                        st.success("⚙️ **Post-Processed (Normalized)**")
                        st.code(res.get("post_processed"), language="text")
                else:
                    st.error(f"Lỗi API: {response.text}")
            except:
                st.error("Không thể kết nối tới Backend API.")