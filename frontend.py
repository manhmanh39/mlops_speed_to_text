import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Wav2Vec2 MLOps Dashboard", page_icon="🎙️", layout="wide")

# 1. BỘ CHỌN GIAO DIỆN
# --- THÊM LOGO VÀ TÊN NHÓM ---
st.sidebar.image("Logo-NEU.png", width=150) 

# In đậm tên nhóm
st.sidebar.markdown("### Group 7 - DSEB65B")
st.sidebar.markdown("###### Nguyễn Thị Mai Anh")
st.sidebar.markdown("###### Phạm Thị Ngọc Ánh")
st.sidebar.markdown("###### Nguyễn Thanh Mơ")
st.sidebar.markdown("###### Nguyễn Khánh Huyền")
st.sidebar.markdown("###### Nguyễn Thị Hương Giang")
st.sidebar.markdown("###### Lê Lan Hương")
st.sidebar.markdown("---")
theme = st.sidebar.radio("🎨 Chọn giao diện", ["Tối (Dark)", "Sáng (Light)"])

# 2. THIẾT LẬP BẢNG MÀU ĐỘNG
if theme == "Tối (Dark)":
    colors = {
        "main_bg": "#111217",
        "card_bg": "#1e1e24",
        "text_main": "#ffffff",
        "text_sub": "#d1d5db",
        "border": "#444444",
        "btn_bg": "#2b2c36",
        "btn_text": "#ffffff", 
        "dropzone_bg": "#1e1e24"
    }
else:
    colors = {
        "main_bg": "#ffffff",
        "card_bg": "#f8f9fa",
        "text_main": "#111217",
        "text_sub": "#4b4b4b",
        "border": "#cccccc",
        "btn_bg": "#ffffff",
        "btn_text": "#111217", 
        "dropzone_bg": "#f0f2f6"
    }

# Địa chỉ các service trong mạng Docker
PROMETHEUS_URL = "http://prometheus:9090/api/v1/query"
PROMETHEUS_RANGE_URL = "http://prometheus:9090/api/v1/query_range"
#API_URL = "http://wav2vec2-api:8000/predict"
API_URL = "http://localhost:8000/predict"

# 3. CSS ÁP DỤNG MÀU ĐỘNG & SỬA LỖI GIAO DIỆN
st.markdown(
    f"""
    <style>
    /* Nền tổng thể và màu chữ chính */
    .stApp {{ 
        background-color: {colors['main_bg']}; 
        color: {colors['text_main']}; 
    }}
    
    /* Đồng bộ màu Sidebar (giúp sidebar không bị trắng khi ở chế độ Dark) */
    [data-testid="stSidebar"] {{
        background-color: {colors['card_bg']} !important;
    }}
    
    /* Chữ nhỏ, label, caption (Không áp dụng cho các hộp thông báo st.success/st.info) */
    div:not([data-testid="stAlert"]) > [data-testid="stMarkdownContainer"] p, 
    [data-testid="stCaptionContainer"] p, 
    label {{
        color: {colors['text_sub']} !important; 
    }}
    
    /* Trả lại màu chữ mặc định cho các hộp thông báo st.success, st.info để dễ đọc */
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {{
        color: inherit !important;
    }}
    
    /* Tên file sau khi tải lên */
    [data-testid="stUploadedFile"] p,
    [data-testid="stUploadedFile"] span {{
        color: {colors['text_main']} !important;
    }}

    /* --- NÚT BẤM CHUNG (Refresh, Run Inference...) --- */
    div[data-testid="stButton"] button {{
        background-color: {colors['btn_bg']} !important;
        border: 1px solid {colors['border']} !important;
    }}
    div[data-testid="stButton"] button,
    div[data-testid="stButton"] button * {{
        color: {colors['btn_text']} !important;
    }}

    /* --- SỬA LỖI VÙNG KÉO THẢ VÀ NÚT UPLOAD --- */
    [data-testid="stFileUploadDropzone"] {{
        background-color: {colors['dropzone_bg']} !important;
        border: 1px dashed {colors['border']} !important;
    }}
    
    /* Chữ "Drag and drop..." và giới hạn dung lượng */
    [data-testid="stFileUploadDropzone"] > div > div > span,
    [data-testid="stFileUploadDropzone"] > div > div > small {{
        color: {colors['text_sub']} !important;
    }}

    /* KHUNG NGOÀI NÚT UPLOAD: LUÔN CỐ ĐỊNH NỀN SÁNG */
    div[data-testid="stFileUploader"] button {{
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
        border-radius: 5px !important;
    }}
    
    /* ÉP TẤT CẢ CÁC THẺ CON BÊN TRONG NÚT LÀ MÀU ĐEN */
    div[data-testid="stFileUploader"] button p,
    div[data-testid="stFileUploader"] button span,
    div[data-testid="stFileUploader"] button div,
    div[data-testid="stFileUploader"] button * {{
        color: #000000 !important;
        font-weight: 600 !important;
    }}

    /* --- HIỆU ỨNG HOVER CHO NÚT UPLOAD --- */
    div[data-testid="stFileUploader"] button:hover {{
        background-color: #f0f2f6 !important;
        border-color: #ff4b4b !important; /* Đổi viền sang đỏ khi hover cho đồng bộ */
    }}
    
    div[data-testid="stFileUploader"] button:hover p,
    div[data-testid="stFileUploader"] button:hover span,
    div[data-testid="stFileUploader"] button:hover div,
    div[data-testid="stFileUploader"] button:hover * {{
        color: #ff4b4b !important; 
    }}

    /* Hover cho nút chung */
    div[data-testid="stButton"] button:hover {{
        background-color: #ff4b4b !important;
        border-color: #ff4b4b !important;
    }}
    div[data-testid="stButton"] button:hover,
    div[data-testid="stButton"] button:hover * {{
        color: #ffffff !important;
    }}

    /* --- Box chứa Metrics Dashboard --- */
    div[data-testid="metric-container"] {{
        background-color: {colors['card_bg']}; 
        border: 1px solid {colors['border']}; 
        padding: 15px; 
        border-radius: 5px;
    }}
    div[data-testid="stMetricLabel"] > div {{
        color: {colors['text_sub']} !important; 
    }}
    div[data-testid="stMetricValue"] > div {{
        color: {colors['text_main']} !important; 
    }}

    /* Màu chữ cho Tabs */
    .stTabs [data-baseweb="tab-list"] button {{
        color: {colors['text_sub']};
    }}
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        color: #ff4b4b;
    }}
    
    /* Màu chữ của bộ chọn Radio trong Sidebar */
    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] [data-testid="stRadio"] p,
    [data-testid="stSidebar"] [data-testid="stRadio"] div {{
        color: {colors['text_main']} !important;
    }}
    </style>
""",
    unsafe_allow_html=True,
)

# --- Helper Functions for Prometheus ---
def get_prom_value(query):
    try:
        response = requests.get(PROMETHEUS_URL, params={"query": query}, timeout=2)
        results = response.json()["data"]["result"]
        return float(results[0]["value"][1]) if results else 0.0
    except Exception:
        return 0.0


def get_prom_series(query, minutes=5):
    try:
        end = time.time()
        start = end - (minutes * 60)
        response = requests.get(
            PROMETHEUS_RANGE_URL,
            params={"query": query, "start": start, "end": end, "step": "10s"},
            timeout=2,
        )
        values = response.json()["data"]["result"][0]["values"]
        df = pd.DataFrame(values, columns=["timestamp", "value"])
        df["value"] = df["value"].astype(float)
        return df
    except Exception:
        return pd.DataFrame(columns=["value"])


# --- UI Header ---
st.title("🎙️ MLOps: Vietnamese Speech-to-Text")
tab_inference, tab_dashboard = st.tabs(
    ["🎯 Test Model (Real-time)", "📊 Monitoring Dashboard (Real-data)"]
)

# ==========================================
# TAB 1: GỌI API THẬT
# ==========================================
with tab_inference:
    st.header("Upload Audio file to Test Inference")
    uploaded_file = st.file_uploader("Chọn file âm thanh (.wav)", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("🚀 Chạy Nhận Diện (Inference)", type="primary"):
            with st.spinner("Model đang xử lý..."):
                try:
                    start_time = time.time()
                    
                    # === ĐOẠN CODE GỐC BỊ ẨN ĐI ===
                    # files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "audio/wav")}
                    # response = requests.post(API_URL, files=files)
                    
                    # === ĐOẠN CODE MOCK (GIẢ LẬP) ===
                    time.sleep(1.5)  # Giả vờ như model đang chạy mất 1.5 giây
                    mock_result = {
                        "filename": uploaded_file.name,
                        "transcription": "vu thi yen mock",
                        "post_processed": "Vũ Thị Yến (Dữ liệu Mock)"
                    }
                    
                    process_time = time.time() - start_time
                    
                    # In kết quả giả lập ra màn hình
                    st.success(f"Hoàn thành trong {process_time:.2f} giây!")
                    st.info(f"**Kết quả:** {mock_result['post_processed']}")

                except Exception as e:
                    st.error(f"Lỗi kết nối API: {e}")

# ==========================================
# TAB 2: MONITORING DASHBOARD (REAL DATA)
# ==========================================
with tab_dashboard:
    st.button("🔄 Refresh Data")

    # 1. Lấy dữ liệu thực từ Prometheus
    total_requests = get_prom_value('http_requests_total{handler="/predict"}')
    avg_latency = get_prom_value(
        'rate(http_request_duration_seconds_sum{handler="/predict"}[5m]) '
        '/ rate(http_request_duration_seconds_count{handler="/predict"}[5m])'
    )
    rps = get_prom_value('rate(http_requests_total{handler="/predict"}[1m])')

    # 2. Hiển thị Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Inferences", int(total_requests))
    col2.metric("Current RPS", f"{rps:.2f}")
    col3.metric("Avg Latency", f"{avg_latency*1000:.1f} ms")
    col4.metric("Model Status", "Healthy", "GPU Active")

    # 3. Vẽ biểu đồ thực tế
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Real-time RPS (Last 5m)")
        rps_df = get_prom_series('rate(http_requests_total{handler="/predict"}[1m])')
        if not rps_df.empty:
            st.line_chart(rps_df["value"], color="#32CD32")
        else:
            st.write("Đang chờ dữ liệu từ Prometheus...")

    with chart_col2:
        st.subheader("Latency History (s)")
        lat_df = get_prom_series(
            'rate(http_request_duration_seconds_sum{handler="/predict"}[1m]) '
            '/ rate(http_request_duration_seconds_count{handler="/predict"}[1m])'
        )
        if not lat_df.empty:
            st.area_chart(lat_df["value"], color="#FFA500")
        else:
            st.write("Đang chờ dữ liệu...")

    st.caption(
        f"Last sync: {datetime.now().strftime('%H:%M:%S')} - Data source: http://prometheus:9090"
    )