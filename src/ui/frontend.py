import time
import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Wav2Vec2 MLOps Dashboard", page_icon="🎙️", layout="wide")

# --- 1. INTERFACE SETTINGS & SIDEBAR ---
# Dynamic path for the Logo in the new folder structure
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "assets", "Logo-NEU.png")

if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

st.sidebar.markdown("### Group 7 - DSEB65B")
st.sidebar.markdown("###### Nguyễn Thị Mai Anh")
st.sidebar.markdown("###### Phạm Thị Ngọc Ánh")
st.sidebar.markdown("###### Nguyễn Thanh Mơ")
st.sidebar.markdown("###### Nguyễn Khánh Huyền")
st.sidebar.markdown("###### Nguyễn Thị Hương Giang")
st.sidebar.markdown("###### Lê Lan Hương")
st.sidebar.markdown("---")

theme = st.sidebar.radio("🎨 Select Theme", ["Dark", "Light"])

# --- 2. DYNAMIC COLOR PALETTE ---
if theme == "Dark":
    colors = {
        "main_bg": "#0E1117",
        "card_bg": "#161B22",
        "text_main": "#FFFFFF",
        "text_sub": "#FAFBFC",
        "border": "#161B22",
        "btn_bg": "#161B22",
        "btn_text": "#FFFFFF",
        "btn_border": "#FFFFFF",
        "btn_hover": "#262730",
        "dropzone_bg": "#0D1117"
    }
else:
    colors = {
        "main_bg": "#FFFFFF",
        "card_bg": "#F0F2F6",
        "text_main": "#000000",
        "text_sub": "#31333F",
        "border": "#DADDE1",
        "btn_bg": "#FFFFFF",
        "btn_text": "#000000",
        "btn_border": "#000000",
        "btn_hover": "#F0F2F6",
        "dropzone_bg": "#F0F2F6"
    }

# Service Endpoints within Docker Network
PROMETHEUS_URL = "http://prometheus:9090/api/v1/query"
PROMETHEUS_RANGE_URL = "http://prometheus:9090/api/v1/query_range"
API_URL = "http://wav2vec2-api:8000/predict"

# --- 3. CUSTOM CSS ---
st.markdown(
    f"""
    <style>
    /* 1. Nền tổng thể */
    .stApp {{ 
        background-color: {colors['main_bg']}; 
        color: {colors['text_main']}; 
    }}

    /* 2. Tiêu đề và Header (Title, st.header) */
    h1, h2, h3, h4, h5, h6 {{
        color: {colors['text_main']} !important;
        opacity: 1 !important;
    }}

    /* 3. Các dòng nhãn (Select Theme, Upload Audio file...) */
    div[data-testid="stWidgetLabel"] p, label p, .stMarkdown p {{
        color: {colors['text_main']} !important;
        opacity: 1 !important;
    }}

    /* 4. Nút bấm (Refresh, Run Inference) */
    div.stButton > button {{
        background-color: {colors['card_bg']} !important;
        color: {colors['text_main']} !important;
        border: 1px solid {colors['border']} !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        transition: all 0.3s !important;
    }}
    
    div.stButton > button:hover {{
        background-color: {colors['btn_hover']} !important;
        border-color: {colors['border']} !important;
    }}

    /* 5. Khung File Uploader & Thanh File Đã Upload */
    
    /* a. Khung kéo thả file (Dropzone) */
    [data-testid="stFileUploadDropzone"] {{
        background-color: {colors['card_bg']} !important; 
        border: 2px dashed {colors['border']} !important; 
        border-radius: 8px !important;
    }}

    /* Ép màu mọi thành phần chữ, icon bên trong Dropzone */
    [data-testid="stFileUploadDropzone"] * {{
        color: {colors['text_main']} !important;
        fill: {colors['text_main']} !important; 
    }}

    /* Nút "Browse files" mặc định của Streamlit */
    [data-testid="baseButton-secondary"] {{
        background-color: {colors['btn_bg']} !important;
        color: {colors['text_main']} !important;
        border: 1px solid {colors['border']} !important;
    }}

    /* b. Thanh hiển thị file đã upload (Uploaded File Card) */
    [data-testid="stUploadedFile"] {{
        background-color: {colors['card_bg']} !important;
        border: 1px solid {colors['border']} !important;
        border-radius: 8px !important;
    }}

    /* Ép màu tên file, dung lượng bên trong thanh upload */
    [data-testid="stUploadedFile"] * {{
        color: {colors['text_main']} !important;
    }}

    /* c. Nút X (Xóa file) trên thanh upload */
    [data-testid="stUploadedFile"] button {{
        background-color: transparent !important;
    }}
    
    [data-testid="stUploadedFile"] button svg {{
        fill: {colors['text_main']} !important;
    }}

    
    /* 6. Toàn bộ nền Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {colors['main_bg']} !important; /* Đổ màu tối cho cả thanh sidebar */
        border-right: 1px solid {colors['border']} !important; /* Đường kẻ dọc chia sidebar và nội dung */
    }}

    /* Chữ trong Sidebar giữ nguyên màu trắng, không có nền riêng lẻ */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h6 {{
        color: {colors['text_main']} !important;
        background-color: transparent !important; /* Đảm bảo không bị hiện ô màu riêng lẻ */
        margin-bottom: 2px !important;
    }}
    
    /* Chỉnh luôn màu cho phần Radio Button chọn Theme trong Sidebar */
    [data-testid="stSidebar"] label p {{
        color: {colors['text_main']} !important;
    }}

    /* 7. Metric Cards */
    div[data-testid="metric-container"] {{ 
        background-color: {colors['card_bg']}; 
        border: 2px solid {colors['border']}; 
    }}
    
    /* 8. Tabs */
    .stTabs [data-baseweb="tab-list"] button {{
        color: {colors['text_sub']} !important;
    }}
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{ 
        color: #ff4b4b !important; 
        border-bottom: 2px solid #ff4b4b !important;
    }}
    </style>
""",
    unsafe_allow_html=True,
)

# --- HELPER FUNCTIONS ---
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
        response = requests.get(PROMETHEUS_RANGE_URL, params={"query": query, "start": start, "end": end, "step": "10s"}, timeout=2)
        values = response.json()["data"]["result"][0]["values"]
        df = pd.DataFrame(values, columns=["timestamp", "value"])
        df["value"] = df["value"].astype(float)
        return df
    except Exception:
        return pd.DataFrame(columns=["value"])

# --- UI HEADER ---
st.title("🎙️ MLOps: Vietnamese Speech-to-Text")
tab_inference, tab_dashboard = st.tabs(
    ["🎯 Model Testing (Real-time)", "📊 Monitoring Dashboard (Real-data)"]
)

# --- TAB 1: INFERENCE ---
with tab_inference:
    st.header("Upload Audio file to Test Inference")
    uploaded_file = st.file_uploader("Choose an audio file (.wav)", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("🚀 Run Inference", type="primary"):
            with st.spinner("Model is processing..."):
                try:
                    start_time = time.time()
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "audio/wav")}
                    response = requests.post(API_URL, files=files)
                    process_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        api_result = response.json()
                        st.success(f"Completed in {process_time:.2f} seconds!")
                        final_text = api_result.get("post_processed", api_result.get("transcription", "Inference failed"))
                        st.info(f"**Transcription:** {final_text}")
                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"API Connection Failed: {e}")

# --- TAB 2: MONITORING ---
with tab_dashboard:
    if st.button("🔄 Refresh Data"):
        st.rerun()

    # Metrics from Prometheus
    total_requests = get_prom_value('http_requests_total{handler="/predict"}')
    avg_latency = get_prom_value('rate(http_request_duration_seconds_sum{handler="/predict"}[5m]) / rate(http_request_duration_seconds_count{handler="/predict"}[5m])')
    rps = get_prom_value('rate(http_requests_total{handler="/predict"}[1m])')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Inferences", int(total_requests))
    col2.metric("Current RPS", f"{rps:.2f}")
    col3.metric("Avg Latency", f"{avg_latency*1000:.1f} ms")
    col4.metric("Model Status", "Healthy", "GPU Active")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Real-time RPS (Last 5m)")
        rps_df = get_prom_series('rate(http_requests_total{handler="/predict"}[1m])')
        if not rps_df.empty:
            st.line_chart(rps_df["value"], color="#32CD32")
        else:
            st.write("Waiting for data from Prometheus...")

    with chart_col2:
        st.subheader("Latency History (s)")
        lat_df = get_prom_series('rate(http_request_duration_seconds_sum{handler="/predict"}[1m]) / rate(http_request_duration_seconds_count{handler="/predict"}[1m])')
        if not lat_df.empty:
            st.area_chart(lat_df["value"], color="#FFA500")
        else:
            st.write("Waiting for data...")

    st.caption(f"Last sync: {datetime.now().strftime('%H:%M:%S')} - Data source: http://prometheus:9090")

    st.markdown("---")
    st.subheader("🕵️ Data Drift Detection (K-S Test)")
    
    drift_score = get_prom_value('asr_ks_drift_score{feature_name="audio_duration"}')
    
    if drift_score > 0.3:
        st.error(f"⚠️ HIGH DRIFT DETECTED: {drift_score:.4f} (Input significantly differs from Training set!)")
    elif drift_score > 0.15:
        st.warning(f"🟡 Warning: Minor Drift detected: {drift_score:.4f}")
    else:
        st.success(f"✅ Data Stable: {drift_score:.4f} (Input distribution matches Training set)")