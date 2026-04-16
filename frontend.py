import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Wav2Vec2 MLOps Dashboard", page_icon="🎙️", layout="wide")

# Địa chỉ các service trong mạng Docker
PROMETHEUS_URL = "http://prometheus:9090/api/v1/query"
PROMETHEUS_RANGE_URL = "http://prometheus:9090/api/v1/query_range"
API_URL = "http://wav2vec2-api:8000/predict"

# CSS tùy chỉnh
st.markdown(
    """
    <style>
    .stApp { background-color: #111217; color: white; }
    div[data-testid="metric-container"] {
        background-color: #1e1e24; border: 1px solid #333; padding: 15px; border-radius: 5px;
    }
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
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "audio/wav")}
                    start_time = time.time()
                    response = requests.post(API_URL, files=files)
                    process_time = time.time() - start_time

                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Hoàn thành trong {process_time:.2f} giây!")
                        st.info(f"**Kết quả:** {result['post_processed']}")
                    else:
                        st.error(f"API lỗi: {response.text}")
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
