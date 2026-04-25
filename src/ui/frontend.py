import os
from datetime import datetime

import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Wav2Vec2 MLOps Dashboard", page_icon="🎙️", layout="wide")

# --- 1. CONFIG & ENDPOINTS (Đưa lên đầu để dùng chung) ---
PROMETHEUS_URL = "http://prometheus:9090/api/v1/query"
API_URL = "http://wav2vec2-api:8000/predict"
# Link Grafana Real-time
GRAFANA_BASE = "http://localhost:3000/d-solo/adzwt9p/mlops-speed-to-text?orgId=1&from=now-15m&to=now&theme=light&kiosk"


# --- 2. HELPER FUNCTIONS (Phải định nghĩa TRƯỚC khi gọi) ---
def get_prom_value(query):
    try:
        response = requests.get(PROMETHEUS_URL, params={"query": query}, timeout=2)
        results = response.json()["data"]["result"]
        return float(results[0]["value"][1]) if results else 0.0
    except Exception:
        return 0.0


# --- 3. INTERFACE SETTINGS & SIDEBAR ---
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "assets", "Logo-NEU.png")

if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

st.sidebar.markdown("### Group 7 - DSEB65B")
members = [
    "Nguyễn Thị Mai Anh", "Phạm Thị Ngọc Ánh", "Nguyễn Thanh Mơ",
    "Nguyễn Khánh Huyền", "Nguyễn Thị Hương Giang", "Lê Lan Hương"
]
for member in members:
    st.sidebar.markdown(f"###### {member}")

st.sidebar.markdown("---")

# --- 4. UI HEADER & TABS ---
st.title("🎙️ MLOps: Vietnamese Speech-to-Text")
tab_inference, tab_dashboard = st.tabs(
    ["🎯 Model Testing (Real-time)", "📊 Monitoring Dashboard (Real-data)"]
)

# --- TAB 1: INFERENCE ---
with tab_inference:
    st.header("Upload Audio file to Test Inference")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "flac", "ogg"])

    if uploaded_file is not None:
        st.audio(uploaded_file)
        if st.button("🚀 Run Inference", type="primary"):
            with st.spinner("Model is processing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "audio/wav")}
                    response = requests.post(API_URL, files=files)
                    if response.status_code == 200:
                        api_result = response.json()
                        st.success("Inference Completed!")
                        st.info(f"**Transcription:** {api_result.get('post_processed')}")
                    else:
                        st.error(f"API Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Failed: {e}")

# --- TAB 2: MONITORING ---
with tab_dashboard:
    if st.button("🔄 Refresh Data"):
        st.rerun()

    # SECTION 1: QUICK METRICS (KPIs)
    st.subheader("📍 Key Performance Indicators")
    m1, m2, m3, m4 = st.columns(4)

    total_req = get_prom_value('http_requests_total{handler="/predict"}')
    rps = get_prom_value('rate(http_requests_total{handler="/predict"}[1m])')
    latency = get_prom_value('rate(http_request_duration_seconds_sum[5m])/rate(http_request_duration_seconds_count[5m])')
    ram_usage = get_prom_value('container_memory_usage_bytes{name="wav2vec2-api"}/1024/1024')

    m1.metric("Total Requests", int(total_req))
    m2.metric("Current RPS", f"{rps:.2f}")
    m3.metric("Avg Latency", f"{latency*1000:.1f} ms")
    m4.metric("API RAM Usage", f"{ram_usage:.1f} MB", "Stable")

    st.markdown("---")

    # SECTION 2: OPERATIONAL
    st.subheader("🚀 System Operational Monitoring")
    chart_row1 = st.columns(2)
    with chart_row1[0]:
        st.markdown("##### Real-time Traffic (RPS)")
        components.iframe(f"{GRAFANA_BASE}&panelId=3", height=300)
    with chart_row1[1]:
        st.markdown("##### Latency P95 Stability")
        components.iframe(f"{GRAFANA_BASE}&panelId=4", height=300)

    st.markdown("---")

    # SECTION 3: MODEL QUALITY & DRIFT
    st.subheader("🕵️ Advanced Model Quality & Drift")
    chart_row2 = st.columns([2, 1])
    with chart_row2[0]:
        st.markdown("##### Audio Duration Drift (K-S Test)")
        components.iframe(f"{GRAFANA_BASE}&panelId=6", height=300)
    with chart_row2[1]:
        st.markdown("##### System Error Rate (%)")
        components.iframe(f"{GRAFANA_BASE}&panelId=2", height=300)

    st.markdown("---")

    # SECTION 4: DATA DISTRIBUTION
    st.subheader("📊 Data Distribution Insights")
    chart_row3 = st.columns(2)
    with chart_row3[0]:
        st.markdown("##### Transcription Word Count Distribution")
        components.iframe(f"{GRAFANA_BASE}&panelId=7", height=300)
    with chart_row3[1]:
        st.markdown("##### Input Audio Duration (Buckets)")
        components.iframe(f"{GRAFANA_BASE}&panelId=8", height=300)

    st.caption(f"Last sync: {datetime.now().strftime('%H:%M:%S')} | Engine: Grafana v10.x | Group 7 - DSEB65B")
