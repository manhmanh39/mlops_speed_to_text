import streamlit as st
import pandas as pd
import numpy as np
import time

# Giả lập giao diện tối mặc định của Grafana
st.set_page_config(page_title="Grafana - ML Monitoring", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS để làm cho nó giống Grafana Dashboard hơn
st.markdown("""
    <style>
    .stApp { background-color: #111217; color: white; }
    div[data-testid="metric-container"] {
        background-color: #1e1e24; border: 1px solid #333; padding: 15px; border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 MLOps Model Monitoring Dashboard (Mock)")
st.caption("This is a mock dashboard simulating Prometheus/Grafana metrics for the Wav2Vec2 Model.")

# Lấy dữ liệu giả lập
def get_mock_data(base_val, noise_level, rows=50):
    return pd.DataFrame(
        np.random.randn(rows, 1) * noise_level + base_val,
        columns=["Metric"]
    )

# --- KHU VỰC THẺ CHỈ SỐ (METRICS) ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current RPS (Requests/sec)", "24.3", "+1.2")
col2.metric("Latency p95 (ms)", "142 ms", "-5 ms")
col3.metric("Error Rate", "0.02 %", "0.00 %")
col4.metric("Feature Drift Score", "0.015", "Normal")

st.divider()

# --- KHU VỰC BIỂU ĐỒ (CHARTS) ---
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Requests Per Second (RPS)")
    st.caption("rate(model_predictions_total[1m])")
    # Biểu đồ mô phỏng lượng request dao động
    chart_data = get_mock_data(20, 5)
    st.line_chart(chart_data, color="#32CD32", height=250)

with chart_col2:
    st.subheader("Inference Latency (p50, p95, p99)")
    st.caption("histogram_quantile(0.95, rate(model_inference_latency_seconds_bucket[5m]))")
    # Biểu đồ mô phỏng Latency với 3 đường
    latency_data = pd.DataFrame({
        "p50": np.random.randn(50) * 10 + 80,
        "p95": np.random.randn(50) * 15 + 140,
        "p99": np.random.randn(50) * 20 + 200,
    })
    st.line_chart(latency_data, color=["#00BFFF", "#FFA500", "#FF4500"], height=250)

st.divider()

bottom_col1, bottom_col2 = st.columns(2)
with bottom_col1:
    st.subheader("Prediction Distribution")
    hist_data = pd.DataFrame(np.random.normal(50, 15, size=(100, 1)), columns=["Confidence Score"])
    st.bar_chart(hist_data, height=200, color="#8A2BE2")

with bottom_col2:
    st.subheader("Active Alerts")
    st.success("✅ No active alerts. Model is healthy.")
    st.info("ℹ️ Next scheduled drift check in: 14 mins")