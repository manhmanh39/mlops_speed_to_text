# 🎙️ Vietnamese Name Recognition System (Speech-to-Text) - MLOps Pipeline

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20Now!-brightgreen?style=for-the-badge&logo=streamlit)](https://raft-coeditor-stout.ngrok-free.dev/)

This repository implements a production-ready, end-to-end MLOps pipeline for a Speech-to-Text (STT) system optimized specifically for recognizing Vietnamese personal names. 

## 🚀 How to Use (Quick Start)

Depending on your goal, you can choose one of the following methods:

### 🌐 Option 1: Live Web UI (Zero Setup)
The fastest way to experience our STT system is through our live deployed Streamlit frontend. No installation or downloading is required!
👉 **[Access the Live STT Application Here](https://raft-coeditor-stout.ngrok-free.dev/)**
*(Note: This URL is temporarily tunneled via ngrok for the final presentation).*

---

### 🐳 Option 2: Run via Docker (For End Users)
If you want to run the API service locally without setting up a Python environment or downloading datasets, you can pull our pre-built container from the **GitHub Container Registry (GHCR)**. This image already includes the trained model weights and all dependencies.

```bash
# Pull and run the API service directly
docker run -p 8000:8000 ghcr.io/manhmanh39/mlops_speed_to_text:latest
```
*Access the API Swagger documentation at `http://localhost:8000/docs`.*

---

### 💻 Option 3: Development & Training (For Developers)
If you want to modify the code, view the datasets, or reproduce the training experiments:

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/manhmanh39/mlops_speed_to_text.git](https://github.com/manhmanh39/mlops_speed_to_text.git)
   cd mlops_speed_to_text
   ```

2. **Pull Data & Model Weights (DVC):**
   Large files (audio data and model checkpoints) are managed by DVC. To download them from our remote storage:
   ```bash
   pip install dvc[all]
   dvc pull
   ```

3. **Run with Docker Compose:**
   ```bash
   docker compose up wav2vec2-api
   ```

## 🛠️ Technology Stack & MLOps Pipeline

1. **Data Management:** `DVC` (Data Version Control) for tracking 2,804 training audio samples.
2. **Experiment Tracking:** `MLflow` for systematically logging hyperparameters and WER/CER metrics.
3. **CI/CD:** `GitHub Actions` for automated linting (`Ruff`), testing (`Pytest` with 36/36 passing tests), and building Docker images to **GHCR**.
4. **Orchestration:** `Kubernetes` manifests for scalable deployment and zero-downtime rolling updates.
5. **Observability:** `Prometheus` & `Grafana` for real-time monitoring of Data Drift using the **Kolmogorov-Smirnov (K-S) test**.

## 📊 Performance Summary
* **Final Model:** Fine-tuned `wav2vec2-large-vi-vlsp2020` with Connectionist Temporal Classification (CTC).
* **Accuracy:** 75.00% PASS rate on strict Vietnamese name recognition.
* **Inference:** FastAPI-based REST service coupled with a Streamlit interactive frontend.

## 👥 The MLOps Team (Class: DSEB 65B)
* **Nguyen Thi Mai Anh:** System Architect & MLOps Lead.
* **Pham Thi Ngoc Anh:** Model Serving & Frontend.
* **Nguyen Thi Huong Giang:** Data Engineering (DVC).
* **Le Lan Huong:** Machine Learning & MLflow Tracking.
* **Nguyen Khanh Huyen:** DevOps & CI Quality Gates.
* **Nguyen Thanh Mo:** Observability, CD & Drift Monitoring.

*Supervisor: Dr. Nguyen Manh Toan | Hanoi, April 2026*
