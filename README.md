# 🎙️ MLOps Speed-to-Text with Wav2Vec 2.0

This repository implements a production-ready MLOps pipeline for a Speech-to-Text (STT) system based on the **Wav2Vec 2.0** architecture. The project focuses on automating the lifecycle of an AI model—from code quality enforcement and automated testing to containerized deployment.

## Team Member
* Nguyễn Thị Mai Anh
* Phạm Thị Ngọc Ánh
* Nguyễn Thị Hương Giang
* Nguyễn Khánh Huyền
* Lê Lan Hương
* Nguyễn Thanh Mơ
## 🚀 Key Features

* **State-of-the-art Model:** Utilizes a fine-tuned Wav2Vec 2.0 model optimized for the Vietnamese language.
* **Robust CI/CD Pipeline:** Fully automated workflow using GitHub Actions to enforce code quality (**Ruff, Flake8**) and run **Unit Tests** (36/36 passed) on every push.
* **Full Containerization:** Standardized environment using **Docker** and **Docker Compose**, ensuring "it works on my machine" translates to "it works everywhere."
* **Production API:** High-performance RESTful API built with **FastAPI**, featuring automatic Swagger UI documentation at `localhost:8000/docs`.
* **Security Best Practices:** Sensitive information (Hugging Face Tokens, API Keys) is managed securely via environment variables (`.env`).

## 🛠 Prerequisites

* **Docker Desktop** (with WSL2 backend enabled).
* **Docker Compose**.
* **Hugging Face Account** (to generate a User Access Token for model weights).

## 📥 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/manhmanh39/mlops_speed_to_text.git
cd mlops_speed_to_text
```

### 2. Configuration
Create a `.env` file in the root directory and add your credentials:
```env
HF_TOKEN=your_huggingface_token_here
WANDB_API_KEY=your_wandb_key_here
```

### 3. Deploy with Docker Compose
To launch the STT API service, run:
```bash
docker compose up wav2vec2-api
```
*Note: During the first run, the system will automatically download the pre-trained model weights (~1.2GB) from the Hugging Face Hub.*

## 🧪 Quality Assurance & Testing

This project adheres to strict software engineering standards:
* **Linting:** `Ruff` and `Flake8` are used to maintain clean, PEP8-compliant code.
* **Unit Testing:** 36 comprehensive test cases covering audio preprocessing, text normalization, and inference logic.
* **Automated Registry:** Upon passing all tests, the system automatically builds and pushes the Docker image to the **GitHub Container Registry (GHCR)**.

## 📁 Project Structure
* `app.py`: FastAPI application for model serving.
* `train_wav2vec2.py`: Script for model fine-tuning and training logic.
* `eval_wav2vec2.py`: Evaluation utilities and metrics.
* `tests/`: Automated test suite for CI/CD validation.
* `Dockerfile`: Linux-based container configuration for deployment.
