# Makefile — Baseline experiment runner
# Dùng: make eval-e1, make eval-all, make eval-baselines, v.v.
#
# Yêu cầu: Docker + docker compose plugin (compose v2)
# File này đặt cùng cấp với deployment/docker-compose.yml

COMPOSE      := docker compose -f deployment/docker-compose.yml
COMPOSE_RUN  := $(COMPOSE) run --rm

# ─── Build image ─────────────────────────────────────────────────────────────
.PHONY: build
build:
	$(COMPOSE) build wav2vec2-eval-e1

# ─── MLflow ──────────────────────────────────────────────────────────────────
.PHONY: mlflow-up mlflow-down
mlflow-up:
	$(COMPOSE) up -d mlflow
	@echo "MLflow đang chạy tại http://localhost:5000"

mlflow-down:
	$(COMPOSE) stop mlflow

# ─── Từng experiment ─────────────────────────────────────────────────────────

## E1: Base model, không finetune, không preproc, simple match
.PHONY: eval-e1
eval-e1: mlflow-up
	@echo "▶ E1 — Baseline Base"
	$(COMPOSE_RUN) wav2vec2-eval-e1
	@echo "✅ E1 xong"

## E2: Large model, không finetune, không preproc, simple match
.PHONY: eval-e2
eval-e2: mlflow-up
	@echo "▶ E2 — Baseline Large"
	$(COMPOSE_RUN) wav2vec2-eval-e2
	@echo "✅ E2 xong"

## E3: Large model + finetuned weights, không preproc, simple match
.PHONY: eval-e3
eval-e3: mlflow-up
	@echo "▶ E3 — Finetune Only"
	$(COMPOSE_RUN) wav2vec2-eval-e3
	@echo "✅ E3 xong"

## E4: Large model + finetuned weights + preproc, simple match
.PHONY: eval-e4
eval-e4: mlflow-up
	@echo "▶ E4 — Finetune + Data Engineering"
	$(COMPOSE_RUN) wav2vec2-eval-e4
	@echo "✅ E4 xong"


# ─── Chạy gộp ────────────────────────────────────────────────────────────────

.PHONY: eval-baselines
eval-baselines: eval-e1 eval-e2 eval-e3
	@echo ""
	@echo "════════════════════════════════════════"
	@echo "  Tất cả baselines (E1–E3) đã hoàn thành"
	@echo "  Xem kết quả: http://localhost:5000"
	@echo "════════════════════════════════════════"

.PHONY: eval-all
eval-all: eval-e1 eval-e2 eval-e3 eval-e4
	@echo ""
	@echo "════════════════════════════════════════"
	@echo "  Toàn bộ experiments (E1–E4) hoàn thành"
	@echo "  Xem kết quả: http://localhost:5000"
	@echo "════════════════════════════════════════"

# ─── Tiện ích ────────────────────────────────────────────────────────────────

## Dọn containers đã dừng
.PHONY: clean
clean:
	$(COMPOSE) down --remove-orphans
	@echo "Đã dọn containers"

## Xem logs của một service, ví dụ: make logs SERVICE=wav2vec2-eval-e1
.PHONY: logs
logs:
	$(COMPOSE) logs -f $(SERVICE)

.PHONY: help
help:
	@echo ""
	@echo "  make build           Build Docker image"
	@echo "  make mlflow-up       Khởi động MLflow server"
	@echo ""
	@echo "  make eval-e1         Chạy E1 (Baseline Base)"
	@echo "  make eval-e2         Chạy E2 (Baseline Large)"
	@echo "  make eval-e3         Chạy E3 (Finetune Only)"
	@echo "  make eval-e4         Chạy E4 (Finetune + DataEng)"
	@echo "  make eval-e5         Chạy E5 (Production)"
	@echo ""
	@echo "  make eval-baselines  Chạy E1→E2→E3→E4 tuần tự"
	@echo "  make eval-all        Chạy E1→E2→E3→E4→E5 tuần tự"
	@echo ""
	@echo "  make logs SERVICE=wav2vec2-eval-e1   Xem log"
	@echo "  make clean           Dọn containers"
	@echo ""