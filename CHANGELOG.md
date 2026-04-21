# Changelog

All notable changes to this project will be documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-04-20

### Added
- Initial release of the multi-model SignLang recognition system.
- YOLOv8n detector + 3-conv PyTorch CNN + RandomForest on MediaPipe keypoints.
- Weighted-vote ensemble with per-model confidence breakdown.
- FastAPI service: `POST /predict`, `WS /stream`, `GET /analytics`, `GET /health`, `POST /feedback`, `POST /reset-sentence`.
- SQLite (SQLAlchemy 2.x) prediction / session / feedback logging.
- Vanilla-JS frontend with webcam stream, bounding-box overlay, sentence translator, Telugu/Tamil toggle, Chart.js bar chart.
- English → Telugu / Tamil translation via `deep-translator` with offline fallback.
- `scripts/train.py`, `scripts/evaluate.py`, `scripts/realtime.py`.
- `tests/` (8 tests, pytest).
- `docker/Dockerfile` (CPU `python:3.10-slim` base).
- Docs: `README.md`, `docs/{ARCHITECTURE,API,DEPLOYMENT,MODELS}.md`.
- GitHub Actions CI on `main` + PRs.
