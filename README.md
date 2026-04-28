# SignLang Recognition System

HELLO 
HELLO HELLO
HELLO HELLO HELLO 
HELLO HELLO HELLO  
HELLO HELLO HELLO HELLO 


[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#)
[![Models](https://img.shields.io/badge/models-3%20(YOLOv8%20%2B%20CNN%20%2B%20RF)-orange.svg)](#)
[![CPU](https://img.shields.io/badge/runtime-CPU-lightgrey.svg)](#)

End-to-end American Sign Language recognition: **YOLOv8 detection + PyTorch CNN + Random Forest on MediaPipe keypoints**, fused via weighted-vote ensemble, exposed through a FastAPI service with WebSocket real-time streaming, SQLite analytics, and a vanilla-JS web UI with English / Telugu / Tamil translation.

Runs entirely on CPU (tested on Apple M4).

## Architecture

```
                       ┌────────────────────────────────────────────┐
                       │             FastAPI service                │
                       │ /predict   /stream(ws)   /analytics /health│
                       └───────┬───────────────────┬────────────────┘
                               │                   │
                  base64 image │                   │ history + counts
                               ▼                   ▼
   ┌───────────────────────────────────┐   ┌──────────────────────┐
   │           Ensemble                │   │   SQLAlchemy / SQLite│
   │  ┌──────────┐ ┌────────┐ ┌────┐   │   │ predictions sessions │
   │  │ YOLOv8n  │ │  CNN   │ │ RF │   │   │ feedback             │
   │  │ detector │ │ 3-conv │ │ kp │   │   └──────────────────────┘
   │  └────┬─────┘ └────┬───┘ └─┬──┘   │
   │       │ box+crop   │ class │ class│
   │       └────────┬───┴───────┘      │
   │                ▼ weighted vote    │
   │           label + conf            │
   └────────────────┬──────────────────┘
                    │
                    ▼
   ┌────────────────────────────────────┐
   │ Translator (rolling buffer 5)      │
   │ → English sentence                 │
   │ → deep-translator → Telugu / Tamil │
   └────────────────────────────────────┘
                    │
                    ▼
   ┌────────────────────────────────────┐
   │ Frontend (HTML + Chart.js)         │
   │ webcam · boxes · sentence · charts │
   └────────────────────────────────────┘
```

## Project layout

```
signlang-detector/
├── data/             dataset.py, preprocess.py (MediaPipe), analysis.ipynb
├── models/           yolo_model.py, cnn_model.py, rf_model.py, ensemble.py
├── pipeline/         detector.py, classifier.py, translator.py, regional.py
├── api/              main.py, schemas.py, routes/{predict,stream,analytics}.py
├── database/         db.py, models.py, crud.py  (SQLAlchemy + SQLite)
├── frontend/         index.html, stream.js, analytics.js
├── scripts/          train.py, evaluate.py, realtime.py
├── tests/            test_pipeline.py, test_api.py
├── docker/Dockerfile
├── notebooks/analysis.ipynb
└── results/evaluation.csv
```

## Setup

```bash
conda create -n signlang python=3.10 -y
conda activate signlang
pip install -r requirements.txt
```

## Train all three models

```bash
python scripts/train.py \
  --yolo-epochs 5 --yolo-imgsz 256 --yolo-fraction 0.12 \
  --cnn-epochs 1 --cnn-batch 64 \
  --kp-limit 1200
```

Weights land in `models/weights/{yolo_best.pt, cnn_best.pt, rf_model.joblib}`.

## Evaluate

```bash
python scripts/evaluate.py
# -> results/evaluation.csv
```

## Run the API + UI

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

## Standalone webcam demo

```bash
python scripts/realtime.py    # press q to quit
```

## Tests

```bash
pytest -q
```

## Docker

```bash
docker build -f docker/Dockerfile -t signlang .
docker run -p 8000:8000 signlang
```

## Results

Real numbers from the runs in this repo (CPU, Apple M4). Training: YOLO 5 epochs / imgsz 256 / fraction 0.12; CNN 1 epoch on 14k+ crops; RF 200 trees on ~435 MediaPipe keypoint vectors. Test set: 200 labeled `test/` samples (RF effective N smaller — many crops yield no MediaPipe hand on this dataset).

| Model   | mAP50  | Accuracy | F1 (macro) | Avg inference (ms) |
|---------|--------|----------|------------|--------------------|
| YOLOv8n | 0.0396 | 0.020    | 0.020      | 9.9                |
| CNN     | —      | 0.565    | 0.471      | 1.3                |
| RF (kp) | —      | 0.429    | 0.279      | 22.7               |

Notes:
- YOLO is severely under-trained (5 epochs over 12% of data on CPU). Bumping to `--yolo-epochs 50 --yolo-fraction 1.0` on a GPU is what the IEEE paper assumes.
- CNN performs best per second of training on CPU because it consumes the cropped hand region directly (no MediaPipe gating).
- RF accuracy is bounded by MediaPipe hand-detection coverage on the test crops.
- Full training summary: `results/train_summary.json`. Per-class confusion + feature importance: `data/analysis.ipynb`.

## Demo

![demo placeholder](docs/demo.gif)

## How to cite

If this codebase or the underlying approach is useful in academic work, please cite the IEEE ICCRTEE 2025 paper this implementation is modelled after:

```bibtex
@inproceedings{signlang_iccrtee2025,
  title     = {Real-Time Sign Language Recognition with a Multi-Model Ensemble and Regional Translation},
  booktitle = {Proc. IEEE Int'l Conf. on Current Research Trends in Engineering and Education (ICCRTEE)},
  year      = {2025}
}
```

## Further reading

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — full data-flow diagram and module boundaries
- [docs/MODELS.md](docs/MODELS.md) — model cards (YOLO / CNN / RF / Ensemble)
- [docs/API.md](docs/API.md) — REST + WebSocket reference
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — Docker, reverse proxy, production hardening
- [CONTRIBUTING.md](CONTRIBUTING.md) — dev setup + PR checklist
- [CHANGELOG.md](CHANGELOG.md)

## License

[MIT](LICENSE) — dataset is CC BY 4.0 and is **not** redistributed in this repo.
