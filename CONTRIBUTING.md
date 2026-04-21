# Contributing

Thanks for your interest. This is a personal research repo, but PRs that improve
training speed, model accuracy, frontend UX, or test coverage are welcome.

## Dev setup

```bash
conda create -n signlang python=3.10 -y
conda activate signlang
pip install -r requirements.txt
```

Download the dataset (one-time):

```bash
python scripts/download_data.py    # writes American-sign-language-2/
```

## Run before opening a PR

```bash
pytest -q                          # 8 tests, ~4s
python scripts/train.py --skip-yolo --skip-cnn --skip-rf  # smoke test
ruff check . || true               # optional
```

## Branch / commit style

- Branch off `main`: `feat/...`, `fix/...`, `docs/...`
- Conventional commits encouraged (`feat:`, `fix:`, `docs:`, `refactor:`)
- One logical change per PR

## What's intentionally out of scope

- GPU-only optimisations (this project is CPU-first by design).
- Training-data redistribution (we link to Roboflow under CC BY 4.0).
- Heavyweight JS frameworks for the frontend (vanilla JS only).

## Reporting issues

Please include: OS, Python version, output of `pip freeze | grep -E "ultralytics|torch|mediapipe"`, and the exact command that failed.
