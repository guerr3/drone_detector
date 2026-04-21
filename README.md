# drone-detector-cv

## Project Overview
This repository contains a bachelor-level Deep Learning / Computer Vision project focused on drone detection in images. The project is designed as a clean and extensible academic starter setup using Python, TensorFlow/Keras, and notebook-based experimentation.

## Objective
The initial objective is **binary image classification**:
- Class 1: drone present
- Class 0: no drone

After a strong classification baseline, the project can be extended to **object detection** with bounding boxes.

## Dataset
The project expects an image dataset organized by split and class directories (e.g., `train/`, `val/`, `test/`).

Suggested next steps:
- collect/curate drone and non-drone images
- inspect class balance
- ensure consistent image quality and labeling

## Planned Approach
- Baseline image classification with TensorFlow/Keras
- Transfer learning (e.g., **MobileNetV2**) for faster convergence and better performance
- Data augmentation to improve generalization
- Evaluation with metrics, confusion matrix, and visualizations

## Project Structure
```text
.
├── configs/
│   └── config.yaml
├── data/
│   ├── processed/
│   └── raw/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   ├── data/
│   ├── models/
│   └── utils/
├── tests/
├── .gitignore
├── pyproject.toml
├── requirements-dev.txt
└── requirements.txt
```

## Installation
### Option 1: Local (VSCode)
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Option 2: Google Colab
- Open notebooks from `notebooks/`
- Install dependencies as needed in a first cell:
```python
!pip install -r requirements.txt
```

## Usage
1. Configure paths and hyperparameters in `configs/config.yaml`
2. Prepare dataset folders in `data/raw` (and optionally `data/processed`)
3. Start experimentation in `notebooks/` or call scripts in `src/models/`
4. Save trained model artifacts in `models/`

## Roadmap
- **Fase 1:** dataset verkennen en preprocessing
- **Fase 2:** baseline CNN of transfer learning classifier
- **Fase 3:** evaluatie en visualisatie
- **Fase 4:** verbetering via augmentation en fine-tuning
- **Fase 5:** uitbreiding naar object detection

## Future Extensions
- Move from image classification to object detection with bounding boxes
- Compare multiple backbones and fine-tuning strategies
- Add experiment tracking and reproducibility tooling
- Prepare deployment-ready inference pipeline
