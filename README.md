# MRI-Based Alzheimer?s Disease Recognition Using Swin Transformer

## Project Overview

This research prototype extracts multi-stage features from a pretrained
Swin-Tiny Transformer and uses a StandardScaler and RBF Support Vector
Machine (SVM) for four-class MRI image classification.

## Dataset

- Total images: 6,400
- Classes:
  - Mild_Demented
  - Moderate_Demented
  - Non_Demented
  - Very_Mild_Demented

## Model

- Feature extractor: Swin-Tiny Transformer
- Feature dimension: 1,440
- Feature scaling: StandardScaler
- Training imbalance handling: SMOTE
- Classifier: RBF-SVM

## Recorded Evaluation Results

The saved test predictions contain 1,280 samples.

- Accuracy: 96.95%
- Balanced accuracy: 97.66%
- Macro precision: 97.77%
- Macro recall: 97.66%
- Macro F1-score: 97.72%

See `results/final_evaluation_report.txt` for per-class metrics and the
confusion matrix. See `results/model_metrics.csv` for spreadsheet-compatible
metrics.

## Project Structure

- `app.py` ? Streamlit dashboard
- `src/` ? preprocessing, model, training, evaluation, and utility code
- `models/` ? saved SVM and scaler
- `dataset/` ? dataset images
- `results/` ? evaluation reports and figures
- `docs/` ? project documentation
- `notebooks/` ? notebook documentation

## Run the Dashboard

Activate the project virtual environment, then run:

```powershell
python -m streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Limitations

The metrics above summarize saved predictions. The original test split,
image-to-label mapping, and model provenance must be independently verified
before describing the metrics as independently validated performance.
The Moderate_Demented test class contains only 11 samples. External dataset
performance has not been established.

This project is for academic research and demonstration only. It is not a
clinical diagnostic system.
