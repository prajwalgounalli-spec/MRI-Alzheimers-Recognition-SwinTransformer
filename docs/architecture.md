# System Architecture

## Project
An MRI-Based Swin Transformer Approach for Automated Alzheimer?s Disease Recognition

## Workflow

MRI image
    |
    v
Image preprocessing
    |  Resize to 224 x 224
    |  Convert to 3 channels
    |  Normalize using ImageNet statistics
    v
Swin Transformer Tiny feature extractor
    |  Stage feature dimensions: 96, 192, 384, 768
    |  Global average pooling
    v
Concatenated feature vector (1440 dimensions)
    |
    v
Saved feature scaler
    |
    v
Saved SVM classifier
    |
    v
Predicted numeric class
    |
    v
Verified class-name mapping
    |
    v
Streamlit dashboard

## Software
- Python
- PyTorch
- TorchVision
- timm
- Scikit-learn
- Streamlit
- NumPy
- Pillow
- Joblib

## Evaluation
Evaluate on a separate held-out test set using accuracy,
precision, recall, macro F1-score, and a confusion matrix.

## Limitations
- The class mapping must be verified against the original
  Colab training labels.
- Reported test results must be reproducible from saved
  test labels and predictions.
- A single MRI prediction is not a clinical diagnosis.
- The central-region crop utility is not a trained
  hippocampus detector.
