from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import timm
import torch

from PIL import Image
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Alzheimer's MRI Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "fusion_svm.joblib"
SCALER_PATH = MODEL_DIR / "fusion_scaler.joblib"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# COLORFUL DASHBOARD DESIGN
# ============================================================

st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f5f3ff 0%,
            #eff6ff 50%,
            #fdf2f8 100%
        );
    }

    /* Hide Streamlit's default header */
    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.65);
    }

    /* Main title */
    h1 {
        color: #4c1d95 !important;
        font-weight: 800 !important;
    }

    h2, h3 {
        color: #5b21b6 !important;
    }

    /* Gradient hero section */
    .hero-card {
        padding: 32px;
        border-radius: 22px;
        margin: 10px 0 25px 0;
        color: white;
        background: linear-gradient(
            120deg,
            #6d28d9 0%,
            #2563eb 55%,
            #db2777 100%
        );
        box-shadow: 0 10px 30px rgba(76, 29, 149, 0.18);
    }

    .hero-card h2 {
        color: white !important;
        font-size: 30px;
        margin-bottom: 10px;
    }

    .hero-card p {
        color: white;
        font-size: 16px;
        line-height: 1.7;
        margin-bottom: 0;
    }

    /* Upload and result cards */
    .upload-card,
    .result-card,
    .info-card {
        padding: 22px;
        border-radius: 18px;
        margin: 15px 0;
        background: rgba(255, 255, 255, 0.93);
        border: 1px solid #ddd6fe;
        box-shadow: 0 5px 18px rgba(76, 29, 149, 0.07);
    }

    .upload-card h3,
    .result-card h3,
    .info-card h3 {
        color: #5b21b6 !important;
    }

    .upload-card p {
        color: #64748b;
        font-size: 14px;
    }

    /* File upload area */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #8b5cf6;
        border-radius: 15px;
        padding: 14px;
    }

    /* Hide selected file details and filename */
    [data-testid="stFileUploaderFile"] {
        display: none !important;
    }

    [data-testid="stFileUploaderFileName"] {
        display: none !important;
    }

    [data-testid="stFileUploaderDeleteBtn"] {
        display: none !important;
    }

    /* Colorful metric cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.96);
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #ddd6fe;
        box-shadow: 0 5px 15px rgba(76, 29, 149, 0.08);
    }

    [data-testid="stMetricLabel"] {
        color: #6d28d9 !important;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #312e81 !important;
        font-weight: 800;
    }

    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(
            90deg,
            #6d28d9,
            #2563eb
        );
        color: white;
        border: none;
        border-radius: 12px;
        min-height: 48px;
        font-size: 16px;
        font-weight: 700;
        box-shadow: 0 5px 14px rgba(76, 29, 149, 0.20);
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(
            90deg,
            #5b21b6,
            #1d4ed8
        );
        color: white;
        border: none;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #5b21b6;
        font-weight: 600;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #7c3aed;
    }

    /* Data tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #ddd6fe;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Information text */
    .small-note {
        color: #64748b;
        font-size: 13px;
        line-height: 1.7;
    }

    /* Footer */
    .footer-card {
        text-align: center;
        padding: 18px;
        color: #64748b;
        font-size: 13px;
        border-top: 1px solid #ddd6fe;
        margin-top: 35px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD SAVED CLASSIFIER
# ============================================================

@st.cache_resource
def load_classifier():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved classifier not found: {MODEL_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Saved scaler not found: {SCALER_PATH}"
        )

    classifier = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    if scaler.n_features_in_ != 1440:
        raise ValueError(
            f"Expected 1440 input features, but scaler expects "
            f"{scaler.n_features_in_}."
        )

    return classifier, scaler


# ============================================================
# SWIN TRANSFORMER-TINY FEATURE EXTRACTOR
# ============================================================

@st.cache_resource
def load_feature_extractor():

    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=True,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image, feature_model):

    tensor = transform(
        image.convert("RGB")
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        stages = feature_model(tensor)

    pooled_features = []

    for feature, channels in zip(
        stages,
        [96, 192, 384, 768],
    ):

        if feature.ndim != 4:
            raise ValueError(
                f"Unexpected feature shape: {tuple(feature.shape)}"
            )

        # Support NHWC and NCHW feature layouts.
        if feature.shape[-1] == channels:
            pooled = feature.mean(dim=(1, 2))

        elif feature.shape[1] == channels:
            pooled = feature.mean(dim=(2, 3))

        else:
            raise ValueError(
                f"Unexpected feature layout: {tuple(feature.shape)}"
            )

        pooled_features.append(
            pooled.cpu().numpy()
        )

    combined = np.concatenate(
        pooled_features,
        axis=1,
    )

    if combined.shape[1] != 1440:
        raise ValueError(
            f"Expected 1440 features; received {combined.shape[1]}."
        )

    return combined


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <h2>🧠 Alzheimer's Disease MRI Prediction</h2>
        <p>
            An MRI-Based Swin Transformer Approach for Automated
            Alzheimer's Disease Recognition
        </p>
        <p>
            Upload an MRI image to explore the research model's
            classification output.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Research prototype | Four-class MRI image classification"
)

st.warning(
    "This application is for academic research and demonstration only. "
    "It is not a medical diagnostic tool."
)


# ============================================================
# NAVIGATION TABS
# ============================================================

tab_predict, tab_results, tab_info = st.tabs([
    "🧠 MRI Prediction",
    "📊 Project Results",
    "📘 Project Information",
])


# ============================================================
# TAB 1: MRI PREDICTION
# ============================================================

with tab_predict:

    st.markdown(
        """
        <div class="upload-card">
            <h3>📤 Upload MRI Scan</h3>
            <p>
                Select an image in PNG, JPG, JPEG, or BMP format.
                Your selected filename and MRI preview will be hidden.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Select MRI image",
        type=["png", "jpg", "jpeg", "bmp"],
        key="mri_upload",
        label_visibility="collapsed",
    )

    # Process the uploaded image without displaying its name or preview.
    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.success(
                "MRI image uploaded successfully."
            )

            if st.button(
                "🔍 Analyze MRI",
                type="primary",
                use_container_width=True,
            ):

                try:

                    with st.spinner(
                        "Analyzing MRI image. Please wait..."
                    ):

                        classifier, scaler = load_classifier()

                        feature_model = load_feature_extractor()

                        features = extract_features(
                            image,
                            feature_model,
                        )

                        scaled_features = scaler.transform(
                            features
                        )

                        prediction = classifier.predict(
                            scaled_features
                        )[0]

                    st.markdown(
                        """
                        <div class="result-card">
                            <h3>🧠 Prediction Result</h3>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Interpret numeric labels only when the saved
                    # classifier's class labels confirm the mapping.
                    class_names = {
                        0: "Mild Demented",
                        1: "Moderate Demented",
                        2: "Non-Demented",
                        3: "Very Mild Demented",
                    }

                    if hasattr(classifier, "classes_"):

                        try:

                            predicted_label = int(prediction)

                            actual_labels = set(
                                int(value)
                                for value in classifier.classes_
                            )

                            if (
                                actual_labels == {0, 1, 2, 3}
                                and predicted_label in class_names
                            ):

                                predicted_name = class_names[
                                    predicted_label
                                ]

                                st.success(
                                    f"### Predicted Class: {predicted_name}"
                                )

                            else:

                                st.warning(
                                    f"Model output: {prediction}. "
                                    "The saved classifier's class-label "
                                    "mapping must be verified before "
                                    "displaying a class name."
                                )

                        except (TypeError, ValueError):

                            st.warning(
                                f"Model output: {prediction}. "
                                "Verify the saved class-label mapping."
                            )

                    else:

                        st.warning(
                            f"Model output: {prediction}. "
                            "Verify the class-label mapping."
                        )

                    st.caption(
                        "This is an experimental model output, "
                        "not a clinical diagnosis."
                    )

                except Exception as error:

                    st.error(
                        f"Prediction failed: {error}"
                    )

                    st.info(
                        "Check the saved model, scaler, preprocessing, "
                        "and Swin feature extractor compatibility."
                    )

        except Exception:

            st.error(
                "The uploaded file could not be opened. "
                "Please select a valid MRI image."
            )

    else:

        st.info(
            "Choose an MRI image above to begin."
        )


# ============================================================
# TAB 2: PROJECT RESULTS
# ============================================================

with tab_results:

    st.markdown(
        """
        <div class="hero-card">
            <h2>📊 Model Evaluation Dashboard</h2>
            <p>
                Reported performance of the selected
                Swin Transformer-Tiny + RBF-SVM experiment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "The following values come from the saved evaluation report "
        "dated October 13, 2026. They have not been independently "
        "re-evaluated by this dashboard."
    )

    # Overall metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "🎯 Test Accuracy",
        "96.95%",
    )

    col2.metric(
        "⚖️ Balanced Accuracy",
        "97.66%",
    )

    col3.metric(
        "🏆 Macro F1-score",
        "97.72%",
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Precision",
        "97.77%",
    )

    col5.metric(
        "Recall",
        "97.66%",
    )

    col6.metric(
        "Test Images",
        "1,280",
    )

    st.divider()

    # Per-class results
    st.subheader("📋 Per-Class Performance")

    class_results = pd.DataFrame({
        "Class": [
            "Mild Demented",
            "Moderate Demented",
            "Non-Demented",
            "Very Mild Demented",
        ],
        "Precision (%)": [
            97.62,
            100.00,
            97.33,
            96.13,
        ],
        "Recall (%)": [
            97.04,
            100.00,
            97.48,
            96.13,
        ],
        "F1-score (%)": [
            97.33,
            100.00,
            97.40,
            96.13,
        ],
        "Test Support": [
            169,
            11,
            635,
            465,
        ],
    })

    st.dataframe(
        class_results,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Confusion matrix
    st.subheader("🔎 Confusion Matrix")

    st.write(
        "Rows represent actual classes. Columns represent predicted classes."
    )

    confusion_matrix = pd.DataFrame(
        [
            [164, 0, 2, 3],
            [0, 11, 0, 0],
            [1, 0, 619, 15],
            [3, 0, 15, 447],
        ],
        index=[
            "Actual Mild",
            "Actual Moderate",
            "Actual Non-Demented",
            "Actual Very Mild",
        ],
        columns=[
            "Predicted Mild",
            "Predicted Moderate",
            "Predicted Non-Demented",
            "Predicted Very Mild",
        ],
    )

    st.dataframe(
        confusion_matrix.style.background_gradient(
            cmap="Purples"
        ),
        use_container_width=True,
    )

    st.caption(
        "Diagonal cells represent correct predictions. "
        "Off-diagonal cells represent misclassifications."
    )

    col_correct, col_incorrect = st.columns(2)

    col_correct.metric(
        "✅ Correct Predictions",
        "1,241",
    )

    col_incorrect.metric(
        "❌ Incorrect Predictions",
        "39",
    )

    st.progress(
        0.9695,
        text="Reported test accuracy: 96.95%",
    )

    st.divider()

    # Limitations
    st.subheader("⚠️ Evaluation Limitations")

    st.warning(
        """
        - The Moderate Demented test class contains only 11 images.
        - The original training pipeline and split-to-image mapping
          require independent verification.
        - External-dataset performance has not been established.
        - These experimental results do not establish clinical
          diagnostic performance.
        """
    )

    st.caption(
        "Sources: results/final_evaluation_report.txt and "
        "results/model_metrics.csv. Report date: October 13, 2026."
    )


# ============================================================
# TAB 3: PROJECT INFORMATION
# ============================================================

with tab_info:

    st.markdown(
        """
        <div class="hero-card">
            <h2>📘 About the Project</h2>
            <p>
                An MRI-Based Swin Transformer Approach for
                Automated Alzheimer's Disease Recognition
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("🎯 Project Objective")

    st.write(
        "To investigate MRI image classification using multi-stage "
        "Swin Transformer-Tiny features and an RBF Support Vector "
        "Machine classifier."
    )

    st.divider()

    st.subheader("⚙️ Proposed Methodology")

    st.markdown(
        """
        1. **MRI Input:** Select an MRI image.
        2. **Preprocessing:** Convert the image to three channels
           and resize it to 224 × 224.
        3. **Feature Extraction:** Extract features from four
           Swin Transformer-Tiny stages.
        4. **Feature Fusion:** Combine the features into a
           1,440-dimensional representation.
        5. **Feature Scaling:** Apply the saved StandardScaler.
        6. **Classification:** Use the saved RBF-SVM classifier.
        7. **Evaluation:** Review the saved accuracy, precision,
           recall, F1-score, and confusion matrix.
        """
    )

    st.divider()

    st.subheader("🗂️ Dataset Categories")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="info-card">
                <h3>🟣 Mild Demented</h3>
                <p>One of the four dataset classification categories.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="info-card">
                <h3>🔵 Non-Demented</h3>
                <p>One of the four dataset classification categories.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="info-card">
                <h3>🩷 Moderate Demented</h3>
                <p>One of the four dataset classification categories.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="info-card">
                <h3>🟢 Very Mild Demented</h3>
                <p>One of the four dataset classification categories.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("💻 Technologies Used")

    st.markdown(
        """
        - Python
        - PyTorch and TorchVision
        - Swin Transformer-Tiny via timm
        - StandardScaler
        - RBF Support Vector Machine
        - Streamlit
        - NumPy and Pandas
        - Pillow
        """
    )

    st.divider()

    st.subheader("⚠️ Research Disclaimer")

    st.warning(
        "This application is an academic research prototype. "
        "It must not be used to diagnose Alzheimer's disease or "
        "make treatment decisions. Independent validation and "
        "external-dataset testing are still needed."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-card">
        Alzheimer's Disease MRI Recognition |
        Swin Transformer-Tiny + RBF-SVM |
        Academic Research Prototype
    </div>
    """,
    unsafe_allow_html=True,
)