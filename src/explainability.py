"""Basic explainability utilities for MRI image predictions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_input_visualization(image_path, output_path="results/input_mri.png"):
    """Save the uploaded MRI image for visual inspection.

    This is not a model explanation or a medical segmentation.
    """
    image = Image.open(image_path).convert("RGB")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return str(output)


def save_occlusion_sensitivity(
    predict_proba, image_path, output_path="results/occlusion_sensitivity.png",
    patch_size=32, stride=16
):
    """Create a simple occlusion sensitivity map.

    predict_proba must accept a PIL image and return class probabilities.
    Lower values indicate the selected class score after a patch is hidden.
    This is an exploratory visualization, not clinical evidence.
    """
    image = Image.open(image_path).convert("RGB")
    baseline = np.asarray(predict_proba(image))[0]
    class_id = int(np.argmax(baseline))
    pixels = np.asarray(image).copy()
    height, width = pixels.shape[:2]
    scores = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            masked = pixels.copy()
            masked[y:y + patch_size, x:x + patch_size] = 0
            score = np.asarray(
                predict_proba(Image.fromarray(masked))
            )[0][class_id]
            change = float(baseline[class_id] - score)
            scores[y:y + patch_size, x:x + patch_size] += change
            counts[y:y + patch_size, x:x + patch_size] += 1

    heatmap = scores / np.maximum(counts, 1)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(image)
    plt.imshow(heatmap, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()
    return str(output)


if __name__ == "__main__":
    print("Explainability utilities ready.")
    print("Occlusion maps require a compatible probability prediction function.")
