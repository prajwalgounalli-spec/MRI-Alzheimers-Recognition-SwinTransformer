"""Optional hippocampus-region image processing utilities.

This module does not provide a trained hippocampus detector.
It creates a central-region crop for exploratory visualization only.
"""

from pathlib import Path

from PIL import Image


def crop_central_region(image_path, output_path="results/central_region.png",
                        crop_fraction=0.5):
    """Save a central crop; this is NOT anatomical hippocampus detection."""
    if not 0 < crop_fraction <= 1:
        raise ValueError("crop_fraction must be in (0, 1].")

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    crop_w = int(width * crop_fraction)
    crop_h = int(height * crop_fraction)
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2

    crop = image.crop((left, top, left + crop_w, top + crop_h))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output)
    return str(output)


if __name__ == "__main__":
    print("Central-region crop utility ready.")
    print("This is not a validated hippocampus detector.")
