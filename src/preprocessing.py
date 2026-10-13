from pathlib import Path
from collections import Counter
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
import hashlib

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"

CLASS_NAMES = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented",
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

def inspect_dataset():
    print("=" * 55)
    print("ALZHEIMER MRI DATASET INSPECTION")
    print("=" * 55)

    total = 0

    for class_name in CLASS_NAMES:
        folder = DATASET_DIR / class_name

        if not folder.exists():
            print(f"MISSING CLASS FOLDER: {folder}")
            continue

        images = [
            p for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        valid = 0
        invalid = 0
        dimensions = Counter()

        for path in images:
            try:
                with Image.open(path) as image:
                    image.verify()

                with Image.open(path) as image:
                    dimensions[image.size] += 1

                valid += 1

            except (OSError, UnidentifiedImageError):
                invalid += 1

        total += valid

        print(f"\nClass: {class_name}")
        print(f"Image files: {len(images)}")
        print(f"Readable images: {valid}")
        print(f"Unreadable images: {invalid}")
        print(f"Dimensions: {dict(dimensions)}")

    print(f"\nTOTAL READABLE IMAGES: {total}")

    if total == 0:
        print("Check that the four class folders contain images.")

if __name__ == "__main__":
    inspect_dataset()
