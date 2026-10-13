from pathlib import Path
import csv
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
RESULTS_DIR = ROOT / "results"
SPLIT_FILE = RESULTS_DIR / "dataset_split.csv"

CLASS_NAMES = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented",
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

SEED = 42


def collect_images():
    paths = []
    labels = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class folder: {class_dir}"
            )

        images = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not images:
            raise RuntimeError(
                f"No images found in {class_dir}"
            )

        paths.extend(str(p.resolve()) for p in images)
        labels.extend([label] * len(images))

    return paths, labels


def create_split():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    paths, labels = collect_images()

    train_paths, remaining_paths, train_labels, remaining_labels = (
        train_test_split(
            paths,
            labels,
            test_size=0.30,
            random_state=SEED,
            stratify=labels,
        )
    )

    val_paths, test_paths, val_labels, test_labels = (
        train_test_split(
            remaining_paths,
            remaining_labels,
            test_size=0.50,
            random_state=SEED,
            stratify=remaining_labels,
        )
    )

    rows = []

    for split_name, split_paths, split_labels in [
        ("train", train_paths, train_labels),
        ("validation", val_paths, val_labels),
        ("test", test_paths, test_labels),
    ]:
        for path, label in zip(split_paths, split_labels):
            rows.append({
                "path": path,
                "label": label,
                "class_name": CLASS_NAMES[label],
                "split": split_name,
            })

    with SPLIT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "path",
                "label",
                "class_name",
                "split",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 55)
    print("DATASET SPLIT CREATED")
    print("=" * 55)
    print(f"Total images: {len(paths)}")
    print(f"Training: {len(train_paths)}")
    print(f"Validation: {len(val_paths)}")
    print(f"Testing: {len(test_paths)}")

    for split_name in ["train", "validation", "test"]:
        print(f"\n{split_name.upper()} CLASS COUNTS:")

        for class_name in CLASS_NAMES:
            count = sum(
                1 for row in rows
                if row["split"] == split_name
                and row["class_name"] == class_name
            )
            print(f"  {class_name}: {count}")

    print(f"\nSaved split manifest: {SPLIT_FILE}")


if __name__ == "__main__":
    create_split()
