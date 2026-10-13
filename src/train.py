from pathlib import Path
import csv
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, classification_report
from torchvision import transforms

from model import create_model
from preprocessing import CLASS_NAMES

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"

SPLIT_FILE = RESULTS_DIR / "dataset_split.csv"
CHECKPOINT_FILE = MODEL_DIR / "best_swin_model.pth"
HISTORY_FILE = RESULTS_DIR / "swin_training_history.json"
TEST_REPORT_FILE = RESULTS_DIR / "swin_test_report.txt"

SEED = 42
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 0.0001
NUM_WORKERS = 0

ImageFile.LOAD_TRUNCATED_IMAGES = False

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


class MRIDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image_path = Path(record["path"])
        label = int(record["label"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def read_split():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            "Dataset split missing. Run: python .\\src\\dataset.py"
        )

    with SPLIT_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        records = list(csv.DictReader(file))

    for record in records:
        record["label"] = int(record["label"])

    expected = {
        "train", "validation", "test"
    }

    actual = {record["split"] for record in records}

    if not expected.issubset(actual):
        raise ValueError(
            f"Expected train, validation and test splits; found {actual}"
        )

    for record in records:
        if not Path(record["path"]).is_file():
            raise FileNotFoundError(
                f"Image not found: {record['path']}"
            )

        label = record["label"]

        if not 0 <= label < len(CLASS_NAMES):
            raise ValueError(f"Invalid label: {label}")

        if record["class_name"] != CLASS_NAMES[label]:
            raise ValueError(
                f"Class mismatch in split manifest: {record}"
            )

    return records


def make_loader(records, transform, training=False):
    dataset = MRIDataset(records, transform)

    sampler = None

    if training:
        counts = np.bincount(
            [record["label"] for record in records],
            minlength=len(CLASS_NAMES),
        )

        class_weights = 1.0 / np.maximum(counts, 1)

        sample_weights = [
            float(class_weights[record["label"]])
            for record in records
        ]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )


def evaluate(model, loader, criterion=None):
    model.eval()

    total_loss = 0.0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            if criterion is not None:
                total_loss += (
                    criterion(outputs, labels).item()
                    * labels.size(0)
                )

            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            total += labels.size(0)

    loss = total_loss / total if criterion is not None else None
    accuracy = accuracy_score(all_labels, all_predictions)

    return loss, accuracy, all_labels, all_predictions


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SWIN TRANSFORMER-TINY TRAINING")
    print("=" * 60)
    print("Device:", DEVICE)
    print("Classes:", CLASS_NAMES)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)

    records = read_split()

    train_records = [
        r for r in records if r["split"] == "train"
    ]
    val_records = [
        r for r in records if r["split"] == "validation"
    ]
    test_records = [
        r for r in records if r["split"] == "test"
    ]

    print("Training images:", len(train_records))
    print("Validation images:", len(val_records))
    print("Test images:", len(test_records))

    train_loader = make_loader(
        train_records, TRAIN_TRANSFORM, training=True
    )
    val_loader = make_loader(
        val_records, EVAL_TRANSFORM
    )
    test_loader = make_loader(
        test_records, EVAL_TRANSFORM
    )

    print("\nLoading pretrained Swin Transformer-Tiny...")
    print("The first run may download pretrained weights.")

    model = create_model(pretrained=True).to(DEVICE)

    # Stage 1: freeze the pretrained backbone and train the head.
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.head.parameters():
        parameter.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    best_val_accuracy = -1.0
    history = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        seen = 0

        for batch_index, (images, labels) in enumerate(
            train_loader, start=1
        ):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

            if batch_index % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {batch_index}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f}"
                )

        train_loss = running_loss / seen

        val_loss, val_accuracy, _, _ = evaluate(
            model, val_loader, criterion
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
        }

        history.append(epoch_record)

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {val_loss:.4f} | "
            f"Validation accuracy: {val_accuracy * 100:.2f}%"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "num_classes": len(CLASS_NAMES),
                    "epoch": epoch + 1,
                    "validation_accuracy": val_accuracy,
                    "seed": SEED,
                    "image_size": 224,
                },
                CHECKPOINT_FILE,
            )

            print("Best Swin checkpoint saved.")

        with HISTORY_FILE.open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(history, file, indent=4)

    print("\nTraining complete.")
    print("Best validation accuracy:", f"{best_val_accuracy * 100:.2f}%")
    print("Checkpoint:", CHECKPOINT_FILE)

    # Evaluate the selected checkpoint on the held-out test set.
    print("\nLoading best checkpoint for final test evaluation...")

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    _, test_accuracy, y_true, y_pred = evaluate(
        model, test_loader
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4,
    )

    report_text = (
        "SWIN TRANSFORMER-TINY: HELD-OUT TEST RESULTS\n"
        + "=" * 55
        + f"\nTest images: {len(y_true)}"
        + f"\nTest accuracy: {test_accuracy * 100:.2f}%"
        + "\n\nClassification report:\n"
        + report
        + "\n\nCaution: results are from an image-level split. "
        "Patient-level independence has not been established.\n"
    )

    TEST_REPORT_FILE.write_text(
        report_text, encoding="utf-8"
    )

    print("\n", report_text)
    print("Test report saved:", TEST_REPORT_FILE)
    print("Elapsed minutes:", f"{(time.time() - start_time) / 60:.1f}")


if __name__ == "__main__":
    main()
