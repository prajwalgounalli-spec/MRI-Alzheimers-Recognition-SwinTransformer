from pathlib import Path
from collections import defaultdict
from PIL import Image
import hashlib

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_hash(path):
    digest = hashlib.sha256()

    with Image.open(path) as image:
        image = image.convert("RGB")
        digest.update(image.tobytes())

    return digest.hexdigest()


def main():
    hashes = defaultdict(list)
    total = 0
    unreadable = 0

    print("Checking for exact duplicate MRI images...")
    print("Please wait; this may take a few minutes.")

    for path in DATASET_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue

        total += 1

        try:
            digest = image_hash(path)
            hashes[digest].append(path)
        except Exception as error:
            unreadable += 1
            print(f"Could not read: {path} ({error})")

    duplicate_groups = [
        paths for paths in hashes.values() if len(paths) > 1
    ]

    duplicate_extra = sum(
        len(paths) - 1 for paths in duplicate_groups
    )

    print("\n" + "=" * 55)
    print("DUPLICATE IMAGE CHECK")
    print("=" * 55)
    print("Total image files:", total)
    print("Unique image contents:", len(hashes))
    print("Duplicate groups:", len(duplicate_groups))
    print("Extra duplicate files:", duplicate_extra)
    print("Unreadable files:", unreadable)

    if duplicate_groups:
        print("\nFirst 10 duplicate groups:")
        for group in duplicate_groups[:10]:
            print("\nGroup:")
            for path in group:
                print(" ", path.relative_to(PROJECT_DIR))

    report = PROJECT_DIR / "results" / "duplicate_report.txt"
    report.parent.mkdir(parents=True, exist_ok=True)

    with open(report, "w", encoding="utf-8") as file:
        file.write(f"Total image files: {total}\n")
        file.write(f"Unique image contents: {len(hashes)}\n")
        file.write(f"Duplicate groups: {len(duplicate_groups)}\n")
        file.write(f"Extra duplicate files: {duplicate_extra}\n")
        file.write(f"Unreadable files: {unreadable}\n")

        for group in duplicate_groups:
            file.write("\nDuplicate group:\n")
            for path in group:
                file.write(
                    str(path.relative_to(PROJECT_DIR)) + "\n"
                )

    print("\nReport saved to:", report)


if __name__ == "__main__":
    main()
