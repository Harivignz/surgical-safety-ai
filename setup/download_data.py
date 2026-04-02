# setup/download_data.py
"""
Cholec80 Dataset Downloader
Harivignesh — SurgSentinel Prototype
"""
import os
import sys
import zipfile
import shutil
from pathlib import Path


def setup_kaggle_credentials():
    """
    Ensure kaggle.json is in the right place.
    User should have their kaggle.json at:
      Windows: C:/Users/<username>/.kaggle/kaggle.json
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if not kaggle_json.exists():
        print("⚠️  kaggle.json not found at:", kaggle_json)
        print("📌 Steps to fix:")
        print("   1. Go to https://www.kaggle.com/settings")
        print("   2. Click 'Create New Token'")
        print("   3. Move the downloaded kaggle.json to:", kaggle_dir)
        sys.exit(1)

    # Set permissions (important on Windows too)
    os.chmod(kaggle_json, 0o600)
    print(f"✅ Kaggle credentials found at: {kaggle_json}")


def download_cholec80():
    """Download Cholec80 dataset from Kaggle."""
    import kaggle

    data_dir = Path("data/cholec80")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Cholec80 on Kaggle — most common dataset slug
    # NOTE: If this slug fails, search kaggle.com for "cholec80" and update
    dataset_slug = "chewchewfan/cholec80"  # Update if needed

    print(f"📥 Downloading Cholec80 from Kaggle: {dataset_slug}")
    print("   This may take a while (dataset is ~17GB full, we'll use a subset)...")

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset_slug,
        path=str(data_dir),
        unzip=True,
        quiet=False
    )

    print(f"✅ Download complete. Data at: {data_dir}")
    return data_dir


def verify_download(data_dir: Path):
    """Verify that key files exist after download."""
    print("\n📋 Verifying download structure...")

    # Look for video files
    video_files = list(data_dir.rglob("*.mp4")) + list(data_dir.rglob("*.avi"))
    print(f"   Videos found: {len(video_files)}")

    # Look for annotation files (phase labels are .txt files)
    ann_files = list(data_dir.rglob("*phase*.txt")) + list(data_dir.rglob("*Phase*.txt"))
    print(f"   Annotation files found: {len(ann_files)}")

    if video_files:
        print(f"   Sample video: {video_files[0].name}")
    if ann_files:
        print(f"   Sample annotation: {ann_files[0].name}")

    # Print full directory tree (2 levels)
    print("\n📁 Directory structure:")
    for p in sorted(data_dir.rglob("*"))[:30]:
        indent = "  " * (len(p.relative_to(data_dir).parts) - 1)
        print(f"   {indent}{p.name}")

    return video_files, ann_files


if __name__ == "__main__":
    setup_kaggle_credentials()
    data_dir = download_cholec80()
    videos, anns = verify_download(data_dir)

    print("\n✅ Setup complete!")
    print(f"   {len(videos)} video(s) available")
    print(f"   {len(anns)} annotation file(s) available")
    print("\n▶️  Next step: python inference/demo.py")
