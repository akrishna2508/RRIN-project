"""
scripts/kaggle_setup.py
========================
Automatically downloads all required datasets from Kaggle.

HOW TO USE (for beginners - step by step):

STEP 1 — Get your Kaggle API key:
  a) Go to https://www.kaggle.com and log in (create a free account if needed)
  b) Click your profile picture (top-right) → "Settings"
  c) Scroll down to the "API" section
  d) Click "Create New API Token" → a file called kaggle.json downloads
  e) Open that file. It looks like: {"username":"yourname","key":"abc123..."}

STEP 2 — Set environment variables:
  On Mac/Linux (paste in your terminal):
    export KAGGLE_USERNAME=yourname
    export KAGGLE_KEY=abc123...

  On Windows (paste in Command Prompt):
    set KAGGLE_USERNAME=yourname
    set KAGGLE_KEY=abc123...

  OR: Copy .env.example to .env and fill in the values there.

STEP 3 — Accept dataset licenses on Kaggle:
  You MUST click "Accept" on the dataset page before downloading.
  Required links (open each in your browser and accept):
    - EyePACS:  https://www.kaggle.com/c/diabetic-retinopathy-detection/data
    - APTOS:    https://www.kaggle.com/c/aptos2019-blindness-detection/data
    - RFMiD:    https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification

STEP 4 — Run this script:
  python scripts/kaggle_setup.py

STEP 5 — Then run training:
  python main.py
"""

import os
import sys
import subprocess


# ---- Dataset definitions -----------------------------------

KAGGLE_DATASETS = [
    {
        "name":         "eyepacs",
        "type":         "competition",           # Kaggle competition dataset
        "slug":         "diabetic-retinopathy-detection",
        "target_path":  "data/eyepacs",
        "license_url":  "https://www.kaggle.com/c/diabetic-retinopathy-detection/data",
        "size_gb":      ~80,                      # approximate
        "description":  "EyePACS — 88,702 fundus images, Diabetic Retinopathy grading",
    },
    {
        "name":         "aptos",
        "type":         "competition",
        "slug":         "aptos2019-blindness-detection",
        "target_path":  "data/aptos",
        "license_url":  "https://www.kaggle.com/c/aptos2019-blindness-detection/data",
        "size_gb":      9,
        "description":  "APTOS 2019 — 5,590 images, Diabetic Retinopathy severity",
    },
    {
        "name":         "rfmid",
        "type":         "dataset",               # Kaggle dataset (not competition)
        "slug":         "andrewmvd/retinal-disease-classification",
        "target_path":  "data/rfmid",
        "license_url":  "https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification",
        "size_gb":      3,
        "description":  "RFMiD — 3,200 images, 45 retinal disease categories",
    },
    {
        "name":         "odir",
        "type":         "dataset",
        "slug":         "andrewmvd/ocular-disease-recognition-odir5k",
        "target_path":  "data/odir",
        "license_url":  "https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k",
        "size_gb":      7,
        "description":  "ODIR-5K — 10,000 fundus images, 8 disease categories",
    },
]

# IDRiD and MESSIDOR-2 must be downloaded manually from their official sites
MANUAL_DOWNLOADS = [
    {
        "name":        "idrid",
        "url":         "https://idrid.grand-challenge.org/",
        "target_path": "data/idrid",
        "description": "IDRiD — 516 images, DR and Diabetic Macular Edema grading",
        "instructions": (
            "1. Go to https://idrid.grand-challenge.org/\n"
            "2. Register for a free account\n"
            "3. Go to the 'Disease Grading' task page\n"
            "4. Download the training and test image archives\n"
            "5. Extract them to data/idrid/"
        ),
    },
    {
        "name":        "messidor2",
        "url":         "https://www.adcis.net/en/third-party/messidor2/",
        "target_path": "data/messidor2",
        "description": "MESSIDOR-2 — 1,748 images (ALWAYS held out, never for training)",
        "instructions": (
            "1. Go to https://www.adcis.net/en/third-party/messidor2/\n"
            "2. Fill in the registration form (free academic use)\n"
            "3. You will receive a download link by email\n"
            "4. Extract the images to data/messidor2/\n"
            "   (These images are NEVER used for training — only for cross-domain evaluation)"
        ),
    },
]


# ---- Setup functions ---------------------------------------

def check_kaggle_credentials() -> bool:
    """Check that Kaggle API credentials are set."""
    username = os.environ.get("KAGGLE_USERNAME")
    key      = os.environ.get("KAGGLE_KEY")

    if not username or not key:
        # Try loading from ~/.kaggle/kaggle.json
        kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
        if os.path.exists(kaggle_json):
            print(f"  Found credentials at {kaggle_json}")
            return True

        # Try loading from .env file
        env_path = ".env"
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("KAGGLE_USERNAME="):
                        os.environ["KAGGLE_USERNAME"] = line.split("=", 1)[1]
                    elif line.startswith("KAGGLE_KEY="):
                        os.environ["KAGGLE_KEY"] = line.split("=", 1)[1]

        username = os.environ.get("KAGGLE_USERNAME")
        key      = os.environ.get("KAGGLE_KEY")

        if not username or not key or "PASTE_YOUR" in (username + key):
            print(
                "\n ERROR: Kaggle credentials not found!\n"
                "  Please follow these steps:\n"
                "  1. Go to https://www.kaggle.com → Settings → API → Create New Token\n"
                "  2. Open .env.example, copy it to .env, and fill in KAGGLE_USERNAME and KAGGLE_KEY\n"
                "  3. Run this script again.\n"
            )
            return False

    # Write to ~/.kaggle/kaggle.json (the format kaggle CLI expects)
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(kaggle_json_path, "w") as f:
        import json
        json.dump({"username": username, "key": key}, f)
    os.chmod(kaggle_json_path, 0o600)   # Protect from other users

    print(f"  Kaggle credentials written to {kaggle_json_path}")
    return True


def download_kaggle_competition(slug: str, target_path: str) -> bool:
    """Download a Kaggle competition dataset."""
    os.makedirs(target_path, exist_ok=True)
    cmd = [
        sys.executable, "-m", "kaggle",
        "competitions", "download",
        "-c", slug,
        "-p", target_path,
        "--unzip"
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def download_kaggle_dataset(slug: str, target_path: str) -> bool:
    """Download a Kaggle public dataset."""
    os.makedirs(target_path, exist_ok=True)
    cmd = [
        sys.executable, "-m", "kaggle",
        "datasets", "download",
        "-d", slug,
        "-p", target_path,
        "--unzip"
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def show_manual_download_instructions():
    """Print instructions for datasets that cannot be auto-downloaded."""
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD REQUIRED FOR THESE DATASETS:")
    print("="*60)
    for ds in MANUAL_DOWNLOADS:
        print(f"\n  [{ds['name'].upper()}] — {ds['description']}")
        print(f"  URL: {ds['url']}")
        print(f"  Target folder: {ds['target_path']}")
        print(f"  Instructions:\n" + "\n".join(f"    {l}" for l in ds["instructions"].split("\n")))
    print()


def main():
    print("\n" + "="*60)
    print("RRIN Dataset Downloader")
    print("="*60)

    # Check credentials
    print("\n[1/3] Checking Kaggle credentials...")
    if not check_kaggle_credentials():
        sys.exit(1)
    print("  Credentials OK.\n")

    # Download Kaggle datasets
    print("[2/3] Downloading Kaggle datasets...")
    print(
        "\n  IMPORTANT: You must have accepted the dataset licences on Kaggle's website.\n"
        "  If a download fails with a 403 error, open the URL listed above\n"
        "  in your browser, click 'Accept', and re-run this script.\n"
    )

    failed = []
    for ds in KAGGLE_DATASETS:
        print(f"\n  Downloading {ds['name']} (~{ds['size_gb']}GB)...")
        print(f"  Licence page: {ds['license_url']}")

        if os.path.isdir(ds["target_path"]) and len(os.listdir(ds["target_path"])) > 0:
            print(f"  Skipping — already exists at {ds['target_path']}")
            continue

        if ds["type"] == "competition":
            ok = download_kaggle_competition(ds["slug"], ds["target_path"])
        else:
            ok = download_kaggle_dataset(ds["slug"], ds["target_path"])

        if ok:
            print(f"  Downloaded to {ds['target_path']}")
        else:
            print(f"  FAILED for {ds['name']}. Check the licence URL above.")
            failed.append(ds["name"])

    # Show manual download instructions
    print("\n[3/3] Manual downloads required:")
    show_manual_download_instructions()

    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    if failed:
        print(f"  Failed downloads: {failed}")
        print("  Fix the errors above and re-run this script.")
    else:
        print("  All Kaggle datasets downloaded successfully!")
    print()
    print("  NEXT STEPS:")
    print("  1. Complete the manual downloads above (IDRiD and MESSIDOR-2)")
    print("  2. Update config.yaml → set the correct paths under 'dataset_paths:'")
    print("  3. Run training:  python main.py")
    print()


if __name__ == "__main__":
    main()
