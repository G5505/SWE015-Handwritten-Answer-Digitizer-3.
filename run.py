
# SWE015 - Automatic setup and application launcher

from pathlib import Path
from zipfile import ZipFile, BadZipFile
import importlib
import gdown

BASE_DIR = Path(__file__).resolve().parent

DRIVE_FILE_ID = "1kb03-IMr5MeoRTBEFF8pP3X2RxD7juCi"

MODEL_LOCATIONS = {
    "digits_model.keras": "models",
    "letters_model.keras": "models",
    "swe015_cnn_int8.tflite": "quantized_models",
    "bert_int8_weights.pt": "quantized_models",
}

def model_path(name):
    return BASE_DIR / MODEL_LOCATIONS[name] / name

def valid_model(name):
    path = model_path(name)
    return path.is_file() and path.stat().st_size > 1000

def ensure_models():
    missing = [
        name for name in MODEL_LOCATIONS
        if not valid_model(name)
    ]

    if not missing:
        print("All required models are available!")
        return

    print("Missing models:", ", ".join(missing))
    print("Downloading model weights from Google Drive...")

    archive_path = BASE_DIR / "_downloaded_models.zip"

    try:
        result = gdown.download(
            id=DRIVE_FILE_ID,
            output=str(archive_path),
            quiet=False,
        )

        if not result or not archive_path.exists():
            raise RuntimeError("Google Drive download failed.")

        with ZipFile(archive_path, "r") as archive:
            members = {
                Path(item.filename.replace("\\", "/")).name: item
                for item in archive.infolist()
                if not item.is_dir()
            }

            for name in missing:
                if name not in members:
                    raise FileNotFoundError(
                        f"Missing {name} inside the downloaded ZIP."
                    )

            for name in missing:
                destination = model_path(name)
                destination.parent.mkdir(
                    parents=True, exist_ok=True
                )

                with archive.open(members[name]) as source:
                    with open(destination, "wb") as target:
                        import shutil
                        shutil.copyfileobj(source, target)

                print("Installed:", destination)

        still_missing = [
            name for name in MODEL_LOCATIONS
            if not valid_model(name)
        ]

        if still_missing:
            raise RuntimeError(
                f"Invalid or missing files: {still_missing}"
            )

        print("All model weights installed successfully!")

    finally:
        if archive_path.exists():
            archive_path.unlink()


if __name__ == "__main__":
    ensure_models()

    # Run the original application without modifying it
    import aqq

    print("Starting SWE015 application...")
    aqq.app.run(
        host="127.0.0.1",
        port=5001,
        debug=False
    )
