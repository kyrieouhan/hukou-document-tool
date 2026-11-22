"""Fetch the pinned public OCR model weights before running the offline app."""

import argparse
import hashlib
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "source" / "hukou" / "ocr_models"
UPSTREAM = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/"
MODELS = {
    "PP-OCRv5/rec/ch_PP-OCRv5_rec_server.onnx": "e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a",
    "PP-OCRv5/rec/en_PP-OCRv5_rec_mobile.onnx": "c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8",
    "PP-OCRv6/rec/PP-OCRv6_rec_medium.onnx": "eef444829dbbe18d7fea59a3f6eb75647518d2b3a9568d27c92e42940204894b",
}


def sha256(path: Path) -> str:
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def download(relative: str, target: Path, expected: str) -> None:
    with NamedTemporaryFile(dir=MODEL_DIR, suffix=".part", delete=False) as file:
        temporary = Path(file.name)
        try:
            with urlopen(UPSTREAM + relative, timeout=90) as response:
                shutil.copyfileobj(response, file)
            file.close()
            if sha256(temporary) != expected:
                raise ValueError(f"SHA-256 mismatch for {target.name}")
            temporary.replace(target)
        finally:
            file.close()
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify existing files without downloading.")
    args = parser.parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for relative, expected in MODELS.items():
        target = MODEL_DIR / Path(relative).name
        if not target.is_file() or sha256(target) != expected:
            if args.check:
                parser.exit(1, f"Missing or invalid model: {target.name}\n")
            print(f"Downloading {target.name}", flush=True)
            download(relative, target, expected)
        print(f"Verified {target.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
