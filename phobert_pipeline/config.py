"""Đường dẫn và cache đều nằm trong thư mục dự án trên ổ E."""

import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
MODEL_CACHE = CACHE / "huggingface" / "hub"
CHECKPOINT = "vinai/phobert-base"


def configure_runtime():
    """Đặt cache trước khi import Torch/Transformers để không dùng cache ổ C."""
    paths = {
        "HF_HOME": CACHE / "huggingface", "HF_HUB_CACHE": MODEL_CACHE,
        "HUGGINGFACE_HUB_CACHE": MODEL_CACHE,
        "HF_ASSETS_CACHE": CACHE / "huggingface" / "assets",
        "HF_XET_CACHE": CACHE / "huggingface" / "xet",
        "TORCH_HOME": CACHE / "torch", "XDG_CACHE_HOME": CACHE,
        "PIP_CACHE_DIR": CACHE / "pip",
        "TEMP": CACHE / "tmp", "TMP": CACHE / "tmp", "TMPDIR": CACHE / "tmp",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    # Biến cũ có thể trỏ về C; dùng HF_HUB_CACHE để tránh cache cũ/khuyến cáo deprecated.
    os.environ.pop("TRANSFORMERS_CACHE", None)
    # tempfile có thể đã ghi nhớ thư mục mặc định trước khi đổi biến môi trường.
    tempfile.tempdir = str(CACHE / "tmp")
