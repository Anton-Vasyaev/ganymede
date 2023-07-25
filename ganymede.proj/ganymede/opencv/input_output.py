# python
import os

# 3rd party
import cv2 as cv  # type: ignore
import numpy as np
from pathlib import Path


def imread(path: str, flags: int = cv.IMREAD_UNCHANGED) -> np.ndarray:
    if not os.path.exists(path):
        raise Exception(f"path not exist:{path}")

    with open(path, "rb") as fh:
        data = fh.read()
        data = np.frombuffer(data, dtype=np.uint8)

        img = cv.imdecode(data, flags)

        if img is None:
            raise Exception(f"failed to decode img:{path}")

        return img


def imwrite(img: np.ndarray, path: str, mkdir: bool = True):
    path_p = Path(path)
    parent = path_p.parent

    if mkdir:
        if parent != Path("."):
            parent.mkdir(parents=True, exist_ok=True)

    ext = path_p.suffix

    ret, buf = cv.imencode(ext, img)
    if not ret:
        raise Exception(f"failed to encode img with ext:{ext}")

    with open(path, "wb") as fh:
        fh.write(bytearray(buf))
