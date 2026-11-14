"""Step 1: plant segmentation using the Excess Green index (ExG) + Otsu threshold."""
from pathlib import Path

import cv2
import numpy as np

MAX_SIDE = 800  # longest image side after resizing (pixels)


def load_image(path: str | Path) -> np.ndarray:
    """Read an image from disk as a BGR array (OpenCV's default colour order)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def resize_max_side(img: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    """Shrink the image so its longest side is at most max_side (never enlarge)."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def excess_green(img_bgr: np.ndarray) -> np.ndarray:
    """ExG = 2G - R - B on channels normalised to sum to 1. Plants score high."""
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    total = b + g + r + 1e-6  # tiny value avoids division by zero on black pixels
    return 2 * (g / total) - (r / total) - (b / total)


def segment_plant(img: np.ndarray, min_area: int = 200) -> np.ndarray:
    """Return a binary mask (0/255, uint8) of the plant, at the resized resolution.

    min_area: connected components smaller than this many pixels are removed.
    """
    # 1. Resize so the longest side is <= 800 px
    img = resize_max_side(img)

    # 2. Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # 3. Excess Green index (float, roughly -1..2)
    exg = excess_green(blurred)

    # 4. Otsu needs an 8-bit image, so stretch ExG to 0..255 first
    exg_8bit = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(exg_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5-6. Morphological cleanup + small-component removal
    return clean_mask(mask, min_area)


def clean_mask(mask: np.ndarray, min_area: int = 200) -> np.ndarray:
    """Opening (remove specks), closing (fill small holes), drop components < min_area."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, n):  # label 0 is the background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean


def segment_leaf_background(img: np.ndarray, min_area: int = 200, method: str = "lab") -> np.ndarray:
    """Separate leaf from a plain background, whatever the leaf colour (green, yellow, brown).

    method="saturation": Otsu on the HSV saturation channel (grey background = low saturation).
    method="lab":        distance from the background colour, sampled on the image border,
                         measured in Lab space, then Otsu on that distance.
    """
    img = resize_max_side(img)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    if method == "saturation":
        signal = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)[:, :, 1]  # already 8-bit
    elif method == "lab":
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Border pixels are assumed to be background; the median ignores a leaf touching the edge
        border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        bg_colour = np.median(border, axis=0)
        dist = np.linalg.norm(lab - bg_colour, axis=2)  # Euclidean distance per pixel
        signal = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        raise ValueError(f"Unknown method: {method}")

    _, mask = cv2.threshold(signal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return clean_mask(mask, min_area)


def compute_iou(mask: np.ndarray, reference_mask: np.ndarray) -> float:
    """Intersection over Union of two binary masks (1.0 = identical, 0.0 = no overlap)."""
    a, b = mask > 0, reference_mask > 0
    union = (a | b).sum()
    return float((a & b).sum() / union) if union > 0 else 1.0


def save_overlay(img: np.ndarray, mask: np.ndarray, out_path: str | Path) -> None:
    """Save the image with a green tint over the mask and a red outline around it."""
    img = resize_max_side(img)  # same size as the mask
    green = np.zeros_like(img)
    green[:] = (0, 255, 0)  # BGR green
    tinted = cv2.addWeighted(img, 0.6, green, 0.4, 0)  # 60% original, 40% green

    overlay = img.copy()
    overlay[mask > 0] = tinted[mask > 0]  # tint only the plant pixels

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)  # red outline

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


if __name__ == "__main__":
    # Quick test on one image:  python -m src.segment data/raw/some_image.png
    import sys

    path = Path(sys.argv[1])
    image = load_image(path)
    plant_mask = segment_plant(image)
    out = Path("outputs") / f"{path.stem}_overlay.png"
    save_overlay(image, plant_mask, out)
    print(f"Plant pixels: {int((plant_mask > 0).sum())}")
    print(f"Overlay saved to: {out.resolve()}")
