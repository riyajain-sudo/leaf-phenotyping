"""Step 2: compute simple plant traits from an image and its binary mask."""
import cv2
import numpy as np

from src.segment import excess_green, resize_max_side

# Yellow/brown range in OpenCV HSV (H: 0-179, S: 0-255, V: 0-255).
# Hue 10-35 covers orange-brown through yellow; green starts around 35+.
# Minimum saturation/value skips grey, white and near-black pixels.
YB_LOWER = np.array([10, 60, 40])
YB_UPPER = np.array([35, 255, 255])


def compute_traits(img: np.ndarray, mask: np.ndarray, health_mask: np.ndarray | None = None) -> dict:
    """Return a dict of plant traits (plain Python int/float, safe for SQLite).

    img:         BGR image (any size; it is resized to match the mask)
    mask:        binary mask (0/255) from segment_plant (ExG). Used for area, coverage,
                 greenness and leaf count.
    health_mask: binary mask (0/255) from segment_leaf_background(method="saturation"),
                 used ONLY for yellow_brown_pct. If None, `mask` is used instead.
    """
    # The mask was made at the resized resolution, so bring the image to the same size
    img = resize_max_side(img)
    inside = mask > 0  # boolean version of the mask

    leaf_area_px = int(inside.sum())
    plant_coverage_pct = leaf_area_px / mask.size * 100  # mask.size = total pixels

    # Greenness: average ExG over plant pixels only (0.0 if the mask is empty)
    exg = excess_green(img)
    greenness_index = float(exg[inside].mean()) if leaf_area_px > 0 else 0.0

    # Leaf count: number of outer contours in the cleaned mask.
    # NOTE: this is only a simple estimate. Overlapping or touching leaves form
    # ONE blob, so they are merged and counted as a single leaf.
    # A watershed split will be added in a later phase to separate them.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_count = len(contours)

    # Health score: share of leaf pixels whose HSV colour is yellow/brown.
    # Why a separate mask: the ExG mask only keeps green pixels, so it drops the
    # brown/yellow diseased tissue and would under-report the score (potato late
    # blight: 6% with ExG vs ~14% with the saturation mask / reference).
    # The saturation mask separates leaf from the plain background regardless of
    # the leaf colour, so diseased tissue stays inside it.
    health = (mask if health_mask is None else health_mask) > 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yb = cv2.inRange(hsv, YB_LOWER, YB_UPPER) > 0  # True where colour is yellow/brown
    health_area = int(health.sum())
    yellow_brown_pct = float((yb & health).sum() / health_area * 100) if health_area > 0 else 0.0

    return {
        "leaf_area_px": leaf_area_px,
        "plant_coverage_pct": float(plant_coverage_pct),
        "greenness_index": greenness_index,
        "leaf_count": int(leaf_count),
        "yellow_brown_pct": yellow_brown_pct,
    }


if __name__ == "__main__":
    # Quick test:  python -m src.traits data/raw/some_image.jpg
    import sys

    from src.segment import load_image, segment_leaf_background, segment_plant

    image = load_image(sys.argv[1])
    traits = compute_traits(
        image, segment_plant(image), segment_leaf_background(image, method="saturation")
    )
    for key, value in traits.items():
        print(f"{key:20s} {value!r}")
