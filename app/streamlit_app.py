"""Streamlit app for leaf-phenotyping. Run:  streamlit run app/streamlit_app.py

It only reuses the functions in src/ - no pipeline logic is rewritten here.
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Make "from src..." work when Streamlit runs this file from the app/ folder
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.db import DB_PATH, get_measurements_df, init_db  # noqa: E402
from src.pipeline import find_images  # noqa: E402  (skips label/mask files)
from src.segment import compute_iou, save_overlay, segment_leaf_background, segment_plant  # noqa: E402
from src.traits import compute_traits  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
REF_DIR = ROOT / "data" / "reference"
DB_FILE = ROOT / DB_PATH
COMPARISON_PNG = ROOT / "outputs" / "method_comparison.png"

st.set_page_config(page_title="Leaf phenotyping", layout="wide")


def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """OpenCV uses BGR, Streamlit expects RGB."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- PAGE 1
def page_analyze() -> None:
    st.title("Analyze one image")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    raw_images = find_images(RAW_DIR) if RAW_DIR.is_dir() else []
    chosen = st.selectbox("...or pick one from data/raw/", [p.name for p in raw_images]) if raw_images else None

    # An uploaded file wins over the dropdown
    if uploaded is not None:
        img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
        name = None  # not in data/raw/, so there is no reference mask
    elif chosen:
        img = cv2.imread(str(RAW_DIR / chosen), cv2.IMREAD_COLOR)
        name = chosen
    else:
        st.info("Upload an image, or put images in data/raw/.")
        return
    if img is None:
        st.error("Could not read that image.")
        return

    # Same logic as pipeline.py: ExG mask for most traits, saturation mask for yellow/brown
    mask = segment_plant(img)
    health_mask = segment_leaf_background(img, method="saturation")
    traits = compute_traits(img, mask, health_mask)

    # save_overlay writes a file, so write to a temp file and read it back
    with tempfile.TemporaryDirectory() as tmp:
        overlay_path = Path(tmp) / "overlay.png"
        save_overlay(img, mask, overlay_path)
        overlay = cv2.imread(str(overlay_path))

    col1, col2, col3 = st.columns(3)
    col1.image(to_rgb(img), caption="Original", width="stretch")
    col2.image(to_rgb(overlay), caption="ExG mask overlay", width="stretch")
    col3.image(health_mask, caption="Saturation mask (for yellow/brown %)", width="stretch")

    st.subheader("Traits")
    c = st.columns(5)
    c[0].metric("Leaf area (px)", f"{traits['leaf_area_px']:,}")
    c[1].metric("Coverage (%)", f"{traits['plant_coverage_pct']:.1f}")
    c[2].metric("Greenness (ExG)", f"{traits['greenness_index']:.3f}")
    c[3].metric("Blob count", traits["leaf_count"])
    c[4].metric("Yellow/brown (%)", f"{traits['yellow_brown_pct']:.1f}")

    # IoU only when a reference mask exists for this file
    ref_path = REF_DIR / name if name else None
    if ref_path is not None and ref_path.exists():
        ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
        ref_mask = (ref > 10).astype(np.uint8) * 255  # reference: non-black pixels = leaf
        st.metric("IoU vs reference (ExG mask)", f"{compute_iou(mask, ref_mask):.3f}")
        st.caption("The reference is PlantVillage's own segmentation, so IoU is approximate.")


# ---------------------------------------------------------------- PAGE 2
@st.cache_data
def load_measurements(db_file: str, modified: float):
    """Read all measurements. `modified` (file time) makes the cache refresh after a pipeline re-run."""
    conn = init_db(db_file)
    df = get_measurements_df(conn)
    conn.close()
    return df


def page_dashboard() -> None:
    st.title("Dashboard")
    if not DB_FILE.exists():
        st.warning("Database not found. Run `python -m src.pipeline` first.")
        return
    df = load_measurements(str(DB_FILE), DB_FILE.stat().st_mtime)
    if df.empty:
        st.warning("The database has no measurements. Run `python -m src.pipeline` first.")
        return

    by_class = df.groupby("plant_id")[["yellow_brown_pct", "plant_coverage_pct", "greenness_index"]].mean()

    st.subheader("Mean yellow/brown % per class (main result)")
    st.bar_chart(by_class["yellow_brown_pct"])

    # Different scales (percent vs ~0.3), so each gets its own chart
    st.subheader("Mean coverage and greenness per class")
    left, right = st.columns(2)
    left.caption("Plant coverage (%)")
    left.bar_chart(by_class["plant_coverage_pct"])
    right.caption("Greenness index (mean ExG)")
    right.bar_chart(by_class["greenness_index"])

    st.subheader("10 images with the highest yellow/brown %")
    top = df.nlargest(10, "yellow_brown_pct")[["filename", "plant_id", "yellow_brown_pct", "plant_coverage_pct", "greenness_index"]]
    st.dataframe(top.round(2), hide_index=True, width="stretch")


# ---------------------------------------------------------------- PAGE 3
def page_comparison() -> None:
    st.title("Method comparison")
    st.write("Mean IoU against the reference masks (60 images per class), from the earlier evaluation run.")

    # Static numbers copied from that run
    table = {
        "Class": ["tomato healthy", "tomato early blight", "potato late blight", "overall"],
        "ExG": [0.876, 0.754, 0.801, 0.810],
        "Saturation": [0.602, 0.706, 0.695, 0.668],
        "Lab": [0.340, 0.439, 0.440, 0.406],
    }
    st.table(table)

    if COMPARISON_PNG.exists():
        st.image(str(COMPARISON_PNG), caption="Columns: original, reference, ExG, saturation, Lab", width="stretch")

    st.info("Reference masks are PlantVillage's own segmentation, so IoU is approximate.")


# ---------------------------------------------------------------- navigation
page = st.sidebar.radio("Page", ["Analyze", "Dashboard", "Method comparison"])
if page == "Analyze":
    page_analyze()
elif page == "Dashboard":
    page_dashboard()
else:
    page_comparison()
