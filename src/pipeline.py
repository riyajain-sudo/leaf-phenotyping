"""Step 3b: batch pipeline. Usage:  python -m src.pipeline  (see --help for options)."""
import argparse
import logging
import re
from pathlib import Path

from tqdm import tqdm

from src.db import DB_PATH, init_db, insert_image, insert_measurement
from src.segment import load_image, save_overlay, segment_leaf_background, segment_plant
from src.traits import compute_traits

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
# Ground-truth files (e.g. CVPPP plant001_label.png / plant001_fg.png) are skipped
SKIP_WORDS = ("label", "_fg", "mask")


def find_images(folder: Path) -> list[Path]:
    """List image files in the folder, leaving out label/mask files."""
    files = [p for p in sorted(folder.iterdir())
             if p.suffix.lower() in IMAGE_EXTENSIONS
             and not any(word in p.stem.lower() for word in SKIP_WORDS)]
    return files


def plant_id_from_name(path: Path) -> str:
    """Class label kept in the filename: 'tomato_healthy_001.jpg' -> 'tomato_healthy'."""
    return re.sub(r"_\d+$", "", path.stem)


def process_image(path: Path, out_dir: Path, min_area: int) -> dict:
    """Segment one image, save its overlay and return its traits."""
    img = load_image(path)
    mask = segment_plant(img, min_area)  # ExG mask: area, coverage, greenness, leaf count
    health_mask = segment_leaf_background(img, min_area, method="saturation")  # for yellow_brown_pct
    save_overlay(img, mask, out_dir / f"{path.stem}_overlay.png")
    return compute_traits(img, mask, health_mask)


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment plants, extract traits, store them in SQLite.")
    parser.add_argument("--input", default="data/raw", help="folder with plant images")
    parser.add_argument("--output", default="outputs", help="folder for overlays")
    parser.add_argument("--db", default=DB_PATH, help="SQLite database file")
    parser.add_argument("--dataset", default="plantvillage", help="dataset name stored in the images table")
    parser.add_argument("--min-area", type=int, default=200, help="remove blobs smaller than this (pixels)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    input_dir, out_dir = Path(args.input), Path(args.output)
    images = find_images(input_dir) if input_dir.is_dir() else []
    if not images:
        raise SystemExit(f"No images found in {input_dir}/ - put your plant images there first.")

    conn = init_db(args.db)
    processed, failed, areas = 0, [], []

    for path in tqdm(images, desc="Processing", unit="img"):
        try:  # one bad file must not stop the whole batch
            traits = process_image(path, out_dir, args.min_area)
            image_id = insert_image(conn, path.name, args.dataset, plant_id_from_name(path))
            insert_measurement(conn, image_id, traits)
            processed += 1
            areas.append(traits["leaf_area_px"])
        except Exception as err:
            failed.append(path.name)
            logging.error("Failed on %s: %s", path.name, err)

    conn.close()
    print("\n--- Summary ---")
    print(f"Images processed: {processed}")
    print(f"Images failed:    {len(failed)}" + (f" ({', '.join(failed)})" if failed else ""))
    if areas:
        print(f"Mean leaf area:   {sum(areas) / len(areas):.0f} px")


if __name__ == "__main__":
    main()
