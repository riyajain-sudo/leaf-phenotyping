# leaf-phenotyping

Phase 1 of a plant phenotyping pipeline. It segments leaves in images, extracts basic
traits, processes a whole folder in batch, and stores the results in SQLite. A small
Streamlit app lets you inspect single images and the results.

## Install

Python 3.10+ is required.

```bash
pip install -r requirements.txt
```

## Data

Put images (`.png`, `.jpg`, `.jpeg`) in `data/raw/`. Files with `label`, `_fg` or `mask`
in the name are treated as ground-truth masks and skipped (never deleted).

The project was developed on three classes of the
[PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset), 60 images each:
tomato healthy, tomato early blight and potato late blight. File names keep the class,
e.g. `tomato_healthy_001.jpg`, and the class is stored in the `plant_id` column.

To get the same data, shallow-clone the repository, then copy the images from
`raw/color/<class>/` into `data/raw/`. Optionally copy the matching images from
`raw/segmented/` into `data/reference/` (same file names) to use as approximate ground
truth for IoU.

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git
```

Any other plant images work too (for example the CVPPP Arabidopsis dataset). The plain
background assumption of the saturation mask (see below) is the main thing to check.

`data/`, `outputs/` and `*.db` are in `.gitignore`.

## Run

Run everything from the project folder.

```bash
python -m src.pipeline
```

For every image this segments the plant, computes the traits, saves an overlay to
`outputs/` and writes the results to `phenotyping.db`. A progress bar is shown, and a
summary (processed, failed, mean leaf area) is printed at the end. Re-running does not
create duplicate rows. A bad file is logged and skipped, it does not stop the batch.

Options: `--input`, `--output`, `--db`, `--dataset`, `--min-area` (see `--help`).

Test one image on its own:

```bash
python -m src.segment data/raw/tomato_healthy_001.jpg   # saves an overlay to outputs/
python -m src.traits  data/raw/tomato_healthy_001.jpg   # prints the traits
```

Streamlit app (run the pipeline first so the Dashboard has data):

```bash
streamlit run app/streamlit_app.py
```

## How it works

| File | Job |
|---|---|
| `src/segment.py` | Load, resize (longest side 800 px max), blur, ExG + Otsu, cleanup, overlay, IoU |
| `src/traits.py` | Compute the five traits from an image and its masks |
| `src/db.py` | SQLite tables `images` and `measurements`, plus a Pandas helper |
| `src/pipeline.py` | Batch script that ties the three together |
| `app/streamlit_app.py` | Analyze, Dashboard and Method comparison pages |

**Two masks are used.**

- **ExG mask** (`segment_plant`): Excess Green `ExG = 2G - R - B` on normalised channels,
  Otsu threshold, opening then closing, remove small blobs. Used for `leaf_area_px`,
  `plant_coverage_pct`, `greenness_index` and `leaf_count`.
- **Saturation mask** (`segment_leaf_background(method="saturation")`): Otsu on the HSV
  saturation channel, which separates a leaf from a plain grey background whatever the
  leaf colour. Used only for `yellow_brown_pct`.

Why two? ExG keeps only green pixels, so it drops brown or yellow diseased tissue. On
potato late blight this gave about 6% yellow/brown, against about 14% with the saturation
mask and the reference.

## Traits

| Trait | Meaning |
|---|---|
| `leaf_area_px` | Number of mask pixels (pixels only, no real-world scale) |
| `plant_coverage_pct` | Mask area / image area * 100 |
| `greenness_index` | Mean ExG inside the ExG mask |
| `leaf_count` | Number of external contours. Overlapping leaves merge into one blob. Watershed is planned for a later phase |
| `yellow_brown_pct` | Share of leaf pixels in a yellow/brown HSV range (a simple health score) |

## Results on the 180 PlantVillage images

Mean per class (60 images each):

| class | leaf area px | coverage % | greenness | yellow/brown % |
|---|---|---|---|---|
| tomato healthy | 24660 | 37.6 | 0.25 | 3.99 |
| tomato early blight | 22590 | 34.5 | 0.32 | 19.17 |
| potato late blight | 25628 | 39.1 | 0.27 | 14.45 |

Mean IoU against the PlantVillage reference masks:

| class | ExG | saturation | Lab |
|---|---|---|---|
| tomato healthy | 0.876 | 0.602 | 0.340 |
| tomato early blight | 0.754 | 0.706 | 0.439 |
| potato late blight | 0.801 | 0.695 | 0.440 |

## Limitations

- `yellow_brown_pct` separates the classes on average, but it varies a lot between single
  images (medians are well below the means, some healthy leaves reach about 30%).
- The reference masks are PlantVillage's own segmentation, so IoU is approximate.
- The saturation mask assumes a plain, low-saturation background. The Lab method (distance
  from the border colour) failed on uneven lighting.
- ExG alone under-counts diseased area. Area and coverage use the ExG mask, so they are
  biased low for diseased leaves.
- PlantVillage images show single leaves on plain backgrounds, so `leaf_count` is almost
  always 1. It only becomes meaningful on rosette images, and then only with watershed.
- `leaf_area_px` is in pixels. There is no scale reference in these images.
