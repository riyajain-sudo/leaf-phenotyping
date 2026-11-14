"""Step 3a: SQLite storage (built-in sqlite3) for images and their measurements."""
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = "phenotyping.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id           INTEGER PRIMARY KEY,
    filename     TEXT UNIQUE,
    dataset      TEXT,
    plant_id     TEXT,
    capture_date TEXT
);
CREATE TABLE IF NOT EXISTS measurements (
    id                 INTEGER PRIMARY KEY,
    image_id           INTEGER REFERENCES images(id),
    leaf_area_px       REAL,
    plant_coverage_pct REAL,
    greenness_index    REAL,
    leaf_count         INTEGER,
    yellow_brown_pct   REAL,
    processed_at       TEXT
);
"""


def init_db(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Open (and create if needed) the database, make sure both tables exist."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def insert_image(conn: sqlite3.Connection, filename: str, dataset: str,
                 plant_id: str, capture_date: str | None = None) -> int:
    """Insert an image row and return its id. If the filename already exists, reuse that row."""
    # filename is UNIQUE, so INSERT OR IGNORE silently skips duplicates on re-runs
    conn.execute(
        "INSERT OR IGNORE INTO images (filename, dataset, plant_id, capture_date) VALUES (?, ?, ?, ?)",
        (filename, dataset, plant_id, capture_date),
    )
    row = conn.execute("SELECT id FROM images WHERE filename = ?", (filename,)).fetchone()
    conn.commit()
    return int(row[0])


def insert_measurement(conn: sqlite3.Connection, image_id: int, traits: dict) -> None:
    """Store one set of traits for an image, replacing any older measurement of that image."""
    # Delete first so re-running the pipeline never creates duplicate measurements
    conn.execute("DELETE FROM measurements WHERE image_id = ?", (image_id,))
    conn.execute(
        """INSERT INTO measurements (image_id, leaf_area_px, plant_coverage_pct,
               greenness_index, leaf_count, yellow_brown_pct, processed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (image_id, traits["leaf_area_px"], traits["plant_coverage_pct"],
         traits["greenness_index"], traits["leaf_count"], traits["yellow_brown_pct"],
         datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()


def get_measurements_df(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return every measurement joined with its image info as a Pandas DataFrame."""
    query = """
        SELECT i.filename, i.dataset, i.plant_id, i.capture_date,
               m.leaf_area_px, m.plant_coverage_pct, m.greenness_index,
               m.leaf_count, m.yellow_brown_pct, m.processed_at
        FROM measurements m JOIN images i ON i.id = m.image_id
        ORDER BY i.filename
    """
    return pd.read_sql_query(query, conn)
