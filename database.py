"""
Aqua Vision - Logging & Export Engine (SQLite)
Stores detection history, metrics, and exports reports.
"""

import os
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detections.db")


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Creates a connection to the SQLite database."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes the detections table if it does not already exist."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                total_detected INTEGER NOT NULL,
                highest_confidence REAL NOT NULL,
                class_summary TEXT NOT NULL
            )
            """
        )
        conn.commit()


def log_detection(
    filename: str,
    total_detected: int,
    highest_confidence: float,
    class_summary: str = "",
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """
    Logs a detection event into the SQLite database.

    Args:
        filename: Name of the processed image file.
        total_detected: Count of detected objects.
        highest_confidence: Highest confidence score among detections.
        class_summary: Summary text or JSON string of detected classes and counts.
        db_path: Path to SQLite database file.

    Returns:
        int: The inserted record ID.
    """
    init_db(db_path)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO detections (timestamp, filename, total_detected, highest_confidence, class_summary)
            VALUES (?, ?, ?, ?, ?)
            """,
            (now_str, filename, total_detected, round(float(highest_confidence), 4), class_summary),
        )
        conn.commit()
        return cursor.lastrowid


def get_detection_history(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Retrieves all detection history records as a pandas DataFrame.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        pd.DataFrame: DataFrame containing past detection runs.
    """
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT id, timestamp, filename, total_detected, highest_confidence, class_summary FROM detections ORDER BY id DESC",
            conn,
        )
    return df


def clear_history(db_path: str = DEFAULT_DB_PATH) -> None:
    """Clears all records in the detections table."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM detections")
        conn.commit()
