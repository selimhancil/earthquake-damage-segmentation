"""Analiz çıktılarını diske yazan adapter (IOutputWriter).

Her analiz için data/test_results/ altına aynı ön ekle dört dosya yazılır:
    <ad>_<zaman>_annotated.png
    <ad>_<zaman>_mask.png
    <ad>_<zaman>_metrics.csv
    <ad>_<zaman>_summary.json
"""

from __future__ import annotations

import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

import cv2
import numpy as np

from src.application.dtos import SavedOutputsDTO
from src.application.interfaces import IOutputWriter

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "id",
    "object_class",
    "confidence",
    "pixel_area",
    "area_m2",
    "is_partial",
    "bbox_x_min",
    "bbox_y_min",
    "bbox_x_max",
    "bbox_y_max",
]


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned[:80] or "analysis"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"JSON'a çevrilemeyen tip: {type(value).__name__}")


class FileOutputWriter(IOutputWriter):
    def __init__(
        self,
        results_dir: Path,
        now_fn: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._results_dir = Path(results_dir)
        self._now = now_fn

    def save(
        self,
        base_name: str,
        annotated_image: np.ndarray,
        mask_image: np.ndarray,
        region_rows: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> SavedOutputsDTO:
        self._results_dir.mkdir(parents=True, exist_ok=True)
        prefix = self._unique_prefix(_safe_name(base_name))

        paths = SavedOutputsDTO(
            annotated_image_path=self._results_dir / f"{prefix}_annotated.png",
            mask_image_path=self._results_dir / f"{prefix}_mask.png",
            metrics_csv_path=self._results_dir / f"{prefix}_metrics.csv",
            summary_json_path=self._results_dir / f"{prefix}_summary.json",
        )

        self._write_png(paths.annotated_image_path, annotated_image, is_rgb=True)
        self._write_png(paths.mask_image_path, mask_image, is_rgb=False)
        self._write_csv(paths.metrics_csv_path, region_rows)

        summary_out = dict(summary)
        summary_out["outputs"] = {
            "annotated_image": paths.annotated_image_path.name,
            "mask_image": paths.mask_image_path.name,
            "metrics_csv": paths.metrics_csv_path.name,
            "summary_json": paths.summary_json_path.name,
        }
        with paths.summary_json_path.open("w", encoding="utf-8") as fh:
            json.dump(summary_out, fh, ensure_ascii=False, indent=2, default=_json_default)

        logger.info("Çıktılar yazıldı: %s_*", self._results_dir / prefix)
        return paths

    # ------------------------------------------------------------------ #
    def _unique_prefix(self, base: str) -> str:
        stamp = self._now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{base}_{stamp}"
        counter = 1
        while (self._results_dir / f"{prefix}_summary.json").exists():
            counter += 1
            prefix = f"{base}_{stamp}_{counter}"
        return prefix

    @staticmethod
    def _write_png(path: Path, image: np.ndarray, is_rgb: bool) -> None:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            raise ValueError(f"PNG için uint8 bekleniyor, gelen: {array.dtype}")
        if is_rgb and array.ndim == 3 and array.shape[2] == 3:
            array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".png", array)
        if not ok:
            raise OSError(f"PNG kodlanamadı: {path.name}")
        path.write_bytes(encoded.tobytes())

    @staticmethod
    def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        extra = [k for row in rows for k in row if k not in CSV_COLUMNS]
        columns = CSV_COLUMNS + sorted(set(extra))
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
