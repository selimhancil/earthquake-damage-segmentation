"""Entegrasyon testleri için ortak fixture'lar.

Burada use case GERÇEK adapter'larla kurulur: CompositeImageLoader
(OpenCV + Rasterio), CV2MeasurementService, CV2VisualizationService,
FileOutputWriter ve YoloSegmentationService. Yalnızca model nesnesi
(ağırlık dosyası eğitilmediği için) sahte olabilir.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import pytest

from src.application.use_cases import AnalysisLimits, AnalyzeSatelliteImageUseCase
from src.infrastructure.cv2_service import (
    CV2ImageLoader,
    CV2MeasurementService,
    CV2VisualizationService,
)
from src.infrastructure.file_service import FileOutputWriter
from src.infrastructure.raster_service import CompositeImageLoader, RasterioImageLoader
from src.infrastructure.yolo_service import YoloSegmentationService

try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None

requires_rasterio = pytest.mark.skipif(rasterio is None, reason="rasterio kurulu değil")

ALL_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})


@pytest.fixture
def weights_file(tmp_path) -> Path:
    path = tmp_path / "weights" / "best_model.pt"
    path.parent.mkdir()
    path.write_bytes(b"placeholder-weights")
    return path


@pytest.fixture
def results_dir(tmp_path) -> Path:
    return tmp_path / "test_results"


@pytest.fixture
def build_pipeline(weights_file, results_dir) -> Callable[..., Any]:
    """Gerçek adapter'larla use case kurar; model_loader dışarıdan verilir."""

    def _build(
        model: Any = None,
        model_loader: Optional[Callable[[str], Any]] = None,
        weights: Optional[Path] = None,
        limits: Optional[AnalysisLimits] = None,
        now: Optional[Callable[[], datetime]] = None,
    ):
        loader = model_loader or (lambda _path: model)
        segmenter = YoloSegmentationService(weights or weights_file, model_loader=loader)
        use_case = AnalyzeSatelliteImageUseCase(
            image_loader=CompositeImageLoader(CV2ImageLoader(), RasterioImageLoader()),
            segmentation_service=segmenter,
            measurement_service=CV2MeasurementService(),
            visualization_service=CV2VisualizationService(),
            output_writer=FileOutputWriter(results_dir, now_fn=now or datetime.now),
            limits=limits or AnalysisLimits(allowed_extensions=ALL_EXTENSIONS),
        )
        return use_case, segmenter

    return _build


def encode_image(array: np.ndarray, ext: str = ".png") -> bytes:
    """RGB diziyi verilen biçimde kodlar (OpenCV BGR beklediği için çevirir)."""
    bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR) if array.ndim == 3 else array
    ok, buf = cv2.imencode(ext, bgr)
    assert ok
    return buf.tobytes()


def make_geotiff(
    path: Path,
    height: int,
    width: int,
    crs: str = "EPSG:32637",
    pixel_size: float = 0.5,
    nodata: Optional[int] = None,
    nodata_cols: int = 0,
) -> bytes:
    """Malatya civarında (UTM 37N) küçük bir RGB GeoTIFF üretir."""
    from rasterio.transform import from_origin

    data = np.full((3, height, width), 110, dtype=np.uint8)
    if nodata is not None and nodata_cols:
        data[:, :, width - nodata_cols:] = nodata
    if crs == "EPSG:4326":
        transform = from_origin(38.30, 38.36, pixel_size, pixel_size)
    else:
        transform = from_origin(400000.0, 4250000.0, pixel_size, pixel_size)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=3, dtype="uint8",
        crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data)
    return path.read_bytes()
