"""Application katmanı veri taşıyıcıları.

Bu dosyadaki sınıflar yalnızca veri taşır; hesap veya iş kuralı içermez.
Görüntü ve maske dizileri için `np.ndarray` yalnızca string annotation olarak
kullanılır (from __future__ import annotations + TYPE_CHECKING); Application
katmanı NumPy'a runtime'da bağımlı değildir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - yalnızca statik analiz için
    import numpy as np  # noqa: F401


# --------------------------------------------------------------------------- #
# Use case girdisi
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnalysisRequestDTO:
    """Kullanıcının bir analiz çağrısı için verdiği her şey."""

    post_image_bytes: bytes
    post_image_filename: str
    confidence_threshold: Optional[float] = None  # None -> config varsayılanı
    user_gsd_m_per_pixel: Optional[float] = None
    exclude_black_padding: bool = False
    # Deprem öncesi görüntü YALNIZCA görsel karşılaştırma içindir; modele verilmez.
    pre_image_bytes: Optional[bytes] = None
    pre_image_filename: Optional[str] = None


# --------------------------------------------------------------------------- #
# Port'lar arasında taşınan ara veriler
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpatialMetadata:
    """Görüntünün coğrafi bilgisi. Görselleştirme normalizasyonu bunu değiştirmez.

    affine: (a, b, c, d, e, f)  ->  x = a·col + b·row + c ; y = d·col + e·row + f
    """

    has_georeference: bool = False
    crs: Optional[str] = None
    crs_is_projected_metric: bool = False
    affine: Optional[Tuple[float, float, float, float, float, float]] = None
    nodata: Optional[float] = None
    source_format: str = "raster"  # "jpeg" | "png" | "geotiff" ...


@dataclass(frozen=True)
class ImageData:
    """Okunmuş, 3 kanala getirilmiş analiz görüntüsü."""

    pixels: np.ndarray  # H×W×3, uint8, RGB
    width: int
    height: int
    filename: str
    metadata: SpatialMetadata = field(default_factory=SpatialMetadata)
    has_nodata_mask: bool = False  # GeoTIFF nodata maskesi okunduysa True
    nodata_mask: Optional[np.ndarray] = None  # H×W bool, True = geçerli piksel


@dataclass(frozen=True)
class SegmentationInstance:
    """Segmentasyon servisinin tek bir tahmini (tam çözünürlüklü maske ile)."""

    class_id: int
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x_min, y_min, x_max, y_max
    mask: np.ndarray  # H×W bool, analiz görüntüsüyle aynı boyut


@dataclass(frozen=True)
class ModelInfo:
    """Sonuç metadata'sına yazılacak model bilgisi."""

    weights_path: str
    weights_sha256: Optional[str] = None
    model_family: str = "yolov8-seg"
    class_names: Tuple[str, ...] = ("damaged_region",)


@dataclass(frozen=True)
class MeasurementResultDTO:
    """Ölçüm servisinin çıktısı. Sayılar tam sayı pikseldir."""

    instance_pixel_areas: List[int]  # her maske için, verilen sırayla
    damaged_union_pixels: int  # logical_or birleşimi ∩ geçerli AOI
    valid_aoi_pixels: int
    aoi_assumption: str  # AoiAssumption değeri


@dataclass(frozen=True)
class SavedOutputsDTO:
    """Kaydedilen dosyaların yolları."""

    annotated_image_path: Path
    mask_image_path: Path
    metrics_csv_path: Path
    summary_json_path: Path


# --------------------------------------------------------------------------- #
# Use case çıktısı
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RegionMetricDTO:
    id: int
    object_class: str  # "damaged_region"
    confidence: float
    pixel_area: int
    area_m2: Optional[float]
    is_partial: bool
    bounding_box: Tuple[int, int, int, int]


@dataclass(frozen=True)
class AnalysisOutputDTO:
    annotated_image: np.ndarray  # RGB
    original_image: np.ndarray  # RGB, deprem sonrası
    pre_image: Optional[np.ndarray]  # yalnızca yan yana gösterim için
    objects: List[RegionMetricDTO]
    counts: Dict[str, int]  # bölge sayıları; bina sayısı DEĞİL
    damaged_union_pixels: int
    valid_aoi_pixels: int
    damaged_area_ratio: float
    damaged_area_m2: Optional[float]
    not_predicted_damaged_pixels: int
    not_predicted_damaged_area_m2: Optional[float]  # hasarsız bina alanı DEĞİL
    undamaged_building_area_m2: Optional[float]  # yalnızca doğrulanmış footprint ile
    gsd_m_per_pixel: Optional[float]  # yalnızca kullanıcı girdiyse
    pixel_area_m2: Optional[float]  # tüm m² hesaplarında kullanılan tek ölçek
    calibration_source: str
    confidence_threshold: float
    aoi_assumption: str
    saved_image_path: Path
    saved_mask_path: Path
    saved_metrics_path: Path
    saved_summary_path: Path
    inference_time_ms: float
    warnings: List[str]
