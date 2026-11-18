"""Application port'ları (soyut arayüzler).

Infrastructure katmanı bu arayüzleri uygular; use case yalnızca bunları bilir.
Bu dosya YOLO, OpenCV, Rasterio veya Streamlit import ETMEZ.

| Interface              | Sorumluluk                                   | Uygulama                       |
|------------------------|----------------------------------------------|--------------------------------|
| IImageLoader           | Decode, kanal hazırlama, metadata/CRS okuma  | cv2_service, raster_service    |
| ISegmentationService   | Sınıf, confidence, bbox, tam çözünürlük maske| yolo_service                   |
| IMeasurementService    | Maske piksel sayısı, union, geçerli AOI      | cv2_service                    |
| IVisualizationService  | Overlay, bounding box, birleşik maske görseli| cv2_service                    |
| IOutputWriter          | PNG / CSV / JSON kaydı                       | file_service                   |
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

from src.application.dtos import (
    ImageData,
    MeasurementResultDTO,
    ModelInfo,
    SavedOutputsDTO,
    SegmentationInstance,
)

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np  # noqa: F401


class IImageLoader(ABC):
    @abstractmethod
    def load(self, data: bytes, filename: str) -> ImageData:
        """Baytları H×W×3 RGB görüntüye çevirir; coğrafi metadata'yı korur.

        Okunamayan, şüpheli çok bantlı veya desteklenmeyen girdide
        InvalidImageError fırlatır.
        """


class ISegmentationService(ABC):
    @abstractmethod
    def is_ready(self) -> bool:
        """Eğitilmiş ağırlıklar yüklü ve kullanılabilir mi?"""

    @abstractmethod
    def model_info(self) -> ModelInfo:
        """Sonuç metadata'sına yazılacak model bilgisi."""

    @abstractmethod
    def segment(self, image: ImageData, confidence_threshold: float) -> List[SegmentationInstance]:
        """Tek bir (deprem sonrası) görüntüde çıkarım yapar.

        Dönen her maske görüntüyle birebir aynı H×W boyutunda olmalıdır
        (gerekirse INTER_NEAREST ile); aksi hâlde DimensionMismatchError.
        Ağırlık yoksa ModelWeightsUnavailableError.
        """


class IMeasurementService(ABC):
    @abstractmethod
    def measure(
        self,
        image: ImageData,
        masks: Sequence["np.ndarray"],
        exclude_black_padding: bool,
    ) -> MeasurementResultDTO:
        """Maskelerin piksel alanlarını, union ve geçerli AOI piksel sayılarını hesaplar.

        - instance_pixel_areas: her maskenin tüm görüntüdeki piksel sayısı
        - damaged_union_pixels: logical_or(maskeler) ∩ geçerli AOI
        - valid_aoi_pixels: GeoTIFF nodata maskesi; JPEG/PNG'de tüm görüntü,
          yalnızca exclude_black_padding=True ise tam siyah pikseller hariç.
        """


class IVisualizationService(ABC):
    @abstractmethod
    def render_overlay(
        self,
        image: ImageData,
        instances: Sequence[SegmentationInstance],
        partial_flags: Sequence[bool],
        labels: Sequence[str],
    ) -> "np.ndarray":
        """Kırmızı: hasar bölgesi, sarı: kenara değen (partial) bölge. RGB döndürür."""

    @abstractmethod
    def render_union_mask(
        self, image: ImageData, instances: Sequence[SegmentationInstance]
    ) -> "np.ndarray":
        """Tüm kabul edilen maskelerin birleşimini siyah/beyaz görsel olarak döndürür."""


class IOutputWriter(ABC):
    @abstractmethod
    def save(
        self,
        base_name: str,
        annotated_image: "np.ndarray",
        mask_image: "np.ndarray",
        region_rows: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> SavedOutputsDTO:
        """Annotated PNG, maske PNG, metrics.csv ve summary.json dosyalarını kaydeder."""
