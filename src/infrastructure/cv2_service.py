"""OpenCV / NumPy tabanlı adapter'lar.

- CV2ImageLoader         -> IImageLoader (JPEG / PNG)
- CV2MeasurementService  -> IMeasurementService
- CV2VisualizationService-> IVisualizationService
- resize_mask_nearest    -> Kural 3 yardımcı fonksiyonu
"""

from __future__ import annotations

import logging
from pathlib import PurePath
from typing import Sequence

import cv2
import numpy as np

from src.application.dtos import (
    ImageData,
    MeasurementResultDTO,
    SegmentationInstance,
    SpatialMetadata,
)
from src.application.interfaces import (
    IImageLoader,
    IMeasurementService,
    IVisualizationService,
)
from src.domain.entities import AoiAssumption
from src.domain.exceptions import DimensionMismatchError, InvalidImageError

logger = logging.getLogger(__name__)

# Renkler RGB sırasında
COLOR_DAMAGED = (255, 0, 0)  # kırmızı: hasar bölgesi
COLOR_PARTIAL = (255, 255, 0)  # sarı: kenara değen bölge
OVERLAY_ALPHA = 0.45


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def to_rgb_uint8(array: np.ndarray) -> np.ndarray:
    """1/3/4 kanallı bir diziyi H×W×3 uint8 RGB'ye çevirir.

    - 1 kanal: gri -> RGB
    - 3 kanal: OpenCV BGR -> RGB
    - 4 kanal: BGRA -> RGB (alpha atılır)
    - 2 kanal veya >4 kanal: ikinci kanalın anlamı belirsiz olduğu için reddedilir.
    """
    if array.dtype == np.uint16:
        array = (array / 257.0).round().astype(np.uint8)
    elif array.dtype != np.uint8:
        raise InvalidImageError(
            f"Desteklenmeyen piksel tipi: {array.dtype}. 8 veya 16 bit görüntü yükleyin."
        )

    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    if array.ndim != 3:
        raise InvalidImageError("Görüntü boyutları okunamadı.")
    channels = array.shape[2]
    if channels == 1:
        return cv2.cvtColor(array[:, :, 0], cv2.COLOR_GRAY2RGB)
    if channels == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    if channels == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)
    raise InvalidImageError(
        f"{channels} kanallı görüntü desteklenmiyor. Kanalların anlamı (alpha mı, "
        "spektral bant mı) belirsiz; lütfen RGB görüntü yükleyin."
    )


def resize_mask_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Kural 3: maskeyi H×W boyutuna en yakın komşu ile getirir, bool döndürür."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise DimensionMismatchError(f"Maske 2 boyutlu olmalı, gelen: {mask.shape}")
    if mask.shape == (height, width):
        return mask.astype(bool)
    try:
        resized = cv2.resize(
            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        )
    except cv2.error as exc:  # pragma: no cover - OpenCV iç hatası
        raise DimensionMismatchError(f"Maske yeniden boyutlandırılamadı: {exc}") from exc
    if resized.shape != (height, width):
        raise DimensionMismatchError(
            f"Maske boyutu {resized.shape}, beklenen {(height, width)}"
        )
    return resized.astype(bool)


def _require_mask_shape(mask: np.ndarray, image: ImageData) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.shape != (image.height, image.width):
        raise DimensionMismatchError(
            f"Maske boyutu {mask.shape} görüntü boyutu "
            f"{(image.height, image.width)} ile eşleşmiyor."
        )
    return mask.astype(bool)


# --------------------------------------------------------------------------- #
# Görüntü okuma
# --------------------------------------------------------------------------- #
class CV2ImageLoader(IImageLoader):
    """JPEG / PNG okuyucu. Coğrafi bilgi içermez (source_format = uzantı)."""

    def load(self, data: bytes, filename: str) -> ImageData:
        if not data:
            raise InvalidImageError("Yüklenen dosya boş.")
        buffer = np.frombuffer(data, dtype=np.uint8)
        decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise InvalidImageError(
                f"'{filename}' okunamadı. Dosya bozuk veya desteklenmeyen biçimde olabilir."
            )
        rgb = to_rgb_uint8(decoded)
        height, width = rgb.shape[:2]
        fmt = PurePath(filename).suffix.lower().lstrip(".") or "raster"
        if fmt == "jpg":
            fmt = "jpeg"
        logger.debug("CV2 ile okundu: %s %dx%d", filename, width, height)
        return ImageData(
            pixels=rgb,
            width=width,
            height=height,
            filename=filename,
            metadata=SpatialMetadata(source_format=fmt),
        )


# --------------------------------------------------------------------------- #
# Ölçüm
# --------------------------------------------------------------------------- #
class CV2MeasurementService(IMeasurementService):
    def measure(
        self,
        image: ImageData,
        masks: Sequence[np.ndarray],
        exclude_black_padding: bool,
    ) -> MeasurementResultDTO:
        bool_masks = [_require_mask_shape(m, image) for m in masks]
        areas = [int(np.count_nonzero(m)) for m in bool_masks]

        aoi, assumption = self._valid_aoi(image, exclude_black_padding)
        if bool_masks:
            union = np.logical_or.reduce(np.stack(bool_masks, axis=0), axis=0)
            union_pixels = int(np.count_nonzero(union & aoi))
        else:
            union_pixels = 0

        return MeasurementResultDTO(
            instance_pixel_areas=areas,
            damaged_union_pixels=union_pixels,
            valid_aoi_pixels=int(np.count_nonzero(aoi)),
            aoi_assumption=assumption.value,
        )

    @staticmethod
    def _valid_aoi(image: ImageData, exclude_black_padding: bool):
        """Kural 8: GeoTIFF nodata > kullanıcı padding seçeneği > tüm görüntü."""
        if image.has_nodata_mask and image.nodata_mask is not None:
            aoi = _require_mask_shape(image.nodata_mask, image)
            return aoi, AoiAssumption.GEOTIFF_NODATA
        if exclude_black_padding:
            pixels = np.asarray(image.pixels)
            aoi = np.any(pixels != 0, axis=2)
            return aoi, AoiAssumption.USER_BLACK_PADDING_EXCLUDED
        return np.ones((image.height, image.width), dtype=bool), AoiAssumption.FULL_IMAGE


# --------------------------------------------------------------------------- #
# Görselleştirme
# --------------------------------------------------------------------------- #
class CV2VisualizationService(IVisualizationService):
    def render_overlay(
        self,
        image: ImageData,
        instances: Sequence[SegmentationInstance],
        partial_flags: Sequence[bool],
        labels: Sequence[str],
    ) -> np.ndarray:
        if not (len(instances) == len(partial_flags) == len(labels)):
            raise ValueError("instances, partial_flags ve labels aynı uzunlukta olmalı")

        base = np.asarray(image.pixels, dtype=np.uint8).copy()
        overlay = base.copy()
        for inst, partial in zip(instances, partial_flags):
            mask = _require_mask_shape(inst.mask, image)
            overlay[mask] = COLOR_PARTIAL if partial else COLOR_DAMAGED
        blended = cv2.addWeighted(overlay, OVERLAY_ALPHA, base, 1 - OVERLAY_ALPHA, 0)

        thickness = max(1, round(min(image.width, image.height) / 400))
        font_scale = max(0.4, min(image.width, image.height) / 1000)
        for inst, partial, label in zip(instances, partial_flags, labels):
            color = COLOR_PARTIAL if partial else COLOR_DAMAGED
            mask = _require_mask_shape(inst.mask, image).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blended, contours, -1, color, thickness)
            x_min, y_min, x_max, y_max = inst.bounding_box
            cv2.rectangle(blended, (x_min, y_min), (x_max, y_max), color, thickness)
            text_y = y_min - 4 if y_min > 12 else y_max + 14
            cv2.putText(
                blended, label, (x_min, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA,
            )
        return blended

    def render_union_mask(
        self, image: ImageData, instances: Sequence[SegmentationInstance]
    ) -> np.ndarray:
        """Birleşik maske: hasar = 255 (beyaz), diğer = 0. H×W uint8."""
        union = np.zeros((image.height, image.width), dtype=bool)
        for inst in instances:
            union |= _require_mask_shape(inst.mask, image)
        return union.astype(np.uint8) * 255
