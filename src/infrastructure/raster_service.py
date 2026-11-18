"""GeoTIFF okuma (Rasterio) ve uzantıya göre yönlendiren yükleyici.

Rasterio opsiyonel bir bağımlılıktır: yalnızca .tif/.tiff okunurken import edilir.
Görselleştirme için 8-bit'e normalize etme işlemi coğrafi metadata'yı DEĞİŞTİRMEZ;
affine ve CRS dosyadan aynen okunur.
"""

from __future__ import annotations

import logging
from pathlib import PurePath
from typing import Optional, Tuple

import numpy as np

from src.application.dtos import ImageData, SpatialMetadata
from src.application.interfaces import IImageLoader
from src.domain.exceptions import InvalidImageError

logger = logging.getLogger(__name__)

GEOTIFF_EXTENSIONS = frozenset({".tif", ".tiff"})
_METRIC_UNITS = frozenset({"metre", "meter", "metres", "meters", "m"})


# --------------------------------------------------------------------------- #
# Saf yardımcılar (rasterio gerektirmez, test edilebilir)
# --------------------------------------------------------------------------- #
def is_projected_metric(is_projected: bool, linear_units: Optional[str]) -> bool:
    """CRS metre tabanlı projected mi? (ör. EPSG:32637 -> True, EPSG:4326 -> False)"""
    return bool(is_projected) and (linear_units or "").strip().lower() in _METRIC_UNITS


def build_spatial_metadata(
    affine: Optional[Tuple[float, float, float, float, float, float]],
    crs_string: Optional[str],
    crs_is_projected: bool,
    crs_linear_units: Optional[str],
    nodata: Optional[float],
) -> SpatialMetadata:
    """Rasterio'dan okunan değerlerden SpatialMetadata kurar.

    Birim (identity) affine, georeference yok demektir (rasterio CRS'siz
    dosyalarda Affine.identity() döndürür).
    """
    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    has_geo = (
        affine is not None
        and crs_string is not None
        and tuple(float(v) for v in affine) != identity
    )
    return SpatialMetadata(
        has_georeference=has_geo,
        crs=crs_string,
        crs_is_projected_metric=has_geo and is_projected_metric(crs_is_projected, crs_linear_units),
        affine=tuple(float(v) for v in affine) if (has_geo and affine is not None) else None,
        nodata=nodata,
        source_format="geotiff",
    )


def stretch_to_uint8(bands: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """(C, H, W) dizisini görüntüleme/model için uint8'e çevirir.

    uint8 ise aynen döner. Diğer tiplerde her bant geçerli pikseller üzerinden
    %2–%98 aralığına gerilir. Bu yalnızca piksel değerlerini etkiler.
    """
    if bands.dtype == np.uint8:
        return bands
    out = np.zeros(bands.shape, dtype=np.uint8)
    for i, band in enumerate(bands.astype(np.float64)):
        sample = band[valid] if valid is not None and valid.any() else band.ravel()
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            continue
        lo, hi = np.percentile(sample, (2, 98))
        if hi <= lo:
            hi = lo + 1.0
        scaled = np.clip((band - lo) / (hi - lo), 0.0, 1.0) * 255.0
        out[i] = np.nan_to_num(scaled).round().astype(np.uint8)
    return out


# --------------------------------------------------------------------------- #
# GeoTIFF yükleyici
# --------------------------------------------------------------------------- #
class RasterioImageLoader(IImageLoader):
    def load(self, data: bytes, filename: str) -> ImageData:
        try:
            from rasterio.enums import ColorInterp
            from rasterio.errors import RasterioIOError
            from rasterio.io import MemoryFile
        except ImportError as exc:
            raise InvalidImageError(
                "GeoTIFF desteği için 'rasterio' paketi gerekli (pip install rasterio)."
            ) from exc

        if not data:
            raise InvalidImageError("Yüklenen dosya boş.")
        try:
            with MemoryFile(data) as memfile, memfile.open() as ds:
                bands = self._select_rgb_bands(ds, ColorInterp)
                pixels_chw = ds.read(bands)
                valid = ds.dataset_mask() > 0
                has_nodata = ds.nodata is not None or not bool(valid.all())
                crs = ds.crs
                metadata = build_spatial_metadata(
                    affine=tuple(ds.transform)[:6],
                    crs_string=crs.to_string() if crs else None,
                    crs_is_projected=bool(crs and crs.is_projected),
                    crs_linear_units=(crs.linear_units if crs else None),
                    nodata=ds.nodata,
                )
        except RasterioIOError as exc:
            raise InvalidImageError(f"'{filename}' GeoTIFF olarak okunamadı: {exc}") from exc

        pixels_chw = stretch_to_uint8(pixels_chw, valid)
        if pixels_chw.shape[0] == 1:
            pixels_chw = np.repeat(pixels_chw, 3, axis=0)
        rgb = np.ascontiguousarray(np.transpose(pixels_chw, (1, 2, 0)))
        height, width = rgb.shape[:2]
        logger.info(
            "GeoTIFF okundu: %s %dx%d CRS=%s metrik=%s nodata_maskesi=%s",
            filename, width, height, metadata.crs, metadata.crs_is_projected_metric, has_nodata,
        )
        return ImageData(
            pixels=rgb,
            width=width,
            height=height,
            filename=filename,
            metadata=metadata,
            has_nodata_mask=has_nodata,
            nodata_mask=valid if has_nodata else None,
        )

    @staticmethod
    def _select_rgb_bands(ds, ColorInterp) -> list:
        """Okunacak bant indekslerini (1 tabanlı) seçer; belirsiz durumda reddeder."""
        count = ds.count
        interp = list(ds.colorinterp)
        if count == 1:
            return [1]
        if count == 3:
            return [1, 2, 3]
        if count == 4 and interp[3] == ColorInterp.alpha:
            return [1, 2, 3]
        raise InvalidImageError(
            f"{count} bantlı GeoTIFF desteklenmiyor. Bantların anlamı (RGB, NIR, alpha) "
            "belirsiz; lütfen 3 bantlı RGB veya alpha bandı işaretli 4 bantlı dosya yükleyin."
        )


# --------------------------------------------------------------------------- #
# Uzantıya göre yönlendirme
# --------------------------------------------------------------------------- #
class CompositeImageLoader(IImageLoader):
    """.tif/.tiff -> GeoTIFF yükleyici, diğerleri -> varsayılan (OpenCV) yükleyici."""

    def __init__(self, default_loader: IImageLoader, geotiff_loader: Optional[IImageLoader] = None):
        self._default = default_loader
        self._geotiff = geotiff_loader

    def load(self, data: bytes, filename: str) -> ImageData:
        if PurePath(filename).suffix.lower() in GEOTIFF_EXTENSIONS:
            if self._geotiff is None:
                raise InvalidImageError("Bu kurulumda GeoTIFF desteği etkin değil.")
            return self._geotiff.load(data, filename)
        return self._default.load(data, filename)
