"""Tek use case: AnalyzeSatelliteImageUseCase.

Orkestrasyon (şartname 3.4):
    1. Dosya validasyonu (uzantı, boyut, boş dosya)
    2. Görüntüyü oku (ImageData + SpatialMetadata)
    3. Kalibrasyonu belirle (GeoTIFF > kullanıcı GSD > bilinmiyor)
    4. Deprem SONRASI görüntüde segmentasyon
    5. Eşiği geçen maskeleri ölç (piksel alanı, union, geçerli AOI)
    6. Domain aggregate'ini kur, kuralları uygula
    7. Görselleştir, 8. kaydet, 9. DTO döndür

Bu modül YOLO / OpenCV / Streamlit / NumPy import ETMEZ.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePath
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from src.application.dtos import (
    AnalysisOutputDTO,
    AnalysisRequestDTO,
    ImageData,
    RegionMetricDTO,
    SegmentationInstance,
)
from src.application.interfaces import (
    IImageLoader,
    IMeasurementService,
    IOutputWriter,
    ISegmentationService,
    IVisualizationService,
)
from src.domain.entities import (
    AnalysisResult,
    AoiAssumption,
    Calibration,
    DamageRegion,
    MeasurementSettings,
    ObjectClass,
)
from src.domain.exceptions import (
    DomainRuleViolationError,
    InvalidImageError,
    ModelWeightsUnavailableError,
)

logger = logging.getLogger(__name__)

# Uyarı kodları: arayüz ve summary.json bu ön ekleri kullanır.
W_NO_DAMAGE = "NO_DAMAGE_DETECTED"
W_CALIBRATION = "CALIBRATION_UNAVAILABLE"
W_GSD_IGNORED = "USER_GSD_IGNORED"
W_GEOREF_NOT_METRIC = "GEOREFERENCE_NOT_METRIC"
W_PARTIAL = "PARTIAL_REGIONS"
W_OVERLAP = "OVERLAPPING_REGIONS"
W_EMPTY_MASK = "EMPTY_MASK_SKIPPED"
W_SINGLE_DATE = "SINGLE_DATE_MODEL"
W_LOCATION = "LOCATION_NOT_VERIFIED"
W_PRE_IMAGE = "PRE_IMAGE_DISPLAY_ONLY"
W_AOI_ASSUMED = "AOI_FULL_IMAGE_ASSUMED"
W_DEMO = "DEMO_ONLY"
DEMO_FILE_PREFIX = "DEMO_"


@dataclass(frozen=True)
class AnalysisLimits:
    """Uygulama validasyon sınırları; config.py'den composition root'ta doldurulur."""

    default_confidence_threshold: float = 0.70
    max_image_mb: float = 15.0
    max_image_side_px: int = 4096
    allowed_extensions: FrozenSet[str] = frozenset({".jpg", ".jpeg", ".png"})


class AnalyzeSatelliteImageUseCase:
    def __init__(
        self,
        image_loader: IImageLoader,
        segmentation_service: ISegmentationService,
        measurement_service: IMeasurementService,
        visualization_service: IVisualizationService,
        output_writer: IOutputWriter,
        limits: AnalysisLimits = AnalysisLimits(),
    ) -> None:
        self._loader = image_loader
        self._segmenter = segmentation_service
        self._measurer = measurement_service
        self._visualizer = visualization_service
        self._writer = output_writer
        self._limits = limits

    # ------------------------------------------------------------------ #
    def execute(self, request: AnalysisRequestDTO) -> AnalysisOutputDTO:
        warnings: List[str] = []

        # 1) Dosya validasyonu
        self._validate_file(request.post_image_bytes, request.post_image_filename)
        if not self._segmenter.is_ready():
            raise ModelWeightsUnavailableError(
                "Eğitilmiş model ağırlıkları bulunamadı. Önce modeli eğitip "
                "weights/best_model.pt dosyasını yerleştirin."
            )

        # 2) Görüntüyü oku
        image = self._loader.load(request.post_image_bytes, request.post_image_filename)
        self._validate_dimensions(image)
        logger.info("Görüntü yüklendi: %s (%dx%d)", image.filename, image.width, image.height)

        pre_image = self._load_pre_image(request, warnings)

        # 3) Kalibrasyon ve ayarlar
        calibration = self._resolve_calibration(image, request.user_gsd_m_per_pixel, warnings)
        threshold = (
            request.confidence_threshold
            if request.confidence_threshold is not None
            else self._limits.default_confidence_threshold
        )
        settings = MeasurementSettings(calibration=calibration, confidence_threshold=threshold)

        # 4) Segmentasyon (yalnızca deprem sonrası görüntü)
        started = time.perf_counter()
        instances = self._segmenter.segment(image, threshold)
        inference_ms = (time.perf_counter() - started) * 1000.0
        logger.info("Çıkarım: %d aday segment, %.1f ms", len(instances), inference_ms)

        # Kural 1: eşik altı adaylar ölçüme ve aggregate'e girmez
        accepted = [i for i in instances if settings.accepts_confidence(i.confidence)]

        # 5) Ölçüm
        measurement = self._measurer.measure(
            image, [i.mask for i in accepted], request.exclude_black_padding
        )
        if len(measurement.instance_pixel_areas) != len(accepted):
            raise DomainRuleViolationError("Ölçüm sonucu maske sayısıyla eşleşmiyor")

        # 6) Domain aggregate
        result = AnalysisResult(
            image_width=image.width,
            image_height=image.height,
            settings=settings,
            aoi_assumption=AoiAssumption(measurement.aoi_assumption),
        )
        kept_instances: List[SegmentationInstance] = []
        for instance, pixel_area in zip(accepted, measurement.instance_pixel_areas):
            if pixel_area <= 0:
                warnings.append(f"{W_EMPTY_MASK}: Boş maskeli bir tahmin atlandı.")
                continue
            region = result.add_region(
                region_id=len(kept_instances) + 1,
                object_class=ObjectClass(instance.class_id),
                confidence=instance.confidence,
                bounding_box=instance.bounding_box,
                pixel_area=int(pixel_area),
            )
            if region is not None:
                kept_instances.append(instance)
        result.set_pixel_counts(measurement.damaged_union_pixels, measurement.valid_aoi_pixels)

        self._collect_result_warnings(result, warnings)
        model = self._segmenter.model_info()
        if model.is_demo:
            warnings.insert(
                0,
                f"{W_DEMO}: Sonuçlar eğitilmiş bir modelden DEĞİL, arayüzü denemek için "
                "üretilmiş sahte (demo) segmentasyondan gelir. Bilimsel, resmî veya "
                "makale amaçlı KULLANMAYIN.",
            )

        # 7) Görselleştirme
        labels = [f"#{r.id} {r.confidence:.2f}" for r in result.regions]
        annotated = self._visualizer.render_overlay(
            image, kept_instances, [r.is_partial for r in result.regions], labels
        )
        mask_image = self._visualizer.render_union_mask(image, kept_instances)

        # 8) Kayıt
        region_dtos = [self._to_region_dto(r, calibration) for r in result.regions]
        summary = self._build_summary(
            image, request, result, region_dtos, inference_ms, warnings, pre_image is not None
        )
        base_name = PurePath(image.filename).stem or "analysis"
        if model.is_demo:
            base_name = DEMO_FILE_PREFIX + base_name
        saved = self._writer.save(
            base_name=base_name,
            annotated_image=annotated,
            mask_image=mask_image,
            region_rows=[self._region_row(d) for d in region_dtos],
            summary=summary,
        )
        logger.info("Çıktılar kaydedildi: %s", saved.summary_json_path)

        # 9) DTO
        return AnalysisOutputDTO(
            annotated_image=annotated,
            original_image=image.pixels,
            pre_image=pre_image.pixels if pre_image is not None else None,
            objects=region_dtos,
            counts=self._counts(result),
            damaged_union_pixels=result.damaged_union_pixels,
            valid_aoi_pixels=result.valid_aoi_pixels,
            damaged_area_ratio=result.damaged_area_ratio(),
            damaged_area_m2=result.calculate_union_damaged_area(),
            not_predicted_damaged_pixels=result.not_predicted_damaged_pixels,
            not_predicted_damaged_area_m2=result.calculate_not_predicted_damaged_area(),
            undamaged_building_area_m2=result.undamaged_building_area_m2,
            gsd_m_per_pixel=calibration.gsd_m_per_pixel,
            pixel_area_m2=calibration.pixel_area_m2,
            calibration_source=calibration.source.value,
            confidence_threshold=threshold,
            aoi_assumption=result.aoi_assumption.value,
            saved_image_path=saved.annotated_image_path,
            saved_mask_path=saved.mask_image_path,
            saved_metrics_path=saved.metrics_csv_path,
            saved_summary_path=saved.summary_json_path,
            inference_time_ms=inference_ms,
            warnings=warnings,
            demo_mode=model.is_demo,
        )

    # ------------------------------------------------------------------ #
    # Validasyon
    # ------------------------------------------------------------------ #
    def _validate_file(self, data: Optional[bytes], filename: Optional[str]) -> None:
        if not filename:
            raise InvalidImageError("Dosya adı boş olamaz.")
        extension = PurePath(filename).suffix.lower()
        if extension not in self._limits.allowed_extensions:
            allowed = ", ".join(sorted(self._limits.allowed_extensions))
            raise InvalidImageError(
                f"Desteklenmeyen dosya uzantısı: '{extension or '(yok)'}'. "
                f"İzin verilenler: {allowed}"
            )
        if not data:
            raise InvalidImageError("Yüklenen dosya boş.")
        size_mb = len(data) / (1024 * 1024)
        if size_mb > self._limits.max_image_mb:
            raise InvalidImageError(
                f"Dosya çok büyük ({size_mb:.1f} MB). Azami boyut "
                f"{self._limits.max_image_mb:g} MB; görüntüyü kırpın veya parçalara bölün."
            )

    def _validate_dimensions(self, image: ImageData) -> None:
        limit = self._limits.max_image_side_px
        if image.width > limit or image.height > limit:
            raise InvalidImageError(
                f"Görüntü çok büyük ({image.width}x{image.height} piksel). Azami kenar "
                f"{limit} piksel; görüntüyü kırpın veya parçalara bölün. Büyük sahnelerin "
                "otomatik bölünmesi bu sürümde desteklenmiyor."
            )

    def _load_pre_image(
        self, request: AnalysisRequestDTO, warnings: List[str]
    ) -> Optional[ImageData]:
        if not request.pre_image_bytes:
            return None
        filename = request.pre_image_filename or "pre_image.png"
        self._validate_file(request.pre_image_bytes, filename)
        pre = self._loader.load(request.pre_image_bytes, filename)
        warnings.append(
            f"{W_PRE_IMAGE}: Deprem öncesi görüntü yalnızca görsel karşılaştırma içindir; "
            "modele verilmedi. Hizalama ve tarih doğrulanmadıkça değişim haritası olarak yorumlamayın."
        )
        return pre

    # ------------------------------------------------------------------ #
    # Kalibrasyon
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_calibration(
        image: ImageData, user_gsd: Optional[float], warnings: List[str]
    ) -> Calibration:
        meta = image.metadata
        if meta.has_georeference and meta.affine is not None:
            if meta.crs_is_projected_metric:
                a, b, _c, d, e, _f = meta.affine
                calibration = Calibration.from_affine(a, b, d, e, crs_is_projected_metric=True)
                if user_gsd is not None:
                    warnings.append(
                        f"{W_GSD_IGNORED}: GeoTIFF georeference bilgisi kullanıldı; "
                        "girilen GSD değeri yok sayıldı."
                    )
                return calibration
            warnings.append(
                f"{W_GEOREF_NOT_METRIC}: Görüntünün CRS'i ({meta.crs or 'bilinmiyor'}) metrik "
                "projected değil; georeference'tan m² hesaplanamadı."
            )
        if user_gsd is not None:
            return Calibration.from_user_gsd(user_gsd)
        warnings.append(
            f"{W_CALIBRATION}: Doğrulanmış kalibrasyon yok; yalnızca piksel alanları "
            "raporlanır, m²/km² gösterilmez."
        )
        return Calibration.unknown()

    # ------------------------------------------------------------------ #
    # Uyarılar ve çıktı biçimleri
    # ------------------------------------------------------------------ #
    @staticmethod
    def _collect_result_warnings(result: AnalysisResult, warnings: List[str]) -> None:
        if not result.has_detections:
            warnings.append(
                f"{W_NO_DAMAGE}: Eşiği geçen hasar segmenti bulunamadı. "
                "Bu, görüntüde hasar olmadığının kanıtı değildir."
            )
        if result.partial_regions:
            warnings.append(
                f"{W_PARTIAL}: {len(result.partial_regions)} bölge görüntü sınırına değiyor; "
                "yalnızca görüntü içindeki gözlenen kısmı ölçüldü."
            )
        if result.has_detections and result.overlap_pixels > 0:
            warnings.append(
                f"{W_OVERLAP}: Tekil bölge alanlarının toplamı union alanından "
                f"{result.overlap_pixels} piksel fazla (çakışma veya AOI dışı kısım). "
                "Toplam alan union üzerinden bir kez sayıldı."
            )
        if result.aoi_assumption == AoiAssumption.FULL_IMAGE:
            warnings.append(
                f"{W_AOI_ASSUMED}: Geçerli alan maskesi yok; görüntünün tamamı AOI kabul edildi."
            )
        warnings.append(
            f"{W_SINGLE_DATE}: Model tek tarihli (deprem sonrası) görüntüyle çalışır; "
            "öncesi/sonrası değişim tespiti yapmaz."
        )
        warnings.append(
            f"{W_LOCATION}: Görüntünün Malatya'ya ait olduğu uygulama tarafından doğrulanmadı."
        )

    @staticmethod
    def _counts(result: AnalysisResult) -> Dict[str, int]:
        return {
            "damaged_region_count": result.region_count,
            "complete_region_count": len(result.complete_regions),
            "partial_region_count": len(result.partial_regions),
        }

    @staticmethod
    def _to_region_dto(region: DamageRegion, calibration: Calibration) -> RegionMetricDTO:
        return RegionMetricDTO(
            id=region.id,
            object_class=region.object_class.label,
            confidence=region.confidence,
            pixel_area=region.pixel_area,
            area_m2=region.get_physical_area(calibration),
            is_partial=region.is_partial,
            bounding_box=region.bounding_box,
        )

    @staticmethod
    def _region_row(dto: RegionMetricDTO) -> Dict[str, Any]:
        x_min, y_min, x_max, y_max = dto.bounding_box
        return {
            "id": dto.id,
            "object_class": dto.object_class,
            "confidence": round(dto.confidence, 4),
            "pixel_area": dto.pixel_area,
            "area_m2": dto.area_m2,
            "is_partial": dto.is_partial,
            "bbox_x_min": x_min,
            "bbox_y_min": y_min,
            "bbox_x_max": x_max,
            "bbox_y_max": y_max,
        }

    def _build_summary(
        self,
        image: ImageData,
        request: AnalysisRequestDTO,
        result: AnalysisResult,
        regions: List[RegionMetricDTO],
        inference_ms: float,
        warnings: List[str],
        pre_image_provided: bool,
    ) -> Dict[str, Any]:
        calibration = result.settings.calibration
        model = self._segmenter.model_info()
        return {
            "demo_mode": model.is_demo,
            "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": {
                "post_image_filename": image.filename,
                "width": image.width,
                "height": image.height,
                "source_format": image.metadata.source_format,
                "crs": image.metadata.crs,
                "pre_image_provided": pre_image_provided,
                "pre_image_used_by_model": False,
            },
            "model": {
                "family": model.model_family,
                "weights_path": model.weights_path,
                "weights_sha256": model.weights_sha256,
                "class_names": list(model.class_names),
            },
            "settings": {
                "confidence_threshold": result.settings.confidence_threshold,
                "exclude_black_padding": request.exclude_black_padding,
            },
            "calibration": {
                "source": calibration.source.value,
                "pixel_area_m2": calibration.pixel_area_m2,
                "user_gsd_m_per_pixel": calibration.gsd_m_per_pixel,
            },
            "aoi": {
                "assumption": result.aoi_assumption.value,
                "valid_aoi_pixels": result.valid_aoi_pixels,
            },
            "metrics": {
                "damaged_region_count": result.region_count,
                "partial_region_count": len(result.partial_regions),
                "damaged_union_pixels": result.damaged_union_pixels,
                "sum_of_region_pixels": result.sum_of_region_pixels,
                "damaged_area_ratio": result.damaged_area_ratio(),
                "damaged_union_area_m2": result.calculate_union_damaged_area(),
                "not_predicted_damaged_pixels": result.not_predicted_damaged_pixels,
                "not_predicted_damaged_area_m2": result.calculate_not_predicted_damaged_area(),
                "average_region_area_pixels": result.calculate_average_region_area_pixels(),
                "undamaged_building_area_m2": result.undamaged_building_area_m2,
                "inference_time_ms": round(inference_ms, 2),
            },
            "definitions": {
                "damaged_area_ratio": (
                    "damaged_union_pixels / valid_aoi_pixels; incelenen geçerli görüntü "
                    "alanındaki tahmini hasar maskesi oranı. Hasarlı bina oranı DEĞİLDİR."
                ),
                "not_predicted_damaged": (
                    "AOI içinde modelin hasar olarak işaretlemediği alan. Hasarsız bina alanı DEĞİLDİR."
                ),
                "damaged_region_count": "Segment/bölge sayısı; bina sayısı DEĞİLDİR.",
            },
            "regions": [self._region_row(r) for r in regions],
            "warnings": list(warnings),
        }


__all__: Tuple[str, ...] = ("AnalysisLimits", "AnalyzeSatelliteImageUseCase")
