"""AnalyzeSatelliteImageUseCase birim testleri (şartname bölüm 8 - test_use_case.py).

Gerçek model, OpenCV veya NumPy KULLANILMAZ. Maskeler (x, y) piksel kümeleri
olarak temsil edilir; sahte servisler bu kümeler üzerinde çalışır.
"""

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

import pytest

from src.application.dtos import (
    AnalysisRequestDTO,
    ImageData,
    MeasurementResultDTO,
    ModelInfo,
    SavedOutputsDTO,
    SegmentationInstance,
    SpatialMetadata,
)
from src.application.interfaces import (
    IImageLoader,
    IMeasurementService,
    IOutputWriter,
    ISegmentationService,
    IVisualizationService,
)
from src.application.use_cases import AnalysisLimits, AnalyzeSatelliteImageUseCase
from src.domain.exceptions import (
    DomainRuleViolationError,
    InvalidImageError,
    ModelWeightsUnavailableError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def square(x0: int, y0: int, size: int) -> set:
    return {(x, y) for x in range(x0, x0 + size) for y in range(y0, y0 + size)}


def instance(mask: set, confidence: float, class_id: int = 0) -> SegmentationInstance:
    xs = [p[0] for p in mask] or [0]
    ys = [p[1] for p in mask] or [0]
    return SegmentationInstance(class_id, confidence, (min(xs), min(ys), max(xs), max(ys)), mask)


# --------------------------------------------------------------------------- #
# Sahte servisler
# --------------------------------------------------------------------------- #
class FakeLoader(IImageLoader):
    def __init__(self, width=100, height=80, metadata: Optional[SpatialMetadata] = None):
        self.width, self.height = width, height
        self.metadata = metadata or SpatialMetadata(source_format="png")
        self.loaded: List[str] = []

    def load(self, data: bytes, filename: str) -> ImageData:
        self.loaded.append(filename)
        return ImageData(
            pixels=f"PIXELS:{filename}",
            width=self.width,
            height=self.height,
            filename=filename,
            metadata=self.metadata,
        )


class FakeSegmenter(ISegmentationService):
    def __init__(self, instances=None, ready=True):
        self.instances = instances if instances is not None else []
        self.ready = ready
        self.calls: List[tuple] = []

    def is_ready(self) -> bool:
        return self.ready

    def model_info(self) -> ModelInfo:
        return ModelInfo(weights_path="weights/best_model.pt", weights_sha256="abc123")

    def segment(self, image, confidence_threshold):
        self.calls.append((image.filename, confidence_threshold))
        return list(self.instances)


class FakeMeasurer(IMeasurementService):
    """Gerçek ölçüm servisinin sözleşmesini kümelerle taklit eder."""

    def __init__(self, aoi: Optional[set] = None, aoi_assumption: str = "full_image"):
        self.aoi = aoi
        self.aoi_assumption = aoi_assumption
        self.received_masks = None

    def measure(self, image, masks, exclude_black_padding):
        self.received_masks = list(masks)
        aoi = self.aoi
        if aoi is None:
            aoi = {(x, y) for x in range(image.width) for y in range(image.height)}
        union = set().union(*masks) if masks else set()
        return MeasurementResultDTO(
            instance_pixel_areas=[len(m) for m in masks],
            damaged_union_pixels=len(union & aoi),
            valid_aoi_pixels=len(aoi),
            aoi_assumption=self.aoi_assumption,
        )


class FakeVisualizer(IVisualizationService):
    def __init__(self):
        self.overlay_args = None

    def render_overlay(self, image, instances, partial_flags, labels):
        self.overlay_args = (list(instances), list(partial_flags), list(labels))
        return "ANNOTATED"

    def render_union_mask(self, image, instances):
        return "MASK"


class FakeWriter(IOutputWriter):
    def __init__(self):
        self.calls = []

    def save(self, base_name, annotated_image, mask_image, region_rows, summary):
        self.calls.append(
            dict(base=base_name, annotated=annotated_image, mask=mask_image,
                 rows=region_rows, summary=summary)
        )
        out = Path("data/test_results")
        return SavedOutputsDTO(
            out / f"{base_name}_annotated.png",
            out / f"{base_name}_mask.png",
            out / f"{base_name}_metrics.csv",
            out / f"{base_name}_summary.json",
        )


def build(
    instances=None,
    ready=True,
    loader=None,
    measurer=None,
    limits=AnalysisLimits(),
):
    parts = dict(
        loader=loader or FakeLoader(),
        segmenter=FakeSegmenter(instances, ready),
        measurer=measurer or FakeMeasurer(),
        visualizer=FakeVisualizer(),
        writer=FakeWriter(),
    )
    uc = AnalyzeSatelliteImageUseCase(
        parts["loader"], parts["segmenter"], parts["measurer"],
        parts["visualizer"], parts["writer"], limits,
    )
    return uc, parts


def request(**kwargs) -> AnalysisRequestDTO:
    defaults = dict(post_image_bytes=b"\x89PNG-data", post_image_filename="malatya_post.png")
    defaults.update(kwargs)
    return AnalysisRequestDTO(**defaults)


def codes(warnings: List[str]) -> List[str]:
    return [w.split(":")[0] for w in warnings]


# --------------------------------------------------------------------------- #
# Dosya validasyonu
# --------------------------------------------------------------------------- #
class TestFileValidation:
    @pytest.mark.parametrize("name", ["a.png", "a.PNG", "a.jpg", "a.jpeg", "A.JPEG"])
    def test_allowed_extensions(self, name):
        uc, _ = build()
        uc.execute(request(post_image_filename=name))

    @pytest.mark.parametrize("name", ["a.gif", "a.bmp", "a.tif", "noext", "a.png.exe"])
    def test_rejected_extensions(self, name):
        uc, _ = build()
        with pytest.raises(InvalidImageError, match="Desteklenmeyen"):
            uc.execute(request(post_image_filename=name))

    def test_empty_file_is_rejected(self):
        uc, _ = build()
        with pytest.raises(InvalidImageError, match="boş"):
            uc.execute(request(post_image_bytes=b""))

    def test_empty_filename_is_rejected(self):
        uc, _ = build()
        with pytest.raises(InvalidImageError):
            uc.execute(request(post_image_filename=""))

    def test_file_over_size_limit_is_rejected(self):
        uc, _ = build(limits=AnalysisLimits(max_image_mb=1))
        with pytest.raises(InvalidImageError, match="çok büyük"):
            uc.execute(request(post_image_bytes=b"x" * (1024 * 1024 + 1)))

    def test_image_over_side_limit_is_rejected_in_turkish(self):
        uc, _ = build(loader=FakeLoader(width=5000, height=100))
        with pytest.raises(InvalidImageError, match="Görüntü çok büyük"):
            uc.execute(request())

    def test_tiff_allowed_when_configured(self):
        limits = AnalysisLimits(allowed_extensions=frozenset({".png", ".tif"}))
        uc, _ = build(limits=limits)
        uc.execute(request(post_image_filename="scene.tif"))

    def test_validation_happens_before_loading(self):
        uc, parts = build()
        with pytest.raises(InvalidImageError):
            uc.execute(request(post_image_filename="a.gif"))
        assert parts["loader"].loaded == []


# --------------------------------------------------------------------------- #
# Model durumu
# --------------------------------------------------------------------------- #
class TestModelAvailability:
    def test_missing_weights_raise_clear_error(self):
        uc, parts = build(ready=False)
        with pytest.raises(ModelWeightsUnavailableError, match="ağırlık"):
            uc.execute(request())
        assert parts["segmenter"].calls == []
        assert parts["writer"].calls == []


# --------------------------------------------------------------------------- #
# Ana akış
# --------------------------------------------------------------------------- #
class TestHappyPath:
    def test_full_pipeline_with_overlap_and_partial(self):
        a = square(10, 10, 10)  # 100 px
        b = square(15, 10, 10)  # 100 px, a ile 50 px çakışır
        edge = square(0, 0, 5)  # 25 px, kenarda
        uc, parts = build([instance(a, 0.9), instance(b, 0.8), instance(edge, 0.75)])

        out = uc.execute(request(user_gsd_m_per_pixel=0.5))

        assert out.counts == {
            "damaged_region_count": 3,
            "complete_region_count": 2,
            "partial_region_count": 1,
        }
        assert out.damaged_union_pixels == 175
        assert out.valid_aoi_pixels == 8000
        assert out.damaged_area_ratio == pytest.approx(175 / 8000)
        assert out.damaged_area_m2 == pytest.approx(175 * 0.25)
        assert out.not_predicted_damaged_pixels == 8000 - 175
        assert out.not_predicted_damaged_area_m2 == pytest.approx((8000 - 175) * 0.25)
        assert out.undamaged_building_area_m2 is None
        assert out.calibration_source == "user_supplied_gsd"
        assert out.gsd_m_per_pixel == 0.5
        assert out.pixel_area_m2 == pytest.approx(0.25)
        assert out.annotated_image == "ANNOTATED"
        assert out.original_image == "PIXELS:malatya_post.png"
        assert out.saved_summary_path.name == "malatya_post_summary.json"
        assert out.inference_time_ms >= 0
        assert {"PARTIAL_REGIONS", "OVERLAPPING_REGIONS"} <= set(codes(out.warnings))

    def test_region_dtos(self):
        uc, _ = build([instance(square(10, 10, 10), 0.9), instance(square(0, 0, 4), 0.8)])
        out = uc.execute(request(user_gsd_m_per_pixel=0.5))
        first, second = out.objects
        assert (first.id, first.object_class, first.pixel_area) == (1, "damaged_region", 100)
        assert first.area_m2 == pytest.approx(25.0)
        assert first.is_partial is False
        assert first.bounding_box == (10, 10, 19, 19)
        assert second.is_partial is True

    def test_partial_flags_and_labels_reach_visualizer(self):
        uc, parts = build([instance(square(10, 10, 10), 0.9), instance(square(0, 0, 4), 0.8)])
        uc.execute(request())
        inst, flags, labels = parts["visualizer"].overlay_args
        assert len(inst) == 2
        assert flags == [False, True]
        assert labels == ["#1 0.90", "#2 0.80"]

    def test_union_respects_aoi(self):
        aoi = square(0, 0, 50)  # 2500 px geçerli alan
        mask = square(45, 45, 10)  # 100 px, yalnızca 25 px AOI içinde
        uc, _ = build([instance(mask, 0.9)], measurer=FakeMeasurer(aoi, "geotiff_nodata"))
        out = uc.execute(request())
        assert out.valid_aoi_pixels == 2500
        assert out.damaged_union_pixels == 25
        assert out.objects[0].pixel_area == 100
        assert out.aoi_assumption == "geotiff_nodata"
        assert "AOI_FULL_IMAGE_ASSUMED" not in codes(out.warnings)


# --------------------------------------------------------------------------- #
# Eşik (Kural 1)
# --------------------------------------------------------------------------- #
class TestConfidenceThreshold:
    def test_default_threshold_from_limits(self):
        uc, parts = build(limits=AnalysisLimits(default_confidence_threshold=0.70))
        out = uc.execute(request())
        assert out.confidence_threshold == 0.70
        assert parts["segmenter"].calls == [("malatya_post.png", 0.70)]

    def test_user_threshold_overrides_default(self):
        uc, parts = build()
        out = uc.execute(request(confidence_threshold=0.5))
        assert out.confidence_threshold == 0.5
        assert parts["segmenter"].calls[0][1] == 0.5

    def test_low_confidence_masks_are_not_measured(self):
        high = square(10, 10, 10)
        low = square(50, 50, 10)
        uc, parts = build([instance(high, 0.9), instance(low, 0.3)])
        out = uc.execute(request())
        assert parts["measurer"].received_masks == [high]
        assert out.counts["damaged_region_count"] == 1
        assert out.damaged_union_pixels == 100

    def test_invalid_threshold_is_rejected(self):
        uc, _ = build()
        with pytest.raises(DomainRuleViolationError):
            uc.execute(request(confidence_threshold=0.0))

    def test_threshold_written_to_summary(self):
        uc, parts = build()
        uc.execute(request(confidence_threshold=0.55))
        assert parts["writer"].calls[0]["summary"]["settings"]["confidence_threshold"] == 0.55


# --------------------------------------------------------------------------- #
# Kalibrasyon (Kural 6)
# --------------------------------------------------------------------------- #
class TestCalibrationResolution:
    def test_no_calibration_means_no_m2(self):
        uc, _ = build([instance(square(10, 10, 10), 0.9)])
        out = uc.execute(request())
        assert out.calibration_source == "unknown"
        assert out.damaged_area_m2 is None
        assert out.not_predicted_damaged_area_m2 is None
        assert out.pixel_area_m2 is None
        assert all(o.area_m2 is None for o in out.objects)
        assert out.damaged_union_pixels == 100  # piksel değeri yine raporlanır
        assert "CALIBRATION_UNAVAILABLE" in codes(out.warnings)

    def test_metric_georeference_is_used(self):
        meta = SpatialMetadata(
            has_georeference=True, crs="EPSG:32637", crs_is_projected_metric=True,
            affine=(0.5, 0.0, 400000.0, 0.0, -0.4, 4250000.0), source_format="geotiff",
        )
        uc, _ = build([instance(square(10, 10, 10), 0.9)], loader=FakeLoader(metadata=meta))
        out = uc.execute(request())
        assert out.calibration_source == "verified_georeference"
        assert out.pixel_area_m2 == pytest.approx(0.2)
        assert out.damaged_area_m2 == pytest.approx(20.0)
        assert out.gsd_m_per_pixel is None

    def test_georeference_wins_over_user_gsd_with_warning(self):
        meta = SpatialMetadata(
            has_georeference=True, crs="EPSG:32637", crs_is_projected_metric=True,
            affine=(0.5, 0.0, 0.0, 0.0, -0.5, 0.0),
        )
        uc, _ = build(loader=FakeLoader(metadata=meta))
        out = uc.execute(request(user_gsd_m_per_pixel=1.0))
        assert out.calibration_source == "verified_georeference"
        assert out.pixel_area_m2 == pytest.approx(0.25)
        assert "USER_GSD_IGNORED" in codes(out.warnings)

    def test_geographic_crs_falls_back_to_unknown(self):
        meta = SpatialMetadata(
            has_georeference=True, crs="EPSG:4326", crs_is_projected_metric=False,
            affine=(0.00001, 0.0, 38.3, 0.0, -0.00001, 38.35),
        )
        uc, _ = build(loader=FakeLoader(metadata=meta))
        out = uc.execute(request())
        assert out.calibration_source == "unknown"
        assert {"GEOREFERENCE_NOT_METRIC", "CALIBRATION_UNAVAILABLE"} <= set(codes(out.warnings))

    def test_geographic_crs_falls_back_to_user_gsd(self):
        meta = SpatialMetadata(
            has_georeference=True, crs="EPSG:4326", crs_is_projected_metric=False,
            affine=(0.00001, 0.0, 38.3, 0.0, -0.00001, 38.35),
        )
        uc, _ = build(loader=FakeLoader(metadata=meta))
        out = uc.execute(request(user_gsd_m_per_pixel=0.3))
        assert out.calibration_source == "user_supplied_gsd"
        assert out.pixel_area_m2 == pytest.approx(0.09)

    def test_negative_user_gsd_is_rejected(self):
        uc, _ = build()
        with pytest.raises(DomainRuleViolationError):
            uc.execute(request(user_gsd_m_per_pixel=-0.5))


# --------------------------------------------------------------------------- #
# Sonuçsuz analiz ve kenar durumları
# --------------------------------------------------------------------------- #
class TestEdgeCases:
    def test_no_damage_returns_zero_with_warning(self):
        uc, parts = build([])
        out = uc.execute(request(user_gsd_m_per_pixel=0.5))
        assert out.counts["damaged_region_count"] == 0
        assert out.damaged_union_pixels == 0
        assert out.damaged_area_ratio == 0.0
        assert out.damaged_area_m2 == 0.0
        assert out.objects == []
        assert "NO_DAMAGE_DETECTED" in codes(out.warnings)
        assert len(parts["writer"].calls) == 1  # yine de kaydedilir

    def test_empty_mask_is_skipped_with_warning(self):
        uc, _ = build([instance(set(), 0.9), instance(square(10, 10, 10), 0.9)])
        out = uc.execute(request())
        assert out.counts["damaged_region_count"] == 1
        assert out.objects[0].id == 1
        assert "EMPTY_MASK_SKIPPED" in codes(out.warnings)

    def test_unknown_class_id_is_rejected(self):
        uc, _ = build([instance(square(10, 10, 10), 0.9, class_id=1)])
        with pytest.raises(ValueError):
            uc.execute(request())

    def test_measurement_count_mismatch_is_rejected(self):
        class BrokenMeasurer(FakeMeasurer):
            def measure(self, image, masks, exclude_black_padding):
                good = super().measure(image, masks, exclude_black_padding)
                return MeasurementResultDTO([], good.damaged_union_pixels,
                                            good.valid_aoi_pixels, good.aoi_assumption)

        uc, _ = build([instance(square(10, 10, 10), 0.9)], measurer=BrokenMeasurer())
        with pytest.raises(DomainRuleViolationError):
            uc.execute(request())

    def test_always_present_scientific_warnings(self):
        uc, _ = build()
        out = uc.execute(request())
        assert {"SINGLE_DATE_MODEL", "LOCATION_NOT_VERIFIED", "AOI_FULL_IMAGE_ASSUMED"} <= set(
            codes(out.warnings)
        )


# --------------------------------------------------------------------------- #
# Deprem öncesi görüntü
# --------------------------------------------------------------------------- #
class TestPreImage:
    def test_pre_image_is_loaded_but_not_segmented(self):
        uc, parts = build([instance(square(10, 10, 10), 0.9)])
        out = uc.execute(
            request(pre_image_bytes=b"pre", pre_image_filename="malatya_pre.png")
        )
        assert parts["loader"].loaded == ["malatya_post.png", "malatya_pre.png"]
        assert [c[0] for c in parts["segmenter"].calls] == ["malatya_post.png"]
        assert out.pre_image == "PIXELS:malatya_pre.png"
        assert "PRE_IMAGE_DISPLAY_ONLY" in codes(out.warnings)
        summary = parts["writer"].calls[0]["summary"]
        assert summary["input"]["pre_image_provided"] is True
        assert summary["input"]["pre_image_used_by_model"] is False

    def test_no_pre_image(self):
        uc, _ = build()
        out = uc.execute(request())
        assert out.pre_image is None
        assert "PRE_IMAGE_DISPLAY_ONLY" not in codes(out.warnings)

    def test_invalid_pre_image_is_rejected(self):
        uc, _ = build()
        with pytest.raises(InvalidImageError):
            uc.execute(request(pre_image_bytes=b"pre", pre_image_filename="pre.gif"))


# --------------------------------------------------------------------------- #
# Kayıt ve özet
# --------------------------------------------------------------------------- #
class TestOutputs:
    def test_writer_receives_rows_and_summary(self):
        uc, parts = build([instance(square(10, 10, 10), 0.9)])
        uc.execute(request(user_gsd_m_per_pixel=0.5))
        call = parts["writer"].calls[0]
        assert call["base"] == "malatya_post"
        assert call["annotated"] == "ANNOTATED"
        assert call["mask"] == "MASK"
        assert call["rows"][0]["pixel_area"] == 100
        assert call["rows"][0]["area_m2"] == pytest.approx(25.0)

        summary = call["summary"]
        assert summary["model"]["weights_sha256"] == "abc123"
        assert summary["calibration"]["source"] == "user_supplied_gsd"
        assert summary["metrics"]["damaged_union_pixels"] == 100
        assert summary["metrics"]["undamaged_building_area_m2"] is None
        assert "DEĞİLDİR" in summary["definitions"]["damaged_area_ratio"]
        assert "DEĞİLDİR" in summary["definitions"]["not_predicted_damaged"]
        assert summary["warnings"]

    def test_results_are_deterministic_for_same_input(self):
        insts = [instance(square(10, 10, 10), 0.9), instance(square(15, 10, 10), 0.8)]
        uc, _ = build(insts)
        first = uc.execute(request(user_gsd_m_per_pixel=0.5))
        second = uc.execute(request(user_gsd_m_per_pixel=0.5))
        assert first.objects == second.objects
        assert first.damaged_union_pixels == second.damaged_union_pixels
        assert first.damaged_area_m2 == second.damaged_area_m2


# --------------------------------------------------------------------------- #
# Mimari kural: domain ve application teknoloji bağımsız
# --------------------------------------------------------------------------- #
def test_domain_and_application_do_not_import_heavy_libraries():
    script = textwrap.dedent(
        """
        import builtins, sys
        banned = {"numpy", "cv2", "ultralytics", "streamlit", "rasterio", "torch", "PIL"}
        real = builtins.__import__
        def guard(name, *a, **k):
            if name.split(".")[0] in banned:
                raise ImportError("banned import: " + name)
            return real(name, *a, **k)
        builtins.__import__ = guard
        import src.domain, src.domain.entities, src.domain.exceptions
        import src.application.dtos, src.application.interfaces, src.application.use_cases
        print("ok")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
