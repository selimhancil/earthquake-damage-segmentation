"""Domain katmanı birim testleri (şartname bölüm 8 - test_domain.py)."""

import math

import pytest

from src.domain.entities import (
    AnalysisResult,
    AoiAssumption,
    Calibration,
    CalibrationSource,
    DamageRegion,
    MeasurementSettings,
    ObjectClass,
)
from src.domain.exceptions import DomainRuleViolationError

DR = ObjectClass.DAMAGED_REGION


def make_result(
    width: int = 100,
    height: int = 80,
    threshold: float = 0.70,
    calibration: Calibration = None,
) -> AnalysisResult:
    settings = MeasurementSettings(
        calibration=calibration or Calibration.unknown(), confidence_threshold=threshold
    )
    return AnalysisResult(image_width=width, image_height=height, settings=settings)


# --------------------------------------------------------------------------- #
# ObjectClass (Kural 7)
# --------------------------------------------------------------------------- #
class TestObjectClass:
    def test_only_damaged_region_class_exists(self):
        assert [c.name for c in ObjectClass] == ["DAMAGED_REGION"]
        assert DR.value == 0
        assert DR.label == "damaged_region"

    def test_undamaged_building_class_is_not_defined(self):
        assert not hasattr(ObjectClass, "UNDAMAGED_BUILDING")


# --------------------------------------------------------------------------- #
# Calibration (Kural 2 ve 6)
# --------------------------------------------------------------------------- #
class TestCalibration:
    def test_pixel_area_is_gsd_squared(self):
        cal = Calibration.from_user_gsd(0.5)
        assert cal.source == CalibrationSource.USER_SUPPLIED_GSD
        assert cal.pixel_area_m2 == pytest.approx(0.25)
        assert cal.gsd_m_per_pixel == 0.5

    def test_1000_pixels_at_half_meter_gsd_is_250_m2(self):
        assert Calibration.from_user_gsd(0.5).to_m2(1000) == pytest.approx(250.0)

    @pytest.mark.parametrize("gsd", [0, -0.3, None])
    def test_non_positive_gsd_raises(self, gsd):
        with pytest.raises(DomainRuleViolationError):
            Calibration.from_user_gsd(gsd)

    def test_unknown_calibration_returns_none_for_m2(self):
        cal = Calibration.unknown()
        assert cal.source == CalibrationSource.UNKNOWN
        assert not cal.is_available
        assert cal.to_m2(1000) is None

    def test_unknown_calibration_cannot_carry_a_scale(self):
        with pytest.raises(DomainRuleViolationError):
            Calibration(source=CalibrationSource.UNKNOWN, pixel_area_m2=0.25)

    def test_known_source_requires_positive_pixel_area(self):
        with pytest.raises(DomainRuleViolationError):
            Calibration(source=CalibrationSource.VERIFIED_GEOREFERENCE, pixel_area_m2=None)
        with pytest.raises(DomainRuleViolationError):
            Calibration(source=CalibrationSource.VERIFIED_GEOREFERENCE, pixel_area_m2=-1.0)

    def test_affine_square_pixels_with_north_up_negative_e(self):
        cal = Calibration.from_affine(0.5, 0.0, 0.0, -0.5, crs_is_projected_metric=True)
        assert cal.source == CalibrationSource.VERIFIED_GEOREFERENCE
        assert cal.pixel_area_m2 == pytest.approx(0.25)
        assert cal.gsd_m_per_pixel is None

    def test_affine_non_square_pixels(self):
        cal = Calibration.from_affine(0.5, 0.0, 0.0, -0.4, crs_is_projected_metric=True)
        assert cal.pixel_area_m2 == pytest.approx(0.2)

    def test_affine_rotation_preserves_pixel_area(self):
        theta = math.radians(30)
        size = 0.5
        a, b = size * math.cos(theta), -size * math.sin(theta)
        d, e = size * math.sin(theta), size * math.cos(theta)
        cal = Calibration.from_affine(a, b, d, e, crs_is_projected_metric=True)
        assert cal.pixel_area_m2 == pytest.approx(0.25)

    def test_affine_with_geographic_crs_is_rejected(self):
        with pytest.raises(DomainRuleViolationError):
            Calibration.from_affine(0.00001, 0, 0, -0.00001, crs_is_projected_metric=False)

    def test_degenerate_affine_is_rejected(self):
        with pytest.raises(DomainRuleViolationError):
            Calibration.from_affine(0.5, 0.5, 0.5, 0.5, crs_is_projected_metric=True)

    def test_negative_pixel_count_is_rejected(self):
        with pytest.raises(DomainRuleViolationError):
            Calibration.from_user_gsd(0.5).to_m2(-1)


# --------------------------------------------------------------------------- #
# MeasurementSettings (Kural 1)
# --------------------------------------------------------------------------- #
class TestMeasurementSettings:
    def test_default_threshold_is_070(self):
        assert MeasurementSettings().confidence_threshold == 0.70

    @pytest.mark.parametrize("threshold", [0.0, -0.1, 1.01])
    def test_threshold_must_be_in_open_closed_unit_interval(self, threshold):
        with pytest.raises(DomainRuleViolationError):
            MeasurementSettings(confidence_threshold=threshold)

    def test_threshold_of_one_is_allowed(self):
        assert MeasurementSettings(confidence_threshold=1.0).accepts_confidence(1.0)

    def test_accepts_at_or_above_threshold(self):
        s = MeasurementSettings(confidence_threshold=0.7)
        assert s.accepts_confidence(0.7)
        assert s.accepts_confidence(0.95)
        assert not s.accepts_confidence(0.69)

    @pytest.mark.parametrize("confidence", [-0.01, 1.01])
    def test_out_of_range_confidence_raises(self, confidence):
        with pytest.raises(DomainRuleViolationError):
            MeasurementSettings().accepts_confidence(confidence)


# --------------------------------------------------------------------------- #
# DamageRegion (Kural 2 ve 4)
# --------------------------------------------------------------------------- #
class TestDamageRegion:
    def test_valid_region(self):
        r = DamageRegion(1, DR, 0.9, (10, 10, 20, 20), 50)
        assert r.pixel_area == 50
        assert not r.is_partial

    @pytest.mark.parametrize("confidence", [-0.1, 1.1])
    def test_confidence_must_be_in_closed_unit_interval(self, confidence):
        with pytest.raises(DomainRuleViolationError):
            DamageRegion(1, DR, confidence, (10, 10, 20, 20), 50)

    @pytest.mark.parametrize("confidence", [0.0, 1.0])
    def test_confidence_bounds_are_inclusive(self, confidence):
        DamageRegion(1, DR, confidence, (10, 10, 20, 20), 50)

    @pytest.mark.parametrize("area", [0, -5])
    def test_non_positive_pixel_area_raises(self, area):
        with pytest.raises(DomainRuleViolationError):
            DamageRegion(1, DR, 0.9, (10, 10, 20, 20), area)

    def test_float_pixel_area_is_rejected(self):
        with pytest.raises(DomainRuleViolationError):
            DamageRegion(1, DR, 0.9, (10, 10, 20, 20), 50.5)

    @pytest.mark.parametrize("bbox", [(20, 10, 10, 20), (10, 20, 20, 10), (-1, 0, 5, 5)])
    def test_invalid_bounding_box_raises(self, bbox):
        with pytest.raises(DomainRuleViolationError):
            DamageRegion(1, DR, 0.9, bbox, 10)

    @pytest.mark.parametrize(
        "bbox, expected",
        [
            ((10, 10, 20, 20), False),
            ((0, 10, 20, 20), True),  # sol kenar
            ((10, 0, 20, 20), True),  # üst kenar
            ((10, 10, 99, 20), True),  # sağ kenar (width-1)
            ((10, 10, 20, 79), True),  # alt kenar (height-1)
            ((1, 1, 98, 78), False),  # kenara bir piksel mesafe
        ],
    )
    def test_edge_detection(self, bbox, expected):
        r = DamageRegion(1, DR, 0.9, bbox, 10)
        assert r.is_touching_edge(image_width=100, image_height=80) is expected

    def test_physical_area_uses_mask_pixels_not_bbox(self):
        # bbox 11x11=121 piksel ama maske yalnızca 40 piksel
        r = DamageRegion(1, DR, 0.9, (10, 10, 20, 20), 40)
        assert r.get_physical_area(Calibration.from_user_gsd(0.5)) == pytest.approx(10.0)

    def test_physical_area_is_none_without_calibration(self):
        r = DamageRegion(1, DR, 0.9, (10, 10, 20, 20), 40)
        assert r.get_physical_area(Calibration.unknown()) is None


# --------------------------------------------------------------------------- #
# AnalysisResult aggregate
# --------------------------------------------------------------------------- #
class TestAnalysisResultRegions:
    def test_below_threshold_region_is_not_added(self):
        result = make_result(threshold=0.7)
        assert result.add_region(1, DR, 0.69, (10, 10, 20, 20), 50) is None
        assert result.region_count == 0

    def test_region_at_threshold_is_added(self):
        result = make_result(threshold=0.7)
        region = result.add_region(1, DR, 0.7, (10, 10, 20, 20), 50)
        assert region is not None
        assert result.region_count == 1

    def test_partial_flag_is_set_automatically(self):
        result = make_result()
        inside = result.add_region(1, DR, 0.9, (10, 10, 20, 20), 50)
        edge = result.add_region(2, DR, 0.9, (0, 0, 5, 5), 20)
        assert not inside.is_partial
        assert edge.is_partial
        assert result.complete_regions == [inside]
        assert result.partial_regions == [edge]

    def test_partial_region_area_is_kept_in_totals(self):
        result = make_result()
        result.add_region(1, DR, 0.9, (0, 0, 9, 9), 100)
        result.set_pixel_counts(damaged_union_pixels=100, valid_aoi_pixels=8000)
        assert result.damaged_union_pixels == 100
        assert result.sum_of_region_pixels == 100

    def test_duplicate_region_id_raises(self):
        result = make_result()
        result.add_region(1, DR, 0.9, (10, 10, 20, 20), 50)
        with pytest.raises(DomainRuleViolationError):
            result.add_region(1, DR, 0.9, (30, 30, 40, 40), 50)

    def test_bbox_outside_image_raises(self):
        result = make_result(width=100, height=80)
        with pytest.raises(DomainRuleViolationError):
            result.add_region(1, DR, 0.9, (10, 10, 100, 20), 50)

    def test_cannot_add_region_after_counts_are_set(self):
        result = make_result()
        result.add_region(1, DR, 0.9, (10, 10, 20, 20), 50)
        result.set_pixel_counts(50, 8000)
        with pytest.raises(DomainRuleViolationError):
            result.add_region(2, DR, 0.9, (30, 30, 40, 40), 50)

    def test_region_count_is_not_called_building_count(self):
        result = make_result()
        assert not hasattr(result, "building_count")

    @pytest.mark.parametrize("w, h", [(0, 10), (10, 0), (-1, 10)])
    def test_invalid_image_size_raises(self, w, h):
        with pytest.raises(DomainRuleViolationError):
            make_result(width=w, height=h)


class TestAnalysisResultCounts:
    def test_metrics_require_counts(self):
        result = make_result()
        with pytest.raises(DomainRuleViolationError):
            _ = result.damaged_union_pixels
        with pytest.raises(DomainRuleViolationError):
            result.damaged_area_ratio()

    def test_union_is_used_instead_of_sum_for_overlaps(self):
        result = make_result(calibration=Calibration.from_user_gsd(0.5))
        result.add_region(1, DR, 0.9, (10, 10, 19, 19), 100)
        result.add_region(2, DR, 0.8, (15, 10, 24, 19), 100)
        result.set_pixel_counts(damaged_union_pixels=150, valid_aoi_pixels=8000)
        assert result.sum_of_region_pixels == 200
        assert result.damaged_union_pixels == 150
        assert result.overlap_pixels == 50
        assert result.calculate_union_damaged_area() == pytest.approx(37.5)

    def test_union_larger_than_sum_is_rejected(self):
        result = make_result()
        result.add_region(1, DR, 0.9, (10, 10, 20, 20), 100)
        with pytest.raises(DomainRuleViolationError):
            result.set_pixel_counts(damaged_union_pixels=101, valid_aoi_pixels=8000)

    def test_union_larger_than_aoi_is_rejected(self):
        result = make_result()
        result.add_region(1, DR, 0.9, (10, 10, 20, 20), 100)
        with pytest.raises(DomainRuleViolationError):
            result.set_pixel_counts(damaged_union_pixels=100, valid_aoi_pixels=50)

    def test_aoi_larger_than_image_is_rejected(self):
        result = make_result(width=100, height=80)
        with pytest.raises(DomainRuleViolationError):
            result.set_pixel_counts(0, 8001)

    @pytest.mark.parametrize("union, aoi", [(-1, 8000), (0, 0), (0, -5)])
    def test_invalid_counts_are_rejected(self, union, aoi):
        result = make_result()
        with pytest.raises(DomainRuleViolationError):
            result.set_pixel_counts(union, aoi)

    def test_damaged_area_ratio_uses_valid_aoi(self):
        result = make_result()
        result.add_region(1, DR, 0.9, (10, 10, 20, 20), 400)
        result.set_pixel_counts(damaged_union_pixels=400, valid_aoi_pixels=4000)
        assert result.damaged_area_ratio() == pytest.approx(0.1)

    def test_not_predicted_damaged_area(self):
        result = make_result(calibration=Calibration.from_user_gsd(0.5))
        result.add_region(1, DR, 0.9, (10, 10, 20, 20), 400)
        result.set_pixel_counts(damaged_union_pixels=400, valid_aoi_pixels=4000)
        assert result.not_predicted_damaged_pixels == 3600
        assert result.calculate_not_predicted_damaged_area() == pytest.approx(900.0)

    def test_unknown_calibration_gives_no_m2_anywhere(self):
        result = make_result(calibration=Calibration.unknown())
        result.add_region(1, DR, 0.9, (10, 10, 20, 20), 400)
        result.set_pixel_counts(400, 4000)
        assert result.calculate_union_damaged_area() is None
        assert result.calculate_not_predicted_damaged_area() is None
        assert result.calculate_average_region_area_m2() is None
        # piksel metrikleri yine de mevcut
        assert result.damaged_union_pixels == 400
        assert result.damaged_area_ratio() == pytest.approx(0.1)

    def test_undamaged_building_area_is_always_none(self):
        result = make_result(calibration=Calibration.from_user_gsd(0.5))
        result.add_region(1, DR, 0.9, (10, 10, 20, 20), 400)
        result.set_pixel_counts(400, 4000)
        assert result.undamaged_building_area_m2 is None

    def test_no_detection_returns_zero_without_crashing(self):
        result = make_result(calibration=Calibration.from_user_gsd(0.5))
        result.set_pixel_counts(damaged_union_pixels=0, valid_aoi_pixels=8000)
        assert not result.has_detections
        assert result.damaged_union_pixels == 0
        assert result.damaged_area_ratio() == 0.0
        assert result.calculate_union_damaged_area() == 0.0
        assert result.calculate_average_region_area_pixels() is None

    def test_average_region_area(self):
        result = make_result(calibration=Calibration.from_user_gsd(0.5))
        result.add_region(1, DR, 0.9, (10, 10, 20, 20), 100)
        result.add_region(2, DR, 0.9, (30, 30, 40, 40), 300)
        assert result.calculate_average_region_area_pixels() == pytest.approx(200.0)
        assert result.calculate_average_region_area_m2() == pytest.approx(50.0)

    def test_default_aoi_assumption_is_full_image(self):
        assert make_result().aoi_assumption == AoiAssumption.FULL_IMAGE
