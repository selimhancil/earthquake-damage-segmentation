"""Ölçüm, görüntü okuma ve görselleştirme testleri (şartname bölüm 8 - test_measurement.py)."""

import cv2
import numpy as np
import pytest

from src.application.dtos import ImageData, SegmentationInstance, SpatialMetadata
from src.domain.entities import Calibration
from src.domain.exceptions import DimensionMismatchError, InvalidImageError
from src.infrastructure.cv2_service import (
    COLOR_DAMAGED,
    COLOR_PARTIAL,
    CV2ImageLoader,
    CV2MeasurementService,
    CV2VisualizationService,
    resize_mask_nearest,
    to_rgb_uint8,
)
from src.infrastructure.raster_service import (
    build_spatial_metadata,
    is_projected_metric,
    stretch_to_uint8,
)

H, W = 40, 60


def image_data(pixels=None, nodata_mask=None) -> ImageData:
    if pixels is None:
        pixels = np.full((H, W, 3), 120, dtype=np.uint8)
    return ImageData(
        pixels=pixels,
        width=pixels.shape[1],
        height=pixels.shape[0],
        filename="test.png",
        has_nodata_mask=nodata_mask is not None,
        nodata_mask=nodata_mask,
    )


def rect_mask(x0, y0, x1, y1, h=H, w=W) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def encode(array: np.ndarray, ext: str = ".png") -> bytes:
    ok, buf = cv2.imencode(ext, array)
    assert ok
    return buf.tobytes()


# --------------------------------------------------------------------------- #
# Union ve AOI
# --------------------------------------------------------------------------- #
class TestMeasurement:
    def test_single_mask_pixel_count(self):
        m = rect_mask(10, 10, 20, 20)  # 100 px
        out = CV2MeasurementService().measure(image_data(), [m], False)
        assert out.instance_pixel_areas == [100]
        assert out.damaged_union_pixels == 100
        assert out.valid_aoi_pixels == H * W
        assert out.aoi_assumption == "full_image"

    def test_overlapping_masks_are_not_double_counted(self):
        a = rect_mask(10, 10, 20, 20)  # 100
        b = rect_mask(15, 10, 25, 20)  # 100, 50 çakışma
        out = CV2MeasurementService().measure(image_data(), [a, b], False)
        assert out.instance_pixel_areas == [100, 100]
        assert sum(out.instance_pixel_areas) == 200
        assert out.damaged_union_pixels == 150

    def test_identical_masks_count_once(self):
        a = rect_mask(10, 10, 20, 20)
        out = CV2MeasurementService().measure(image_data(), [a, a.copy(), a.copy()], False)
        assert out.damaged_union_pixels == 100

    def test_no_masks(self):
        out = CV2MeasurementService().measure(image_data(), [], False)
        assert out.instance_pixel_areas == []
        assert out.damaged_union_pixels == 0
        assert out.valid_aoi_pixels == H * W

    def test_union_excludes_pixels_outside_geotiff_nodata_aoi(self):
        aoi = np.zeros((H, W), dtype=bool)
        aoi[:, :30] = True  # sol yarı geçerli
        m = rect_mask(20, 0, 40, 10)  # 200 px, 100 px AOI içinde
        out = CV2MeasurementService().measure(image_data(nodata_mask=aoi), [m], False)
        assert out.instance_pixel_areas == [200]
        assert out.damaged_union_pixels == 100
        assert out.valid_aoi_pixels == H * 30
        assert out.aoi_assumption == "geotiff_nodata"

    def test_black_shadows_are_kept_by_default(self):
        pixels = np.full((H, W, 3), 100, dtype=np.uint8)
        pixels[5:10, 5:10] = 0  # tam siyah gölge
        out = CV2MeasurementService().measure(image_data(pixels), [], False)
        assert out.valid_aoi_pixels == H * W
        assert out.aoi_assumption == "full_image"

    def test_black_padding_excluded_only_when_requested(self):
        pixels = np.full((H, W, 3), 100, dtype=np.uint8)
        pixels[:, 50:] = 0  # sağda siyah dolgu
        m = rect_mask(45, 0, 60, 10)  # 150 px, 50 px geçerli alanda
        out = CV2MeasurementService().measure(image_data(pixels), [m], True)
        assert out.valid_aoi_pixels == H * 50
        assert out.damaged_union_pixels == 50
        assert out.aoi_assumption == "user_black_padding_excluded"

    def test_dark_but_not_black_pixels_stay_in_aoi(self):
        pixels = np.zeros((H, W, 3), dtype=np.uint8)
        pixels[..., 2] = 1  # neredeyse siyah ama tam siyah değil
        out = CV2MeasurementService().measure(image_data(pixels), [], True)
        assert out.valid_aoi_pixels == H * W

    def test_wrong_mask_size_raises(self):
        with pytest.raises(DimensionMismatchError):
            CV2MeasurementService().measure(image_data(), [np.ones((10, 10), bool)], False)

    def test_non_bool_masks_are_accepted(self):
        m = rect_mask(0, 0, 10, 10).astype(np.uint8) * 255
        out = CV2MeasurementService().measure(image_data(), [m], False)
        assert out.instance_pixel_areas == [100]

    def test_measurement_plus_calibration_gives_m2(self):
        m = rect_mask(0, 0, 40, 25)  # 1000 px
        out = CV2MeasurementService().measure(image_data(), [m], False)
        assert Calibration.from_user_gsd(0.5).to_m2(out.damaged_union_pixels) == pytest.approx(250.0)


# --------------------------------------------------------------------------- #
# Maske boyutlandırma (Kural 3)
# --------------------------------------------------------------------------- #
class TestMaskResize:
    def test_same_size_returns_bool(self):
        m = np.zeros((H, W), dtype=np.uint8)
        m[0, 0] = 1
        out = resize_mask_nearest(m, H, W)
        assert out.dtype == bool and out.shape == (H, W) and out[0, 0]

    def test_nearest_neighbour_keeps_binary_values(self):
        small = np.zeros((4, 4), dtype=bool)
        small[1:3, 1:3] = True
        big = resize_mask_nearest(small, 16, 16)
        assert big.shape == (16, 16)
        assert big.dtype == bool
        assert set(np.unique(big)) <= {False, True}
        assert np.count_nonzero(big) == 64  # 4 px × (4×4) büyütme

    def test_downscale(self):
        big = np.ones((80, 120), dtype=bool)
        assert resize_mask_nearest(big, H, W).all()

    def test_3d_mask_is_rejected(self):
        with pytest.raises(DimensionMismatchError):
            resize_mask_nearest(np.ones((4, 4, 3)), 4, 4)


# --------------------------------------------------------------------------- #
# Kanal dönüşümü ve okuma
# --------------------------------------------------------------------------- #
class TestChannelConversion:
    def test_gray_to_rgb(self):
        out = to_rgb_uint8(np.full((5, 6), 77, dtype=np.uint8))
        assert out.shape == (5, 6, 3) and (out == 77).all()

    def test_bgr_to_rgb(self):
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        bgr[..., 0] = 255  # mavi (BGR)
        out = to_rgb_uint8(bgr)
        assert tuple(out[0, 0]) == (0, 0, 255)

    def test_bgra_drops_alpha(self):
        bgra = np.zeros((2, 2, 4), dtype=np.uint8)
        bgra[..., 2] = 200  # kırmızı (BGR'de 2. indeks)
        bgra[..., 3] = 10
        out = to_rgb_uint8(bgra)
        assert out.shape == (2, 2, 3) and tuple(out[0, 0]) == (200, 0, 0)

    def test_two_channel_image_is_rejected(self):
        with pytest.raises(InvalidImageError, match="belirsiz"):
            to_rgb_uint8(np.zeros((2, 2, 2), dtype=np.uint8))

    def test_many_band_image_is_rejected(self):
        with pytest.raises(InvalidImageError):
            to_rgb_uint8(np.zeros((2, 2, 5), dtype=np.uint8))

    def test_uint16_is_scaled(self):
        out = to_rgb_uint8(np.full((2, 2), 65535, dtype=np.uint16))
        assert (out == 255).all()

    def test_float_image_is_rejected(self):
        with pytest.raises(InvalidImageError):
            to_rgb_uint8(np.zeros((2, 2, 3), dtype=np.float32))


class TestCV2ImageLoader:
    def test_png_roundtrip_is_rgb(self):
        bgr = np.zeros((H, W, 3), dtype=np.uint8)
        bgr[..., 2] = 255  # kırmızı
        img = CV2ImageLoader().load(encode(bgr), "scene.png")
        assert (img.width, img.height) == (W, H)
        assert tuple(img.pixels[0, 0]) == (255, 0, 0)
        assert img.metadata.source_format == "png"
        assert not img.metadata.has_georeference
        assert not img.has_nodata_mask

    def test_jpeg_format_name(self):
        img = CV2ImageLoader().load(encode(np.zeros((8, 8, 3), np.uint8), ".jpg"), "a.JPG")
        assert img.metadata.source_format == "jpeg"

    def test_grayscale_png(self):
        img = CV2ImageLoader().load(encode(np.full((8, 8), 50, np.uint8)), "g.png")
        assert img.pixels.shape == (8, 8, 3)

    def test_corrupt_bytes_raise(self):
        with pytest.raises(InvalidImageError, match="okunamadı"):
            CV2ImageLoader().load(b"not an image", "bad.png")

    def test_empty_bytes_raise(self):
        with pytest.raises(InvalidImageError):
            CV2ImageLoader().load(b"", "empty.png")


# --------------------------------------------------------------------------- #
# Görselleştirme
# --------------------------------------------------------------------------- #
class TestVisualization:
    def make_instances(self):
        inside = SegmentationInstance(0, 0.9, (10, 10, 19, 19), rect_mask(10, 10, 20, 20))
        edge = SegmentationInstance(0, 0.8, (0, 0, 5, 5), rect_mask(0, 0, 6, 6))
        return [inside, edge]

    def test_overlay_colors_and_shape(self):
        img = image_data(np.zeros((H, W, 3), np.uint8))
        out = CV2VisualizationService().render_overlay(
            img, self.make_instances(), [False, True], ["", ""]
        )
        assert out.shape == (H, W, 3) and out.dtype == np.uint8
        # kontur rengi doğrudan çizilir: kenar pikselleri tam renkte olmalı
        assert tuple(out[10, 14]) == COLOR_DAMAGED
        assert tuple(out[0, 3]) == COLOR_PARTIAL
        # maskelenmemiş uzak bölge değişmez
        assert tuple(out[35, 55]) == (0, 0, 0)

    def test_overlay_does_not_modify_input(self):
        pixels = np.zeros((H, W, 3), np.uint8)
        img = image_data(pixels)
        CV2VisualizationService().render_overlay(img, self.make_instances(), [False, True], ["a", "b"])
        assert not pixels.any()

    def test_overlay_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            CV2VisualizationService().render_overlay(image_data(), self.make_instances(), [False], ["a", "b"])

    def test_union_mask(self):
        out = CV2VisualizationService().render_union_mask(image_data(), self.make_instances())
        assert out.shape == (H, W) and out.dtype == np.uint8
        assert set(np.unique(out)) == {0, 255}
        assert np.count_nonzero(out) == 100 + 36

    def test_union_mask_empty(self):
        out = CV2VisualizationService().render_union_mask(image_data(), [])
        assert not out.any()


# --------------------------------------------------------------------------- #
# Geotransform: affine determinantı ve projected CRS kontrolü
# --------------------------------------------------------------------------- #
class TestGeoreferenceHelpers:
    @pytest.mark.parametrize(
        "projected, units, expected",
        [
            (True, "metre", True),  # EPSG:32637 (UTM 37N)
            (True, "meter", True),
            (True, "US survey foot", False),
            (False, None, False),  # EPSG:4326
            (False, "metre", False),
        ],
    )
    def test_is_projected_metric(self, projected, units, expected):
        assert is_projected_metric(projected, units) is expected

    def test_utm_metadata_enables_verified_calibration(self):
        affine = (0.5, 0.0, 400000.0, 0.0, -0.5, 4250000.0)
        meta = build_spatial_metadata(affine, "EPSG:32637", True, "metre", None)
        assert meta.has_georeference and meta.crs_is_projected_metric
        a, b, _c, d, e, _f = meta.affine
        cal = Calibration.from_affine(a, b, d, e, meta.crs_is_projected_metric)
        assert cal.pixel_area_m2 == pytest.approx(0.25)

    def test_geographic_crs_is_not_metric(self):
        affine = (0.00001, 0.0, 38.3, 0.0, -0.00001, 38.35)
        meta = build_spatial_metadata(affine, "EPSG:4326", False, None, None)
        assert meta.has_georeference
        assert not meta.crs_is_projected_metric

    def test_identity_transform_means_no_georeference(self):
        meta = build_spatial_metadata((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), "EPSG:32637", True, "metre", None)
        assert not meta.has_georeference
        assert meta.affine is None

    def test_missing_crs_means_no_georeference(self):
        meta = build_spatial_metadata((0.5, 0, 0, 0, -0.5, 0), None, False, None, None)
        assert not meta.has_georeference

    def test_stretch_keeps_uint8(self):
        bands = np.full((3, 4, 4), 9, dtype=np.uint8)
        assert stretch_to_uint8(bands) is bands

    def test_stretch_uint16_to_uint8_range(self):
        bands = np.linspace(0, 4000, 3 * 10 * 10).reshape(3, 10, 10).astype(np.uint16)
        out = stretch_to_uint8(bands)
        assert out.dtype == np.uint8
        assert out.min() == 0 and out.max() == 255

    def test_stretch_ignores_invalid_pixels(self):
        bands = np.full((1, 4, 4), 1000, dtype=np.uint16)
        bands[0, 0, 0] = 60000  # nodata değeri
        valid = np.ones((4, 4), bool)
        valid[0, 0] = False
        out = stretch_to_uint8(bands, valid)
        assert out.dtype == np.uint8
