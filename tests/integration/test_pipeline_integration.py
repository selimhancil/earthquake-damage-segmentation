"""Uçtan uca entegrasyon testleri: use case + gerçek infrastructure adapter'ları.

Her test gerçek bir görüntü dosyasını (PNG / JPEG / GeoTIFF) bayt olarak verir,
tüm katmanlardan geçirir ve diske yazılan çıktıları TEKRAR OKUYARAK doğrular.
"""

import csv
import json
from datetime import datetime

import cv2
import numpy as np
import pytest

from src.application.dtos import AnalysisRequestDTO
from src.application.use_cases import AnalysisLimits
from src.domain.exceptions import InvalidImageError, ModelWeightsUnavailableError
from tests.fakes import FakeYolo, rect_mask
from tests.integration.conftest import (
    ALL_EXTENSIONS,
    encode_image,
    make_geotiff,
    requires_rasterio,
)

pytestmark = pytest.mark.integration

H, W = 64, 96


def scene(value: int = 120) -> np.ndarray:
    return np.full((H, W, 3), value, dtype=np.uint8)


def codes(warnings):
    return {w.split(":")[0] for w in warnings}


def read_csv(path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def standard_model(**kwargs) -> FakeYolo:
    """İki çakışan iç bölge + bir kenar bölgesi + bir eşik altı bölge."""
    return FakeYolo(
        masks=[
            rect_mask(10, 10, 30, 30, H, W),  # 400 px
            rect_mask(20, 10, 40, 30, H, W),  # 400 px, 200 px çakışma
            rect_mask(0, 0, 6, 6, H, W),  # 36 px, kenarda
            rect_mask(60, 40, 70, 50, H, W),  # 100 px, conf 0.30 -> elenir
        ],
        conf=[0.95, 0.85, 0.80, 0.30],
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# 1. PNG: tüm zincir ve diske yazılan çıktıların tutarlılığı
# --------------------------------------------------------------------------- #
class TestPngPipeline:
    def test_dto_and_written_files_agree(self, build_pipeline):
        use_case, _ = build_pipeline(model=standard_model())
        out = use_case.execute(
            AnalysisRequestDTO(encode_image(scene()), "malatya_post.png", user_gsd_m_per_pixel=0.5)
        )

        # DTO
        assert out.counts == {"damaged_region_count": 3, "complete_region_count": 2,
                              "partial_region_count": 1}
        assert out.damaged_union_pixels == 636  # 400 + 400 - 200 + 36
        assert out.damaged_area_m2 == pytest.approx(636 * 0.25)
        assert out.calibration_source == "user_supplied_gsd"

        # Maske PNG'si: beyaz piksel sayısı union ile aynı (tam görüntü AOI)
        mask = cv2.imread(str(out.saved_mask_path), cv2.IMREAD_GRAYSCALE)
        assert mask.shape == (H, W)
        assert int(np.count_nonzero(mask)) == out.damaged_union_pixels

        # Annotated PNG: aynı boyut, renkli, girdiyle aynı değil
        annotated = cv2.imread(str(out.saved_image_path))
        assert annotated.shape == (H, W, 3)
        assert not np.array_equal(annotated, cv2.cvtColor(scene(), cv2.COLOR_RGB2BGR))

        # CSV: DTO'daki bölgelerle birebir
        rows = read_csv(out.saved_metrics_path)
        assert [int(r["id"]) for r in rows] == [o.id for o in out.objects]
        assert [int(r["pixel_area"]) for r in rows] == [400, 400, 36]
        assert [float(r["area_m2"]) for r in rows] == pytest.approx([100.0, 100.0, 9.0])
        assert [r["is_partial"] for r in rows] == ["False", "False", "True"]

        # JSON özeti: DTO ile aynı sayılar ve tanımlar
        summary = json.loads(out.saved_summary_path.read_text(encoding="utf-8"))
        metrics = summary["metrics"]
        assert metrics["damaged_union_pixels"] == out.damaged_union_pixels
        assert metrics["sum_of_region_pixels"] == 836
        assert metrics["damaged_area_ratio"] == pytest.approx(out.damaged_area_ratio)
        assert metrics["damaged_union_area_m2"] == pytest.approx(out.damaged_area_m2)
        assert metrics["undamaged_building_area_m2"] is None
        assert summary["calibration"] == {"source": "user_supplied_gsd",
                                          "pixel_area_m2": 0.25, "user_gsd_m_per_pixel": 0.5}
        assert summary["model"]["weights_sha256"] and len(summary["model"]["weights_sha256"]) == 64
        assert summary["outputs"]["mask_image"] == out.saved_mask_path.name
        assert codes(summary["warnings"]) == codes(out.warnings)

    def test_model_receives_bgr_full_resolution_input(self, build_pipeline):
        pixels = scene()
        pixels[..., 0] = 200  # RGB kırmızı
        model = standard_model()
        use_case, _ = build_pipeline(model=model)
        use_case.execute(AnalysisRequestDTO(encode_image(pixels), "a.png"))
        assert model.received.shape == (H, W, 3)
        assert tuple(model.received[0, 0]) == (120, 120, 200)  # BGR
        assert model.predict_kwargs["retina_masks"] is True
        assert model.predict_kwargs["conf"] == 0.70

    def test_threshold_slider_changes_result(self, build_pipeline):
        use_case, _ = build_pipeline(model=standard_model())
        strict = use_case.execute(AnalysisRequestDTO(encode_image(scene()), "a.png",
                                                     confidence_threshold=0.9))
        loose = use_case.execute(AnalysisRequestDTO(encode_image(scene()), "a.png",
                                                    confidence_threshold=0.2))
        assert strict.counts["damaged_region_count"] == 1
        assert loose.counts["damaged_region_count"] == 4
        assert loose.damaged_union_pixels == 736  # 636 + 100

    def test_no_calibration_writes_empty_m2_cells(self, build_pipeline):
        use_case, _ = build_pipeline(model=standard_model())
        out = use_case.execute(AnalysisRequestDTO(encode_image(scene()), "a.png"))
        assert out.damaged_area_m2 is None
        rows = read_csv(out.saved_metrics_path)
        assert all(r["area_m2"] == "" for r in rows)
        summary = json.loads(out.saved_summary_path.read_text(encoding="utf-8"))
        assert summary["metrics"]["damaged_union_area_m2"] is None
        assert "CALIBRATION_UNAVAILABLE" in codes(out.warnings)

    def test_no_damage_still_writes_outputs(self, build_pipeline, results_dir):
        use_case, _ = build_pipeline(model=FakeYolo(masks=None))
        out = use_case.execute(AnalysisRequestDTO(encode_image(scene()), "a.png"))
        assert out.damaged_union_pixels == 0
        assert "NO_DAMAGE_DETECTED" in codes(out.warnings)
        assert len(list(results_dir.iterdir())) == 4
        assert read_csv(out.saved_metrics_path) == []
        assert not cv2.imread(str(out.saved_mask_path), cv2.IMREAD_GRAYSCALE).any()

    def test_same_input_gives_identical_csv(self, build_pipeline):
        fixed = lambda: datetime(2026, 11, 18, 10, 0, 0)  # noqa: E731
        use_case, _ = build_pipeline(model=standard_model(), now=fixed)
        req = AnalysisRequestDTO(encode_image(scene()), "a.png", user_gsd_m_per_pixel=0.5)
        first = use_case.execute(req)
        second = use_case.execute(req)
        assert first.saved_metrics_path != second.saved_metrics_path  # üzerine yazmaz
        assert first.saved_metrics_path.read_text() == second.saved_metrics_path.read_text()
        assert np.array_equal(cv2.imread(str(first.saved_mask_path)),
                              cv2.imread(str(second.saved_mask_path)))


# --------------------------------------------------------------------------- #
# 2. Diğer biçimler ve AOI seçenekleri
# --------------------------------------------------------------------------- #
class TestFormatsAndAoi:
    def test_jpeg_input(self, build_pipeline):
        use_case, _ = build_pipeline(model=standard_model())
        out = use_case.execute(AnalysisRequestDTO(encode_image(scene(), ".jpg"), "post.jpeg"))
        assert out.counts["damaged_region_count"] == 3
        summary = json.loads(out.saved_summary_path.read_text(encoding="utf-8"))
        assert summary["input"]["source_format"] == "jpeg"

    def test_grayscale_png_input(self, build_pipeline):
        use_case, _ = build_pipeline(model=standard_model())
        gray = np.full((H, W), 90, np.uint8)
        out = use_case.execute(AnalysisRequestDTO(encode_image(gray), "gray.png"))
        assert out.original_image.shape == (H, W, 3)

    def test_black_padding_kept_by_default_and_excluded_on_request(self, build_pipeline):
        pixels = scene()
        pixels[:, 80:] = 0  # sağda 16 sütun siyah dolgu
        model = FakeYolo(masks=[rect_mask(70, 10, 90, 20, H, W)], conf=[0.9])  # 200 px, 100'ü dolguda
        use_case, _ = build_pipeline(model=model)

        default = use_case.execute(AnalysisRequestDTO(encode_image(pixels), "a.png"))
        assert default.valid_aoi_pixels == H * W
        assert default.damaged_union_pixels == 200
        assert default.aoi_assumption == "full_image"

        excluded = use_case.execute(
            AnalysisRequestDTO(encode_image(pixels), "a.png", exclude_black_padding=True)
        )
        assert excluded.valid_aoi_pixels == H * 80
        assert excluded.damaged_union_pixels == 100
        assert excluded.objects[0].pixel_area == 200  # tekil maske alanı değişmez
        assert excluded.aoi_assumption == "user_black_padding_excluded"
        assert excluded.damaged_area_ratio == pytest.approx(100 / (H * 80))

    def test_pre_image_is_displayed_but_never_sent_to_model(self, build_pipeline):
        model = standard_model()
        use_case, _ = build_pipeline(model=model)
        pre = scene(30)
        out = use_case.execute(AnalysisRequestDTO(
            encode_image(scene(200)), "post.png",
            pre_image_bytes=encode_image(pre), pre_image_filename="pre.png",
        ))
        assert model.predict_calls == 1
        assert int(model.received.mean()) == 200  # yalnızca post görüntü
        assert np.array_equal(out.pre_image, pre)
        assert "PRE_IMAGE_DISPLAY_ONLY" in codes(out.warnings)


# --------------------------------------------------------------------------- #
# 3. GeoTIFF: georeference -> doğrulanmış m², nodata -> AOI
# --------------------------------------------------------------------------- #
@requires_rasterio
class TestGeoTiffPipeline:
    def test_utm_geotiff_gives_verified_m2(self, build_pipeline, tmp_path):
        data = make_geotiff(tmp_path / "scene.tif", H, W, pixel_size=0.5)
        use_case, _ = build_pipeline(model=standard_model())
        out = use_case.execute(AnalysisRequestDTO(data, "malatya_scene.tif"))
        assert out.calibration_source == "verified_georeference"
        assert out.pixel_area_m2 == pytest.approx(0.25)
        assert out.damaged_area_m2 == pytest.approx(636 * 0.25)
        assert out.gsd_m_per_pixel is None
        summary = json.loads(out.saved_summary_path.read_text(encoding="utf-8"))
        assert summary["input"]["crs"] == "EPSG:32637"
        assert summary["input"]["source_format"] == "geotiff"
        assert "CALIBRATION_UNAVAILABLE" not in codes(out.warnings)

    def test_non_square_pixels(self, build_pipeline, tmp_path):
        from rasterio.transform import Affine

        import rasterio

        path = tmp_path / "rect.tif"
        with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=3, dtype="uint8",
                           crs="EPSG:32637", transform=Affine(0.5, 0, 400000, 0, -0.4, 4250000)) as dst:
            dst.write(np.full((3, H, W), 100, np.uint8))
        use_case, _ = build_pipeline(model=FakeYolo(masks=[rect_mask(10, 10, 20, 20, H, W)], conf=[0.9]))
        out = use_case.execute(AnalysisRequestDTO(path.read_bytes(), "rect.tif"))
        assert out.pixel_area_m2 == pytest.approx(0.2)
        assert out.damaged_area_m2 == pytest.approx(100 * 0.2)

    def test_georeference_overrides_user_gsd(self, build_pipeline, tmp_path):
        data = make_geotiff(tmp_path / "scene.tif", H, W, pixel_size=0.3)
        use_case, _ = build_pipeline(model=standard_model())
        out = use_case.execute(AnalysisRequestDTO(data, "scene.tif", user_gsd_m_per_pixel=2.0))
        assert out.calibration_source == "verified_georeference"
        assert out.pixel_area_m2 == pytest.approx(0.09)
        assert "USER_GSD_IGNORED" in codes(out.warnings)

    def test_geographic_crs_gives_no_m2(self, build_pipeline, tmp_path):
        data = make_geotiff(tmp_path / "wgs.tif", H, W, crs="EPSG:4326", pixel_size=0.00001)
        use_case, _ = build_pipeline(model=standard_model())
        out = use_case.execute(AnalysisRequestDTO(data, "wgs.tif"))
        assert out.calibration_source == "unknown"
        assert out.damaged_area_m2 is None
        assert {"GEOREFERENCE_NOT_METRIC", "CALIBRATION_UNAVAILABLE"} <= codes(out.warnings)

    def test_nodata_area_is_excluded_from_aoi_and_union(self, build_pipeline, tmp_path):
        data = make_geotiff(tmp_path / "nd.tif", H, W, nodata=0, nodata_cols=16)
        model = FakeYolo(masks=[rect_mask(70, 10, 90, 20, H, W)], conf=[0.9])  # 100 px nodata'da
        use_case, _ = build_pipeline(model=model)
        out = use_case.execute(AnalysisRequestDTO(data, "nd.tif"))
        assert out.aoi_assumption == "geotiff_nodata"
        assert out.valid_aoi_pixels == H * (W - 16)
        assert out.damaged_union_pixels == 100
        assert out.objects[0].pixel_area == 200
        assert out.damaged_area_m2 == pytest.approx(100 * 0.25)
        assert "AOI_FULL_IMAGE_ASSUMED" not in codes(out.warnings)


# --------------------------------------------------------------------------- #
# 4. Hata akışları: hiçbir dosya yazılmamalı
# --------------------------------------------------------------------------- #
class TestFailureFlows:
    def test_corrupt_png_writes_nothing(self, build_pipeline, results_dir):
        use_case, _ = build_pipeline(model=standard_model())
        with pytest.raises(InvalidImageError, match="okunamadı"):
            use_case.execute(AnalysisRequestDTO(b"\x89PNG\r\n\x1a\n-broken", "bad.png"))
        assert not results_dir.exists() or not any(results_dir.iterdir())

    def test_missing_weights_writes_nothing(self, build_pipeline, results_dir, tmp_path):
        model = standard_model()
        use_case, segmenter = build_pipeline(model=model, weights=tmp_path / "yok.pt")
        with pytest.raises(ModelWeightsUnavailableError):
            use_case.execute(AnalysisRequestDTO(encode_image(scene()), "a.png"))
        assert model.predict_calls == 0
        assert "bulunamadı" in segmenter.unavailable_reason
        assert not results_dir.exists()

    def test_coco_checkpoint_is_never_used(self, build_pipeline, results_dir):
        coco = standard_model(names={i: f"class_{i}" for i in range(80)})
        use_case, segmenter = build_pipeline(model=coco)
        with pytest.raises(ModelWeightsUnavailableError):
            use_case.execute(AnalysisRequestDTO(encode_image(scene()), "a.png"))
        assert coco.predict_calls == 0
        assert "COCO" in segmenter.unavailable_reason
        assert not results_dir.exists()

    def test_oversized_image_rejected_before_inference(self, build_pipeline, results_dir):
        model = standard_model()
        limits = AnalysisLimits(max_image_side_px=50, allowed_extensions=ALL_EXTENSIONS)
        use_case, _ = build_pipeline(model=model, limits=limits)
        with pytest.raises(InvalidImageError, match="Görüntü çok büyük"):
            use_case.execute(AnalysisRequestDTO(encode_image(scene()), "big.png"))
        assert model.predict_calls == 0
        assert not results_dir.exists()

    def test_tiff_rejected_when_not_allowed(self, build_pipeline):
        use_case, _ = build_pipeline(model=standard_model(), limits=AnalysisLimits())
        with pytest.raises(InvalidImageError, match="Desteklenmeyen"):
            use_case.execute(AnalysisRequestDTO(b"II*\x00", "scene.tif"))

    def test_sixteen_bit_png_is_scaled_and_analysed(self, build_pipeline):
        # 16 bit PNG (bazı uydu dışa aktarımları) 8 bite ölçeklenip analiz edilmeli
        pixels16 = np.full((H, W, 3), 65535 // 2, dtype=np.uint16)
        ok, buf = cv2.imencode(".png", pixels16)
        assert ok
        use_case, _ = build_pipeline(model=standard_model())
        out = use_case.execute(AnalysisRequestDTO(buf.tobytes(), "sixteen.png"))
        assert out.original_image.dtype == np.uint8
        assert int(out.original_image.mean()) in (127, 128)
        assert out.counts["damaged_region_count"] == 3
