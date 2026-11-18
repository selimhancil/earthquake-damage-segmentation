"""YOLO servisi, dosya kaydı, GeoTIFF yükleyici ve uçtan uca entegrasyon testleri.

Gerçek Ultralytics modeli YÜKLENMEZ: model_loader enjekte edilerek sahte bir
model kullanılır. GeoTIFF testleri rasterio kurulu değilse atlanır.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.application.dtos import AnalysisRequestDTO, ImageData, SpatialMetadata
from src.application.use_cases import AnalysisLimits, AnalyzeSatelliteImageUseCase
from src.domain.exceptions import InvalidImageError, ModelWeightsUnavailableError
from src.infrastructure.cv2_service import (
    CV2ImageLoader,
    CV2MeasurementService,
    CV2VisualizationService,
)
from src.infrastructure.file_service import CSV_COLUMNS, FileOutputWriter
from src.infrastructure.raster_service import CompositeImageLoader, RasterioImageLoader
from src.infrastructure.yolo_service import YoloSegmentationService, bbox_from_mask

H, W = 64, 96


# --------------------------------------------------------------------------- #
# Sahte Ultralytics modeli
# --------------------------------------------------------------------------- #
class _Tensor:
    """torch.Tensor'ın .cpu().numpy() arayüzünü taklit eder."""

    def __init__(self, array):
        self._a = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _Tensor(xyxy), _Tensor(conf), _Tensor(cls)


class _Masks:
    def __init__(self, data):
        self.data = _Tensor(data)


class _Result:
    def __init__(self, masks, boxes):
        self.masks, self.boxes = masks, boxes


class FakeYolo:
    def __init__(self, names=None, masks=None, conf=None, mask_size=(H, W)):
        self.names = names if names is not None else {0: "damaged_region"}
        self.mask_size = mask_size
        self._masks = masks
        self._conf = conf
        self.predict_kwargs = None
        self.received = None

    def predict(self, source, **kwargs):
        self.received = source
        self.predict_kwargs = kwargs
        if self._masks is None:
            return [_Result(None, None)]
        n = len(self._masks)
        xyxy = np.tile([0.0, 0.0, 1.0, 1.0], (n, 1))
        return [_Result(_Masks(np.stack(self._masks).astype(np.float32)),
                        _Boxes(xyxy, self._conf, np.zeros(n)))]


@pytest.fixture
def weights(tmp_path) -> Path:
    path = tmp_path / "best_model.pt"
    path.write_bytes(b"fake-weights")
    return path


def rgb_image(h=H, w=W) -> ImageData:
    pixels = np.zeros((h, w, 3), np.uint8)
    pixels[..., 0] = 200  # kırmızı kanal
    return ImageData(pixels=pixels, width=w, height=h, filename="t.png")


def rect(x0, y0, x1, y1, h=H, w=W):
    m = np.zeros((h, w), np.float32)
    m[y0:y1, x0:x1] = 1.0
    return m


# --------------------------------------------------------------------------- #
# YOLO servisi
# --------------------------------------------------------------------------- #
class TestYoloService:
    def test_missing_weights_not_ready(self, tmp_path):
        svc = YoloSegmentationService(tmp_path / "yok.pt", model_loader=lambda p: FakeYolo())
        assert svc.is_ready() is False
        assert "bulunamadı" in svc.unavailable_reason
        with pytest.raises(ModelWeightsUnavailableError):
            svc.segment(rgb_image(), 0.7)

    def test_coco_checkpoint_is_refused(self, weights):
        coco = FakeYolo(names={0: "person", 1: "bicycle", 2: "car"})
        svc = YoloSegmentationService(weights, model_loader=lambda p: coco)
        assert svc.is_ready() is False
        assert "COCO" in svc.unavailable_reason

    def test_loader_exception_is_wrapped(self, weights):
        def boom(_):
            raise RuntimeError("bozuk dosya")

        svc = YoloSegmentationService(weights, model_loader=boom)
        assert svc.is_ready() is False
        assert "bozuk dosya" in svc.unavailable_reason

    def test_ready_and_model_info(self, weights):
        svc = YoloSegmentationService(weights, model_loader=lambda p: FakeYolo())
        assert svc.is_ready()
        info = svc.model_info()
        assert info.weights_path == str(weights)
        assert len(info.weights_sha256) == 64
        assert info.class_names == ("damaged_region",)

    def test_names_as_list_are_accepted(self, weights):
        svc = YoloSegmentationService(weights, model_loader=lambda p: FakeYolo(names=["damaged_region"]))
        assert svc.is_ready()

    def test_segment_parses_instances(self, weights):
        model = FakeYolo(masks=[rect(10, 5, 30, 25), rect(0, 0, 8, 8)], conf=[0.91, 0.75])
        svc = YoloSegmentationService(weights, model_loader=lambda p: model, device="cpu")
        out = svc.segment(rgb_image(), 0.6)
        assert len(out) == 2
        first = out[0]
        assert first.class_id == 0
        assert first.confidence == pytest.approx(0.91)
        assert first.mask.shape == (H, W) and first.mask.dtype == bool
        assert int(first.mask.sum()) == 20 * 20
        assert first.bounding_box == (10, 5, 29, 24)  # maskeden sıkı bbox
        assert out[1].bounding_box == (0, 0, 7, 7)

    def test_predict_arguments_and_bgr_input(self, weights):
        model = FakeYolo(masks=[rect(0, 0, 4, 4)], conf=[0.9])
        svc = YoloSegmentationService(weights, model_loader=lambda p: model, imgsz=512, device="mps")
        svc.segment(rgb_image(), 0.42)
        kw = model.predict_kwargs
        assert kw["conf"] == 0.42 and kw["imgsz"] == 512
        assert kw["retina_masks"] is True and kw["device"] == "mps"
        # RGB (200,0,0) -> BGR (0,0,200)
        assert tuple(model.received[0, 0]) == (0, 0, 200)

    def test_low_resolution_masks_are_resized_nearest(self, weights):
        small = np.zeros((16, 24), np.float32)
        small[4:8, 6:12] = 1.0  # 4x6 = 24 px -> x4 ölçekte 384 px
        model = FakeYolo(masks=[small], conf=[0.8])
        svc = YoloSegmentationService(weights, model_loader=lambda p: model)
        inst = svc.segment(rgb_image(), 0.5)[0]
        assert inst.mask.shape == (H, W)
        assert int(inst.mask.sum()) == 24 * 16

    def test_no_detections(self, weights):
        svc = YoloSegmentationService(weights, model_loader=lambda p: FakeYolo(masks=None))
        assert svc.segment(rgb_image(), 0.7) == []

    def test_model_is_loaded_once(self, weights):
        calls = []

        def loader(p):
            calls.append(p)
            return FakeYolo()

        svc = YoloSegmentationService(weights, model_loader=loader)
        svc.is_ready()
        svc.is_ready()
        svc.segment(rgb_image(), 0.7)
        assert len(calls) == 1

    def test_bbox_from_mask(self):
        m = np.zeros((10, 10), bool)
        assert bbox_from_mask(m) is None
        m[2:5, 3:7] = True
        assert bbox_from_mask(m) == (3, 2, 6, 4)


# --------------------------------------------------------------------------- #
# Dosya kaydı
# --------------------------------------------------------------------------- #
FIXED_NOW = lambda: datetime(2026, 11, 18, 9, 30, 0)  # noqa: E731


class TestFileOutputWriter:
    def save(self, writer, rows=None, summary=None, name="malatya post.png"):
        annotated = np.zeros((H, W, 3), np.uint8)
        annotated[..., 0] = 255  # RGB kırmızı
        mask = np.zeros((H, W), np.uint8)
        mask[0:10, 0:10] = 255
        return writer.save(
            base_name=name,
            annotated_image=annotated,
            mask_image=mask,
            region_rows=rows if rows is not None else [
                {"id": 1, "object_class": "damaged_region", "confidence": 0.9, "pixel_area": 100,
                 "area_m2": None, "is_partial": False, "bbox_x_min": 0, "bbox_y_min": 0,
                 "bbox_x_max": 9, "bbox_y_max": 9}
            ],
            summary=summary if summary is not None else {"metrics": {"damaged_union_pixels": np.int64(100)}},
        )

    def test_writes_four_files(self, tmp_path):
        paths = self.save(FileOutputWriter(tmp_path / "results", now_fn=FIXED_NOW))
        for p in (paths.annotated_image_path, paths.mask_image_path,
                  paths.metrics_csv_path, paths.summary_json_path):
            assert p.is_file()
        assert paths.annotated_image_path.name == "malatya_post.png_20261118_093000_annotated.png"

    def test_png_is_saved_in_correct_color_order(self, tmp_path):
        paths = self.save(FileOutputWriter(tmp_path, now_fn=FIXED_NOW))
        bgr = cv2.imread(str(paths.annotated_image_path))
        assert tuple(bgr[0, 0]) == (0, 0, 255)  # OpenCV BGR'de kırmızı
        mask = cv2.imread(str(paths.mask_image_path), cv2.IMREAD_GRAYSCALE)
        assert np.count_nonzero(mask) == 100

    def test_csv_content(self, tmp_path):
        paths = self.save(FileOutputWriter(tmp_path, now_fn=FIXED_NOW))
        with paths.metrics_csv_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert list(rows[0].keys()) == CSV_COLUMNS
        assert rows[0]["pixel_area"] == "100"
        assert rows[0]["area_m2"] == ""  # None -> boş hücre (N/A)

    def test_csv_header_only_when_no_regions(self, tmp_path):
        paths = self.save(FileOutputWriter(tmp_path, now_fn=FIXED_NOW), rows=[])
        lines = paths.metrics_csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert lines == [",".join(CSV_COLUMNS)]

    def test_summary_json_handles_numpy_and_turkish(self, tmp_path):
        summary = {"metrics": {"x": np.int64(5), "y": np.float32(0.5)}, "note": "Hasarlı bölge"}
        paths = self.save(FileOutputWriter(tmp_path, now_fn=FIXED_NOW), summary=summary)
        text = paths.summary_json_path.read_text(encoding="utf-8")
        data = json.loads(text)
        assert data["metrics"] == {"x": 5, "y": 0.5}
        assert "Hasarlı bölge" in text  # ensure_ascii=False
        assert data["outputs"]["metrics_csv"] == paths.metrics_csv_path.name

    def test_same_second_does_not_overwrite(self, tmp_path):
        writer = FileOutputWriter(tmp_path, now_fn=FIXED_NOW)
        first = self.save(writer)
        second = self.save(writer)
        assert first.summary_json_path != second.summary_json_path
        assert first.summary_json_path.is_file() and second.summary_json_path.is_file()

    def test_unsafe_names_are_sanitized(self, tmp_path):
        paths = self.save(FileOutputWriter(tmp_path, now_fn=FIXED_NOW), name="../../etc/passwd")
        assert paths.summary_json_path.parent == tmp_path
        assert "/" not in paths.summary_json_path.name


# --------------------------------------------------------------------------- #
# Yükleyici yönlendirme ve GeoTIFF
# --------------------------------------------------------------------------- #
class TestCompositeLoader:
    def test_png_goes_to_default_loader(self):
        ok, buf = cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))
        img = CompositeImageLoader(CV2ImageLoader()).load(buf.tobytes(), "a.png")
        assert img.width == 8

    def test_tif_without_geotiff_loader_is_rejected(self):
        with pytest.raises(InvalidImageError, match="GeoTIFF"):
            CompositeImageLoader(CV2ImageLoader()).load(b"x", "scene.tif")


try:
    import rasterio as _rasterio
except ImportError:  # pragma: no cover - rasterio opsiyonel
    _rasterio = None


@pytest.mark.skipif(_rasterio is None, reason="rasterio kurulu değil")
class TestRasterioLoader:
    rasterio = _rasterio

    def make_tiff(self, tmp_path, count=3, dtype="uint8", crs="EPSG:32637", nodata=None,
                  colorinterp=None):
        from rasterio.transform import from_origin

        path = tmp_path / "scene.tif"
        data = np.zeros((count, 20, 30), dtype=dtype)
        data[0] = 100
        if nodata is not None:
            data[:, :, :5] = nodata
        with self.rasterio.open(
            path, "w", driver="GTiff", height=20, width=30, count=count, dtype=dtype,
            crs=crs, transform=from_origin(400000, 4250000, 0.5, 0.5), nodata=nodata,
        ) as dst:
            dst.write(data)
            if colorinterp is not None:
                dst.colorinterp = colorinterp
        return path.read_bytes()

    def test_utm_geotiff_metadata(self, tmp_path):
        img = RasterioImageLoader().load(self.make_tiff(tmp_path), "scene.tif")
        assert img.pixels.shape == (20, 30, 3) and img.pixels.dtype == np.uint8
        assert img.metadata.has_georeference
        assert img.metadata.crs_is_projected_metric
        assert img.metadata.affine[0] == pytest.approx(0.5)
        assert img.metadata.affine[4] == pytest.approx(-0.5)
        assert img.metadata.source_format == "geotiff"

    def test_geographic_geotiff_is_not_metric(self, tmp_path):
        img = RasterioImageLoader().load(self.make_tiff(tmp_path, crs="EPSG:4326"), "scene.tif")
        assert img.metadata.has_georeference
        assert not img.metadata.crs_is_projected_metric

    def test_nodata_mask_is_read(self, tmp_path):
        img = RasterioImageLoader().load(self.make_tiff(tmp_path, nodata=0), "scene.tif")
        assert img.has_nodata_mask
        assert img.nodata_mask.shape == (20, 30)
        assert np.count_nonzero(img.nodata_mask) == 20 * 25

    def test_single_band_becomes_rgb(self, tmp_path):
        img = RasterioImageLoader().load(self.make_tiff(tmp_path, count=1), "scene.tif")
        assert img.pixels.shape == (20, 30, 3)

    def test_four_unlabelled_bands_are_rejected(self, tmp_path):
        with pytest.raises(InvalidImageError, match="bantlı"):
            RasterioImageLoader().load(self.make_tiff(tmp_path, count=4), "scene.tif")

    def test_uint16_is_stretched(self, tmp_path):
        img = RasterioImageLoader().load(self.make_tiff(tmp_path, dtype="uint16"), "scene.tif")
        assert img.pixels.dtype == np.uint8


# --------------------------------------------------------------------------- #
# Uçtan uca: gerçek OpenCV servisleri + sahte YOLO modeli
# --------------------------------------------------------------------------- #
def test_end_to_end_pipeline_writes_outputs(tmp_path, weights):
    model = FakeYolo(
        masks=[rect(10, 10, 30, 30), rect(20, 10, 40, 30), rect(0, 0, 6, 6), rect(50, 40, 60, 50)],
        conf=[0.95, 0.85, 0.80, 0.30],  # sonuncusu eşik altı
    )
    use_case = AnalyzeSatelliteImageUseCase(
        image_loader=CompositeImageLoader(CV2ImageLoader()),
        segmentation_service=YoloSegmentationService(weights, model_loader=lambda p: model),
        measurement_service=CV2MeasurementService(),
        visualization_service=CV2VisualizationService(),
        output_writer=FileOutputWriter(tmp_path / "results", now_fn=FIXED_NOW),
        limits=AnalysisLimits(),
    )
    ok, buf = cv2.imencode(".png", np.full((H, W, 3), 90, np.uint8))
    out = use_case.execute(
        AnalysisRequestDTO(buf.tobytes(), "malatya_post.png", user_gsd_m_per_pixel=0.5)
    )

    assert out.counts == {"damaged_region_count": 3, "complete_region_count": 2,
                          "partial_region_count": 1}
    # 400 + 400 - 200 çakışma + 36 kenar = 636 piksel
    assert out.damaged_union_pixels == 636
    assert out.damaged_area_m2 == pytest.approx(636 * 0.25)
    assert out.valid_aoi_pixels == H * W
    assert out.annotated_image.shape == (H, W, 3)

    summary = json.loads(out.saved_summary_path.read_text(encoding="utf-8"))
    assert summary["metrics"]["damaged_union_pixels"] == 636
    assert summary["metrics"]["sum_of_region_pixels"] == 836
    assert summary["model"]["weights_sha256"]
    assert out.saved_image_path.is_file() and out.saved_mask_path.is_file()
    assert out.saved_metrics_path.read_text(encoding="utf-8").count("\n") == 4  # başlık + 3
