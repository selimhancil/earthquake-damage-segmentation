"""Demo modu testleri: sahte sonuçların her yerde DEMO ONLY olarak işaretlendiğini doğrular."""

import json

import cv2
import numpy as np
import pytest

import config
from src.application.dtos import AnalysisRequestDTO, ImageData
from src.application.use_cases import AnalysisLimits, AnalyzeSatelliteImageUseCase
from src.infrastructure.cv2_service import (
    CV2ImageLoader,
    CV2MeasurementService,
    CV2VisualizationService,
)
from src.infrastructure.demo_service import (
    DemoSegmentationService,
    DemoWatermarkVisualization,
    add_demo_watermark,
    demo_scene_png_pair,
    make_demo_scene,
)
from src.infrastructure.file_service import FileOutputWriter
from src.infrastructure.yolo_service import YoloSegmentationService
from src.presentation import components as ui

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover
    AppTest = None

requires_streamlit = pytest.mark.skipif(AppTest is None, reason="streamlit kurulu değil")


def as_image(rgb: np.ndarray, name: str = "x.png") -> ImageData:
    return ImageData(pixels=rgb, width=rgb.shape[1], height=rgb.shape[0], filename=name)


@pytest.fixture(scope="module")
def scene():
    return make_demo_scene()


# --------------------------------------------------------------------------- #
# Sentetik sahne ve sezgisel segmentasyon
# --------------------------------------------------------------------------- #
class TestDemoSegmentation:
    def test_scene_shapes_and_determinism(self, scene):
        pre, post = scene
        assert pre.shape == post.shape == (480, 640, 3)
        assert pre.dtype == np.uint8
        again_pre, again_post = make_demo_scene()
        assert np.array_equal(pre, again_pre) and np.array_equal(post, again_post)
        assert not np.array_equal(pre, post)

    def test_intact_scene_has_no_regions(self, scene):
        pre, _ = scene
        assert DemoSegmentationService().segment(as_image(pre), 0.5) == []

    def test_collapsed_buildings_are_found(self, scene):
        _, post = scene
        instances = DemoSegmentationService().segment(as_image(post), 0.5)
        assert len(instances) == 4
        for inst in instances:
            assert inst.class_id == 0
            assert 0.5 <= inst.confidence <= 0.99
            assert inst.mask.shape == (480, 640) and inst.mask.dtype == bool
            x0, y0, x1, y1 = inst.bounding_box
            assert 0 <= x0 <= x1 < 640 and 0 <= y0 <= y1 < 480
        # sağ kenara taşan moloz alanı kenara değmeli
        assert any(inst.bounding_box[2] == 639 for inst in instances)

    def test_threshold_filters_regions(self, scene):
        _, post = scene
        assert DemoSegmentationService().segment(as_image(post), 0.999) == []

    def test_is_deterministic(self, scene):
        _, post = scene
        a = DemoSegmentationService().segment(as_image(post), 0.5)
        b = DemoSegmentationService().segment(as_image(post), 0.5)
        assert [i.bounding_box for i in a] == [i.bounding_box for i in b]
        assert [i.confidence for i in a] == [i.confidence for i in b]

    def test_model_info_is_marked_demo(self):
        info = DemoSegmentationService().model_info()
        assert info.is_demo is True
        assert "DEMO" in info.weights_path
        assert info.weights_sha256 is None

    def test_real_yolo_service_is_not_demo(self, tmp_path):
        assert YoloSegmentationService(tmp_path / "x.pt").model_info().is_demo is False

    def test_png_pair_decodes(self):
        pre_png, post_png = demo_scene_png_pair()
        for data in (pre_png, post_png):
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            assert img.shape == (480, 640, 3)


# --------------------------------------------------------------------------- #
# Filigran
# --------------------------------------------------------------------------- #
class TestWatermark:
    def test_watermark_marks_top_left_only(self):
        rgb = np.full((200, 300, 3), 128, np.uint8)
        out = add_demo_watermark(rgb)
        assert out.shape == rgb.shape
        assert not np.array_equal(out[:20, :60], rgb[:20, :60])
        assert np.array_equal(out[150:, 200:], rgb[150:, 200:])
        assert (rgb == 128).all()  # girdi değişmez

    def test_wrapper_watermarks_overlay_but_not_mask(self, scene):
        _, post = scene
        image = as_image(post)
        instances = DemoSegmentationService().segment(image, 0.5)
        flags = [False] * len(instances)
        labels = [""] * len(instances)
        plain = CV2VisualizationService()
        wrapped = DemoWatermarkVisualization(plain)
        assert not np.array_equal(
            wrapped.render_overlay(image, instances, flags, labels),
            plain.render_overlay(image, instances, flags, labels),
        )
        assert np.array_equal(
            wrapped.render_union_mask(image, instances), plain.render_union_mask(image, instances)
        )


# --------------------------------------------------------------------------- #
# Uçtan uca: DEMO ONLY işaretleri
# --------------------------------------------------------------------------- #
def demo_use_case(tmp_path):
    return AnalyzeSatelliteImageUseCase(
        image_loader=CV2ImageLoader(),
        segmentation_service=DemoSegmentationService(),
        measurement_service=CV2MeasurementService(),
        visualization_service=DemoWatermarkVisualization(CV2VisualizationService()),
        output_writer=FileOutputWriter(tmp_path),
        limits=AnalysisLimits(),
    )


class TestDemoPipeline:
    def test_outputs_are_marked_demo_everywhere(self, tmp_path):
        pre_png, post_png = demo_scene_png_pair()
        out = demo_use_case(tmp_path).execute(
            AnalysisRequestDTO(post_png, "scene.png", pre_image_bytes=pre_png,
                               pre_image_filename="pre.png")
        )
        # DTO
        assert out.demo_mode is True
        assert out.warnings[0].startswith("DEMO_ONLY:")
        assert out.counts["damaged_region_count"] == 4
        # Dosya adları
        for path in (out.saved_image_path, out.saved_mask_path,
                     out.saved_metrics_path, out.saved_summary_path):
            assert path.name.startswith("DEMO_scene_")
            assert path.is_file()
        # JSON özeti
        summary = json.loads(out.saved_summary_path.read_text(encoding="utf-8"))
        assert summary["demo_mode"] is True
        assert summary["model"]["family"] == "demo-texture-heuristic"
        assert summary["warnings"][0].startswith("DEMO_ONLY:")
        # Kaydedilen işaretli görüntüde filigran var (sol üst köşe siyah zemin)
        saved = cv2.imread(str(out.saved_image_path))
        assert (saved[2:6, 2:6] < 30).all()
        # Maske ölçümle tutarlı (filigran maskeye basılmaz)
        mask = cv2.imread(str(out.saved_mask_path), cv2.IMREAD_GRAYSCALE)
        assert int(np.count_nonzero(mask)) == out.damaged_union_pixels

    def test_real_model_path_has_no_demo_markers(self, tmp_path):
        from tests.fakes import FakeYolo, rect_mask

        weights = tmp_path / "best_model.pt"
        weights.write_bytes(b"w")
        model = FakeYolo(masks=[rect_mask(5, 5, 15, 15, 32, 48)], conf=[0.9])
        use_case = AnalyzeSatelliteImageUseCase(
            CV2ImageLoader(), YoloSegmentationService(weights, model_loader=lambda _p: model),
            CV2MeasurementService(), CV2VisualizationService(), FileOutputWriter(tmp_path / "r"),
        )
        ok, buf = cv2.imencode(".png", np.zeros((32, 48, 3), np.uint8))
        out = use_case.execute(AnalysisRequestDTO(buf.tobytes(), "real.png"))
        assert out.demo_mode is False
        assert not any(w.startswith("DEMO_ONLY") for w in out.warnings)
        assert not out.saved_summary_path.name.startswith("DEMO_")
        summary = json.loads(out.saved_summary_path.read_text(encoding="utf-8"))
        assert summary["demo_mode"] is False


# --------------------------------------------------------------------------- #
# Ayarlar ve arayüz yardımcıları
# --------------------------------------------------------------------------- #
class TestConfigAndComponents:
    @pytest.mark.parametrize("value, expected", [
        ("1", True), ("true", True), ("YES", True), ("on", True), ("evet", True),
        ("0", False), ("false", False), ("", False), ("no", False),
    ])
    def test_env_flag(self, monkeypatch, value, expected):
        monkeypatch.setenv("EV_TEST_FLAG", value)
        assert config.env_flag("EV_TEST_FLAG") is expected

    def test_env_flag_default(self, monkeypatch):
        monkeypatch.delenv("EV_TEST_FLAG", raising=False)
        assert config.env_flag("EV_TEST_FLAG") is False
        assert config.env_flag("EV_TEST_FLAG", default=True) is True

    def test_demo_results_go_to_separate_folder(self):
        assert config.DEMO_RESULTS_DIR.parent == config.RESULTS_DIR

    def test_demo_warning_is_shown_first(self):
        parsed = ui.classify_warnings(["SINGLE_DATE_MODEL: a", "DEMO_ONLY: b"])
        assert parsed[0][:2] == ("warning", "DEMO_ONLY")

    def test_training_instructions_mention_demo(self):
        assert "EV_DEMO_MODE=1" in ui.TRAINING_INSTRUCTIONS


# --------------------------------------------------------------------------- #
# Composition root ve arayüz (streamlit gerekir)
# --------------------------------------------------------------------------- #
@requires_streamlit
class TestDemoApp:
    def test_build_context_in_demo_mode(self):
        import main

        ctx = main.build_context(demo_mode=True)
        assert ctx.demo_mode is True
        assert ctx.segmentation_service.is_ready()
        assert ctx.segmentation_service.model_info().is_demo
        pre_png, post_png = ctx.demo_sample_provider()
        assert pre_png and post_png

    def test_build_context_normal_mode_uses_yolo(self):
        import main

        ctx = main.build_context(demo_mode=False)
        assert ctx.demo_mode is False
        assert ctx.demo_sample_provider is None
        assert isinstance(ctx.segmentation_service, YoloSegmentationService)

    def test_demo_banner_and_sample_button(self):
        def script():
            from src.presentation.app import run_app
            from tests.presentation_support import build_demo_context

            run_app(build_demo_context)

        at = AppTest.from_function(script, default_timeout=60).run()
        assert not at.exception
        assert "DEMO ONLY" in " ".join(w.value for w in at.warning)
        labels = [b.label for b in at.button]
        assert "Sentetik örnek sahneyle dene (DEMO)" in labels

        sample = next(b for b in at.button if b.label.startswith("Sentetik"))
        sample.click().run()
        assert not at.exception
        assert "DEMO ONLY" in " ".join(e.value for e in at.error)
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Hasar bölgesi sayısı"] == "4"
        assert any("DEMO_ONLY" in w.value for w in at.warning)
