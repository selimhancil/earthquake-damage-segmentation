"""Gerçek Ultralytics + PyTorch ile entegrasyon testi.

İnternetten ağırlık İNDİRMEZ: paket içindeki yolov8n-seg.yaml mimarisinden
EĞİTİLMEMİŞ (rastgele ağırlıklı) bir model kurulur, .pt olarak kaydedilir ve
YoloSegmentationService bu dosyayı gerçek `ultralytics.YOLO` ile yükler.

Amaç doğruluk değil, API sözleşmesidir: Results.masks / Results.boxes
ayrıştırması, retina_masks ile tam çözünürlüklü maske, sınıf adı koruması ve
use case'in gerçek model çıktısıyla çökmeden çalışması.

ultralytics veya torch kurulu değilse bu dosya atlanır.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch kurulu değil")
pytest.importorskip("ultralytics", reason="ultralytics kurulu değil")

from src.application.dtos import AnalysisRequestDTO, ImageData  # noqa: E402
from src.domain.exceptions import ModelWeightsUnavailableError  # noqa: E402
from src.infrastructure.yolo_service import YoloSegmentationService  # noqa: E402
from tests.integration.conftest import encode_image  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.slow]

H, W = 64, 96


def save_untrained_checkpoint(path, nc: int, names: dict) -> None:
    from ultralytics.nn.tasks import SegmentationModel

    torch.manual_seed(0)
    model = SegmentationModel("yolov8n-seg.yaml", nc=nc, verbose=False)
    model.names = names
    torch.save({"model": model, "train_args": {"task": "segment", "imgsz": 512}}, path)


@pytest.fixture(scope="module")
def damage_weights(tmp_path_factory):
    path = tmp_path_factory.mktemp("w") / "untrained_damage.pt"
    save_untrained_checkpoint(path, nc=1, names={0: "damaged_region"})
    return path


@pytest.fixture(scope="module")
def generic_weights(tmp_path_factory):
    path = tmp_path_factory.mktemp("w") / "untrained_80cls.pt"
    save_untrained_checkpoint(path, nc=80, names={i: str(i) for i in range(80)})
    return path


def textured_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)


def test_real_yolo_loads_single_class_checkpoint(damage_weights):
    svc = YoloSegmentationService(damage_weights, device="cpu")
    assert svc.is_ready(), svc.unavailable_reason
    info = svc.model_info()
    assert info.class_names == ("damaged_region",)
    assert len(info.weights_sha256) == 64


def test_real_yolo_refuses_multi_class_checkpoint(generic_weights):
    svc = YoloSegmentationService(generic_weights, device="cpu")
    assert svc.is_ready() is False
    assert "damaged_region" in svc.unavailable_reason
    with pytest.raises(ModelWeightsUnavailableError):
        svc.segment(ImageData(textured_image(), W, H, "x.png"), 0.5)


def test_real_yolo_output_contract(damage_weights):
    svc = YoloSegmentationService(damage_weights, device="cpu", imgsz=128)
    instances = svc.segment(ImageData(textured_image(), W, H, "x.png"), confidence_threshold=0.001)
    # Eğitilmemiş model: kaç tahmin çıkacağı belirsiz, ama her tahmin sözleşmeye uymalı
    for inst in instances:
        assert inst.class_id == 0
        assert 0.001 <= inst.confidence <= 1.0
        assert inst.mask.shape == (H, W)
        assert inst.mask.dtype == bool
        x_min, y_min, x_max, y_max = inst.bounding_box
        assert 0 <= x_min <= x_max < W
        assert 0 <= y_min <= y_max < H


def test_real_yolo_through_full_use_case(build_pipeline, damage_weights):
    from src.infrastructure.yolo_service import _default_model_loader

    use_case, segmenter = build_pipeline(model_loader=_default_model_loader, weights=damage_weights)
    out = use_case.execute(
        AnalysisRequestDTO(encode_image(textured_image()), "real_yolo.png",
                           confidence_threshold=0.001, user_gsd_m_per_pixel=0.5)
    )
    assert segmenter.is_ready()
    assert out.valid_aoi_pixels == H * W
    assert 0 <= out.damaged_union_pixels <= H * W
    assert out.damaged_union_pixels <= sum(o.pixel_area for o in out.objects) or not out.objects
    assert out.annotated_image.shape == (H, W, 3)
    summary = json.loads(out.saved_summary_path.read_text(encoding="utf-8"))
    assert summary["metrics"]["damaged_union_pixels"] == out.damaged_union_pixels
    assert summary["model"]["weights_path"].endswith("untrained_damage.pt")
