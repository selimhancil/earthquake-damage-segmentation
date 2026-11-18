"""Arayüz testleri için yardımcılar: sahte modelle kurulmuş gerçek bir AppContext.

AppTest.from_function ile çalıştırılan betikler closure taşıyamadığı için
bağlamı bu modülden import ederek kurarlar.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.application.dtos import AnalysisOutputDTO, AnalysisRequestDTO
from src.application.use_cases import AnalysisLimits, AnalyzeSatelliteImageUseCase
from src.infrastructure.cv2_service import (
    CV2ImageLoader,
    CV2MeasurementService,
    CV2VisualizationService,
)
from src.infrastructure.file_service import FileOutputWriter
from src.infrastructure.raster_service import CompositeImageLoader
from src.infrastructure.yolo_service import YoloSegmentationService
from src.presentation.context import AppContext
from tests.fakes import FakeYolo, rect_mask

H, W = 64, 96
EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})


def build_fake_context(model=None, ready: bool = True) -> AppContext:
    workdir = Path(tempfile.mkdtemp(prefix="ev_ui_"))
    weights = workdir / "best_model.pt"
    if ready:
        weights.write_bytes(b"fake")
    model = model or FakeYolo(
        masks=[rect_mask(10, 10, 30, 30, H, W), rect_mask(20, 10, 40, 30, H, W),
               rect_mask(0, 0, 6, 6, H, W)],
        conf=[0.95, 0.72, 0.80],
    )
    segmenter = YoloSegmentationService(weights, model_loader=lambda _p: model)
    use_case = AnalyzeSatelliteImageUseCase(
        image_loader=CompositeImageLoader(CV2ImageLoader()),
        segmentation_service=segmenter,
        measurement_service=CV2MeasurementService(),
        visualization_service=CV2VisualizationService(),
        output_writer=FileOutputWriter(workdir / "results",
                                       now_fn=lambda: datetime(2026, 11, 18, 12, 0, 0)),
        limits=AnalysisLimits(allowed_extensions=EXTENSIONS),
    )
    return AppContext(
        use_case=use_case,
        segmentation_service=segmenter,
        allowed_extensions=EXTENSIONS,
        default_confidence_threshold=0.70,
        max_image_mb=15,
        weights_path=weights,
        geotiff_enabled=False,
    )


def png_bytes(value: int = 120) -> bytes:
    ok, buf = cv2.imencode(".png", np.full((H, W, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def sample_output(gsd=0.5, pre: bool = False) -> AnalysisOutputDTO:
    ctx = build_fake_context()
    return ctx.use_case.execute(
        AnalysisRequestDTO(
            png_bytes(), "malatya_post.png", user_gsd_m_per_pixel=gsd,
            pre_image_bytes=png_bytes(40) if pre else None,
            pre_image_filename="malatya_pre.png" if pre else None,
        )
    )


def build_demo_context() -> AppContext:
    """Demo modundaki AppContext (gerçek adapter'lar + demo segmentasyon, geçici klasör)."""
    from src.infrastructure.demo_service import (
        DemoSegmentationService,
        DemoWatermarkVisualization,
        demo_scene_png_pair,
    )

    workdir = Path(tempfile.mkdtemp(prefix="ev_demo_"))
    segmenter = DemoSegmentationService()
    use_case = AnalyzeSatelliteImageUseCase(
        image_loader=CompositeImageLoader(CV2ImageLoader()),
        segmentation_service=segmenter,
        measurement_service=CV2MeasurementService(),
        visualization_service=DemoWatermarkVisualization(CV2VisualizationService()),
        output_writer=FileOutputWriter(workdir / "results"),
        limits=AnalysisLimits(allowed_extensions=EXTENSIONS),
    )
    return AppContext(
        use_case=use_case,
        segmentation_service=segmenter,
        allowed_extensions=EXTENSIONS,
        default_confidence_threshold=0.70,
        max_image_mb=15,
        weights_path=workdir / "best_model.pt",
        geotiff_enabled=False,
        demo_mode=True,
        demo_sample_provider=demo_scene_png_pair,
    )
