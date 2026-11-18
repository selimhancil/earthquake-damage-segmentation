"""Composition root ve Streamlit giriş noktası.

Çalıştırma:
    streamlit run main.py                   # gerçek model (weights/best_model.pt)
    EV_DEMO_MODE=1 streamlit run main.py    # DEMO ONLY: model olmadan arayüz denemesi

Somut adapter'lar yalnızca burada kurulur ve use case'e enjekte edilir.
Model ve servisler @st.cache_resource ile oturumlar arasında bir kez yüklenir.
"""

from __future__ import annotations

import logging

import streamlit as st

import config
from src.application.use_cases import AnalysisLimits, AnalyzeSatelliteImageUseCase
from src.infrastructure.cv2_service import (
    CV2ImageLoader,
    CV2MeasurementService,
    CV2VisualizationService,
)
from src.infrastructure.demo_service import (
    DemoSegmentationService,
    DemoWatermarkVisualization,
    demo_scene_png_pair,
)
from src.infrastructure.file_service import FileOutputWriter
from src.infrastructure.raster_service import CompositeImageLoader, RasterioImageLoader
from src.infrastructure.yolo_service import YoloSegmentationService
from src.presentation.app import run_app
from src.presentation.context import AppContext


def configure_logging() -> None:
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATEFMT,
    )


def build_context(demo_mode: bool = config.DEMO_MODE) -> AppContext:
    """Tüm bağımlılıkları kurar (Dependency Injection).

    demo_mode=True: gerçek YOLO yerine DEMO ONLY sezgisel segmentasyon, filigranlı
    görseller ve ayrı demo çıktı klasörü kullanılır.
    """
    geotiff_loader = RasterioImageLoader() if config.GEOTIFF_ENABLED else None
    visualizer = CV2VisualizationService()
    if demo_mode:
        segmenter = DemoSegmentationService()
        visualizer = DemoWatermarkVisualization(visualizer)
        results_dir = config.DEMO_RESULTS_DIR
    else:
        segmenter = YoloSegmentationService(
            weights_path=config.WEIGHTS_PATH,
            device=config.YOLO_DEVICE,
            imgsz=config.YOLO_IMGSZ,
        )
        results_dir = config.RESULTS_DIR
    limits = AnalysisLimits(
        default_confidence_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD,
        max_image_mb=config.MAX_IMAGE_MB,
        max_image_side_px=config.MAX_IMAGE_SIDE_PX,
        allowed_extensions=config.ALLOWED_EXTENSIONS,
    )
    use_case = AnalyzeSatelliteImageUseCase(
        image_loader=CompositeImageLoader(CV2ImageLoader(), geotiff_loader),
        segmentation_service=segmenter,
        measurement_service=CV2MeasurementService(),
        visualization_service=visualizer,
        output_writer=FileOutputWriter(results_dir),
        limits=limits,
    )
    return AppContext(
        use_case=use_case,
        segmentation_service=segmenter,
        allowed_extensions=config.ALLOWED_EXTENSIONS,
        default_confidence_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD,
        confidence_threshold_source=config.CONFIDENCE_THRESHOLD_SOURCE,
        max_image_mb=config.MAX_IMAGE_MB,
        weights_path=config.WEIGHTS_PATH,
        geotiff_enabled=config.GEOTIFF_ENABLED,
        demo_mode=demo_mode,
        demo_sample_provider=demo_scene_png_pair if demo_mode else None,
    )


@st.cache_resource(show_spinner="Model yükleniyor...")
def get_context() -> AppContext:
    configure_logging()
    log = logging.getLogger(__name__)
    if config.DEMO_MODE:
        log.warning("DEMO MODU AÇIK: eğitilmiş model kullanılmıyor, sonuçlar DEMO ONLY")
    log.info("Uygulama bağımlılıkları kuruluyor")
    return build_context()


# Streamlit betiği __name__ == "__main__" olarak çalıştırır; testler import edebilir.
if __name__ == "__main__":
    run_app(get_context)
