"""Composition root ve Streamlit giriş noktası.

Çalıştırma:
    streamlit run main.py

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


def build_context() -> AppContext:
    """Tüm bağımlılıkları kurar (Dependency Injection)."""
    geotiff_loader = RasterioImageLoader() if config.GEOTIFF_ENABLED else None
    segmenter = YoloSegmentationService(
        weights_path=config.WEIGHTS_PATH,
        device=config.YOLO_DEVICE,
        imgsz=config.YOLO_IMGSZ,
    )
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
        visualization_service=CV2VisualizationService(),
        output_writer=FileOutputWriter(config.RESULTS_DIR),
        limits=limits,
    )
    return AppContext(
        use_case=use_case,
        segmentation_service=segmenter,
        allowed_extensions=config.ALLOWED_EXTENSIONS,
        default_confidence_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD,
        max_image_mb=config.MAX_IMAGE_MB,
        weights_path=config.WEIGHTS_PATH,
        geotiff_enabled=config.GEOTIFF_ENABLED,
    )


@st.cache_resource(show_spinner="Model yükleniyor...")
def get_context() -> AppContext:
    configure_logging()
    logging.getLogger(__name__).info("Uygulama bağımlılıkları kuruluyor")
    return build_context()


run_app(get_context)
