"""Arayüzün ihtiyaç duyduğu bağımlılıklar (Streamlit'ten bağımsız).

main.py'deki composition root bu nesneyi kurar ve run_app'e verir.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, FrozenSet, Optional, Tuple

from src.application.interfaces import ISegmentationService
from src.application.use_cases import AnalyzeSatelliteImageUseCase


@dataclass(frozen=True)
class AppContext:
    use_case: AnalyzeSatelliteImageUseCase
    segmentation_service: ISegmentationService
    allowed_extensions: FrozenSet[str]
    default_confidence_threshold: float
    max_image_mb: float
    weights_path: Path
    geotiff_enabled: bool
    # Demo modu: sahte model + sentetik örnek sahne. Sonuçlar DEMO ONLY'dir.
    demo_mode: bool = False
    demo_sample_provider: Optional[Callable[[], Tuple[bytes, bytes]]] = None  # (öncesi, sonrası) PNG
