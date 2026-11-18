"""Ultralytics YOLOv8-Seg adapter'ı (ISegmentationService).

Güvenlik kuralı (şartname 7.8): Yalnızca 'damaged_region' sınıfıyla eğitilmiş
ağırlıklar kabul edilir. COCO veya başka sınıflı bir checkpoint yüklenirse servis
"hazır değil" olur ve maskeleri deprem hasarı diye SUNMAZ.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional

import cv2
import numpy as np

from src.application.dtos import ImageData, ModelInfo, SegmentationInstance
from src.application.interfaces import ISegmentationService
from src.domain.exceptions import DimensionMismatchError, ModelWeightsUnavailableError
from src.infrastructure.cv2_service import resize_mask_nearest

logger = logging.getLogger(__name__)

EXPECTED_CLASS_NAMES = ("damaged_region",)


def _default_model_loader(weights_path: str) -> Any:
    from ultralytics import YOLO  # ağır import: yalnızca gerektiğinde

    return YOLO(weights_path)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _to_numpy(value: Any) -> np.ndarray:
    """torch.Tensor veya NumPy dizisini NumPy'a çevirir."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def bbox_from_mask(mask: np.ndarray) -> Optional[tuple]:
    """Maskenin sıkı bounding box'ı (x_min, y_min, x_max, y_max); boşsa None."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


class YoloSegmentationService(ISegmentationService):
    def __init__(
        self,
        weights_path: Path,
        device: Optional[str] = None,
        imgsz: int = 512,
        retina_masks: bool = True,
        model_loader: Callable[[str], Any] = _default_model_loader,
    ) -> None:
        self._weights_path = Path(weights_path)
        self._device = device
        self._imgsz = imgsz
        self._retina_masks = retina_masks
        self._model_loader = model_loader
        self._model: Any = None
        self._load_error: Optional[str] = None
        self._sha256: Optional[str] = None

    # ------------------------------------------------------------------ #
    def is_ready(self) -> bool:
        try:
            self._ensure_model()
            return True
        except ModelWeightsUnavailableError as exc:
            logger.warning("Model hazır değil: %s", exc)
            return False

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            weights_path=str(self._weights_path),
            weights_sha256=self._sha256,
            model_family="yolov8-seg",
            class_names=EXPECTED_CLASS_NAMES,
        )

    @property
    def unavailable_reason(self) -> Optional[str]:
        return self._load_error

    # ------------------------------------------------------------------ #
    def segment(self, image: ImageData, confidence_threshold: float) -> List[SegmentationInstance]:
        model = self._ensure_model()
        # Ultralytics NumPy girdisini BGR kabul eder; ImageData RGB'dir.
        bgr = cv2.cvtColor(np.asarray(image.pixels, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        kwargs = dict(
            conf=confidence_threshold,
            imgsz=self._imgsz,
            retina_masks=self._retina_masks,
            verbose=False,
        )
        if self._device is not None:
            kwargs["device"] = self._device
        results = model.predict(bgr, **kwargs)
        if not results:
            return []
        return self._parse_result(results[0], image)

    # ------------------------------------------------------------------ #
    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self._weights_path.is_file():
            self._load_error = (
                f"Ağırlık dosyası bulunamadı: {self._weights_path}. "
                "Modeli eğitip best_model.pt dosyasını bu konuma koyun."
            )
            raise ModelWeightsUnavailableError(self._load_error)
        try:
            model = self._model_loader(str(self._weights_path))
        except Exception as exc:  # Ultralytics çeşitli hatalar fırlatabilir
            self._load_error = f"Model yüklenemedi: {exc}"
            raise ModelWeightsUnavailableError(self._load_error) from exc

        names = getattr(model, "names", None)
        class_names = tuple(names[k] for k in sorted(names)) if isinstance(names, dict) else tuple(names or ())
        if class_names != EXPECTED_CLASS_NAMES:
            self._load_error = (
                f"Ağırlık dosyası beklenen sınıfları içermiyor (bulunan: {list(class_names)[:5]}"
                f"{'...' if len(class_names) > 5 else ''}). Yalnızca 'damaged_region' ile eğitilmiş "
                "model kullanılabilir; COCO gibi genel modeller deprem hasarı olarak sunulmaz."
            )
            raise ModelWeightsUnavailableError(self._load_error)

        self._sha256 = _sha256(self._weights_path)
        self._model = model
        self._load_error = None
        logger.info("Model yüklendi: %s (sha256=%s…)", self._weights_path, self._sha256[:12])
        return model

    @staticmethod
    def _parse_result(result: Any, image: ImageData) -> List[SegmentationInstance]:
        masks_obj = getattr(result, "masks", None)
        boxes_obj = getattr(result, "boxes", None)
        if masks_obj is None or boxes_obj is None:
            return []

        masks = _to_numpy(masks_obj.data)
        confidences = _to_numpy(boxes_obj.conf).reshape(-1)
        classes = _to_numpy(boxes_obj.cls).reshape(-1).astype(int)
        xyxy = _to_numpy(boxes_obj.xyxy).reshape(-1, 4)
        if masks.ndim != 3 or not (len(masks) == len(confidences) == len(classes) == len(xyxy)):
            raise DimensionMismatchError("Model çıktısındaki maske ve kutu sayıları eşleşmiyor.")

        instances: List[SegmentationInstance] = []
        for raw_mask, conf, cls, box in zip(masks, confidences, classes, xyxy):
            mask = resize_mask_nearest(raw_mask > 0.5, image.height, image.width)
            bbox = bbox_from_mask(mask)
            if bbox is None:  # boş maske: kutuyu görüntüye kırparak kullan
                x1, y1, x2, y2 = (int(round(v)) for v in box)
                bbox = (
                    max(0, min(x1, image.width - 1)),
                    max(0, min(y1, image.height - 1)),
                    max(0, min(x2, image.width - 1)),
                    max(0, min(y2, image.height - 1)),
                )
                bbox = (bbox[0], bbox[1], max(bbox[0], bbox[2]), max(bbox[1], bbox[3]))
            instances.append(
                SegmentationInstance(
                    class_id=int(cls),
                    confidence=float(np.clip(conf, 0.0, 1.0)),
                    bounding_box=bbox,
                    mask=mask,
                )
            )
        return instances
