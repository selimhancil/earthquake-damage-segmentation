"""DEMO ONLY adapter'ları: eğitilmiş model olmadan arayüzü denemek için.

Bu modüldeki hiçbir şey deprem hasarı tespit ETMEZ.
- DemoSegmentationService: yüksek yerel doku (moloz benzeri gürültü) içeren
  alanları basit bir OpenCV sezgiseliyle işaretler. Gerçek modelin yerine
  geçmez; ModelInfo.is_demo=True döndürdüğü için use case tüm çıktılara
  DEMO_ONLY uyarısı ve DEMO_ dosya ön eki ekler.
- DemoWatermarkVisualization: kaydedilen görsellere "DEMO ONLY" filigranı basar.
- make_demo_scene: sentetik "öncesi / sonrası" sahne üretir (gerçek veri değildir).

Yalnızca EV_DEMO_MODE=1 ile composition root'ta etkinleşir.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from src.application.dtos import ImageData, ModelInfo, SegmentationInstance
from src.application.interfaces import ISegmentationService, IVisualizationService

DEMO_LABEL = "DEMO ONLY"


# --------------------------------------------------------------------------- #
# Sahte segmentasyon
# --------------------------------------------------------------------------- #
class DemoSegmentationService(ISegmentationService):
    """Yoğun ince doku (moloz benzeri gürültü) içeren bölgeleri işaretleyen sezgisel.

    Her pikselde Laplacian büyüklüğü güçlü mü diye bakılır; bir pencere içindeki
    güçlü piksel oranı (doku yoğunluğu) eşiği aşarsa alan işaretlenir. Bina
    kenarları gibi ince çizgiler düşük yoğunluk verdiği için işaretlenmez.
    Deterministiktir: aynı görüntü her zaman aynı sonucu verir.
    """

    def __init__(
        self,
        laplacian_threshold: float = 40.0,
        window: int = 15,
        density_threshold: float = 0.5,
        min_area_ratio: float = 0.002,
    ):
        self._lap_threshold = laplacian_threshold
        self._window = window
        self._density_threshold = density_threshold
        self._min_area_ratio = min_area_ratio

    def is_ready(self) -> bool:
        return True

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            weights_path="DEMO (model yok)",
            weights_sha256=None,
            model_family="demo-texture-heuristic",
            class_names=("damaged_region",),
            is_demo=True,
        )

    def segment(self, image: ImageData, confidence_threshold: float) -> List[SegmentationInstance]:
        gray = cv2.cvtColor(np.asarray(image.pixels, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3))
        strong = (lap > self._lap_threshold).astype(np.float32)
        density = cv2.blur(strong, (self._window, self._window))

        textured = (density > self._density_threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        textured = cv2.morphologyEx(textured, cv2.MORPH_CLOSE, kernel)
        textured = cv2.morphologyEx(textured, cv2.MORPH_OPEN, kernel)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(textured, connectivity=8)
        min_area = max(20, int(self._min_area_ratio * image.width * image.height))
        instances: List[SegmentationInstance] = []
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if area < min_area:
                continue
            mask = labels == label
            # Sahte "confidence": bölgedeki ortalama doku yoğunluğu (0.50–0.99)
            confidence = round(float(np.clip(density[mask].mean(), 0.5, 0.99)), 3)
            if confidence < confidence_threshold:
                continue
            instances.append(
                SegmentationInstance(
                    class_id=0,
                    confidence=confidence,
                    bounding_box=(int(x), int(y), int(x + w - 1), int(y + h - 1)),
                    mask=mask,
                )
            )
        instances.sort(key=lambda inst: (-inst.confidence, inst.bounding_box))
        return instances


# --------------------------------------------------------------------------- #
# Filigranlı görselleştirme
# --------------------------------------------------------------------------- #
def add_demo_watermark(rgb: np.ndarray) -> np.ndarray:
    out = np.asarray(rgb, dtype=np.uint8).copy()
    h, w = out.shape[:2]
    scale = max(0.5, min(w, h) / 400)
    thickness = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(DEMO_LABEL, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = max(4, th // 3)
    cv2.rectangle(out, (0, 0), (min(w - 1, tw + 2 * pad), min(h - 1, th + base + 2 * pad)),
                  (0, 0, 0), thickness=-1)
    cv2.putText(out, DEMO_LABEL, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return out


class DemoWatermarkVisualization(IVisualizationService):
    """Gerçek görselleştiriciyi sarar; işaretli görüntüye DEMO ONLY filigranı ekler.

    Maske görseline dokunmaz: maske piksel sayısı ölçümle aynı kalmalıdır.
    """

    def __init__(self, inner: IVisualizationService):
        self._inner = inner

    def render_overlay(self, image: ImageData, instances: Sequence[SegmentationInstance],
                       partial_flags: Sequence[bool], labels: Sequence[str]) -> np.ndarray:
        return add_demo_watermark(self._inner.render_overlay(image, instances, partial_flags, labels))

    def render_union_mask(self, image: ImageData, instances: Sequence[SegmentationInstance]) -> np.ndarray:
        return self._inner.render_union_mask(image, instances)


# --------------------------------------------------------------------------- #
# Sentetik örnek sahne
# --------------------------------------------------------------------------- #
def make_demo_scene(width: int = 640, height: int = 480, seed: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Sentetik (öncesi, sonrası) RGB sahne çifti. GERÇEK VERİ DEĞİLDİR.

    Öncesi: düz zemin üzerinde sağlam 'binalar' (tek renkli dikdörtgenler) ve yollar.
    Sonrası: aynı sahne; bazı binaların yerinde yüksek dokulu 'moloz' alanları.
    Binalardan biri görüntü kenarına taşar (kenar/partial davranışını göstermek için).
    """
    rng = np.random.default_rng(seed)
    pre = np.zeros((height, width, 3), dtype=np.uint8)
    pre[:] = (150, 145, 130)  # zemin
    pre += rng.integers(0, 6, size=pre.shape, dtype=np.uint8)  # hafif doku

    road = (95, 95, 100)
    cv2.rectangle(pre, (0, height // 2 - 12), (width - 1, height // 2 + 12), road, -1)
    cv2.rectangle(pre, (width // 3 - 10, 0), (width // 3 + 10, height - 1), road, -1)

    buildings = [
        (40, 40, 150, 140), (200, 50, 300, 150), (380, 30, 500, 160),
        (60, 290, 170, 420), (240, 300, 360, 430), (430, 280, 590, 400),
        (560, 60, width - 1, 170),  # sağ kenara taşan bina
    ]
    roof_colors = [(180, 90, 70), (200, 200, 205), (170, 160, 150), (190, 110, 80),
                   (210, 210, 200), (160, 150, 140), (185, 95, 75)]
    for (x0, y0, x1, y1), color in zip(buildings, roof_colors):
        cv2.rectangle(pre, (x0, y0), (x1, y1), color, -1)
        cv2.rectangle(pre, (x0, y0), (x1, y1), (60, 60, 60), 2)

    post = pre.copy()
    collapsed = [1, 3, 4, 6]
    for idx in collapsed:
        x0, y0, x1, y1 = buildings[idx]
        x0, y0 = max(0, x0 - 12), max(0, y0 - 12)  # moloz bina sınırını biraz aşar
        x1, y1 = min(width - 1, x1 + 12), min(height - 1, y1 + 12)
        patch = rng.integers(40, 230, size=(y1 - y0 + 1, x1 - x0 + 1, 3), dtype=np.uint8)
        post[y0:y1 + 1, x0:x1 + 1] = patch
    return pre, post


def encode_png(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:  # pragma: no cover
        raise RuntimeError("PNG kodlanamadı")
    return buf.tobytes()


def demo_scene_png_pair() -> Tuple[bytes, bytes]:
    """(öncesi_png, sonrası_png) baytları; arayüzdeki 'örnek sahne' düğmesi için."""
    pre, post = make_demo_scene()
    return encode_png(pre), encode_png(post)
