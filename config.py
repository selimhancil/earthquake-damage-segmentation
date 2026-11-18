"""Proje genelindeki ayarların tek kaynağı.

Ortam değişkenleriyle geçersiz kılınabilen değerler:
    EV_WEIGHTS_PATH   model ağırlık dosyası (varsayılan: weights/best_model.pt)
    EV_RESULTS_DIR    çıktı klasörü (varsayılan: data/test_results)
    EV_DEVICE         YOLO cihazı: "cpu", "mps", "0" ... (varsayılan: Ultralytics seçer)
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Yollar
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "sample_images"
RESULTS_DIR = Path(os.environ.get("EV_RESULTS_DIR", DATA_DIR / "test_results"))
WEIGHTS_PATH = Path(os.environ.get("EV_WEIGHTS_PATH", BASE_DIR / "weights" / "best_model.pt"))
REPORTS_DIR = BASE_DIR / "reports"

# --------------------------------------------------------------------------- #
# Analiz ayarları
# --------------------------------------------------------------------------- #
# Kural 1: 0.70 keyfi bir başlangıçtır, doğrulanmış optimum değildir.
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
# Yalnızca bilimsel değerlendirme (mAP) için; arayüzde KULLANILMAZ.
EVAL_CONFIDENCE_THRESHOLD = 0.001

MAX_IMAGE_MB = 15
MAX_IMAGE_SIDE_PX = 4096

# Bilinmeyen görüntüde fiziksel alan uydurulmaz.
DEFAULT_GSD_M_PER_PIXEL = None

# Veri dönüştürmede poligon sadakati uyarı eşiği (IoU).
POLYGON_FIDELITY_IOU_WARN = 0.90

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
YOLO_IMGSZ = 512
YOLO_DEVICE = os.environ.get("EV_DEVICE") or None

# --------------------------------------------------------------------------- #
# Dosya türleri
# --------------------------------------------------------------------------- #
GEOTIFF_ENABLED = importlib.util.find_spec("rasterio") is not None
RASTER_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
GEOTIFF_EXTENSIONS = frozenset({".tif", ".tiff"})
ALLOWED_EXTENSIONS = RASTER_EXTENSIONS | (GEOTIFF_EXTENSIONS if GEOTIFF_ENABLED else frozenset())

# --------------------------------------------------------------------------- #
# Loglama
# --------------------------------------------------------------------------- #
LOG_LEVEL = os.environ.get("EV_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - [%(levelname)s] - %(name)s: %(message)s"
LOG_DATEFMT = "%H:%M:%S"
