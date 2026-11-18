"""Eğitilmiş modeli değerlendir (şartname 2.4 ve 9). YALNIZCA ölçülen değerler raporlanır.

İki tür metrik:
1. Ultralytics segmentasyon metrikleri: mask mAP50, mAP50-95, precision, recall.
   Bunlar PR eğrisi üzerinden hesaplanır, bu yüzden düşük eşikle (conf=0.001)
   çalıştırılır; arayüzdeki 0.70 eşiği BURADA KULLANILMAZ.
2. Piksel düzeyi IoU ve Dice (F1): her confidence eşiği için tahmin maskelerinin
   birleşimi, ground-truth maskeyle karşılaştırılır. Hangi eşikte ölçüldüğü
   her değerin yanında yazılır.

Değerlendirme kümesi küçük olduğu için (Malatya val+test = 9) piksel metrikleri
görüntüler üzerinden bootstrap ile %95 güven aralığıyla raporlanır.

Çıktılar:
    reports/evaluation_<deney>.json  (+ reports/evaluation.json)
    reports/figures/<deney>/<görüntü>.png  (öncesi | sonrası | ground truth | tahmin)

Çalıştırma:
    python -m src.training.evaluate --weights weights/<deney>_best.pt --variant malatya
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from src.training.convert_to_yolo import binarize, output_dir_for
from src.training.prepare_malatya import VARIANT_MALATYA, VARIANTS, PreparationError, load_passed_audit
from src.training.train import IMAGE_SUFFIXES, write_runtime_yaml

logger = logging.getLogger("evaluate")

EVAL_CONF_FOR_MAP = 0.001  # config.EVAL_CONFIDENCE_THRESHOLD
DEFAULT_PIXEL_THRESHOLDS = (0.25, 0.50, 0.70)
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42


# --------------------------------------------------------------------------- #
# Saf metrik fonksiyonları
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PixelCounts:
    tp: int
    fp: int
    fn: int

    def __add__(self, other: "PixelCounts") -> "PixelCounts":
        return PixelCounts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def pixel_counts(pred: np.ndarray, gt: np.ndarray) -> PixelCounts:
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Boyut farklı: {pred.shape} vs {gt.shape}")
    return PixelCounts(int(np.count_nonzero(pred & gt)), int(np.count_nonzero(pred & ~gt)),
                       int(np.count_nonzero(~pred & gt)))


def iou_from(c: PixelCounts) -> Optional[float]:
    denom = c.tp + c.fp + c.fn
    return None if denom == 0 else c.tp / denom


def dice_from(c: PixelCounts) -> Optional[float]:
    denom = 2 * c.tp + c.fp + c.fn
    return None if denom == 0 else 2 * c.tp / denom


def precision_recall_from(c: PixelCounts):
    p = None if (c.tp + c.fp) == 0 else c.tp / (c.tp + c.fp)
    r = None if (c.tp + c.fn) == 0 else c.tp / (c.tp + c.fn)
    return p, r


def union_mask(masks: Sequence[np.ndarray], confidences: Sequence[float], threshold: float,
               shape) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for m, c in zip(masks, confidences):
        if c >= threshold:
            out |= m.astype(bool)
    return out


def bootstrap_ci(per_image: Sequence[PixelCounts], metric, samples: int = BOOTSTRAP_SAMPLES,
                 seed: int = BOOTSTRAP_SEED, alpha: float = 0.05):
    """Görüntüleri yeniden örnekleyerek mikro-ortalama metrik için (alt, üst) aralık."""
    n = len(per_image)
    if n == 0:
        return None, None
    rng = np.random.default_rng(seed)
    arr = np.array([[c.tp, c.fp, c.fn] for c in per_image], dtype=np.int64)
    values = []
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        tp, fp, fn = arr[idx].sum(axis=0)
        v = metric(PixelCounts(int(tp), int(fp), int(fn)))
        if v is not None:
            values.append(v)
    if not values:
        return None, None
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return round(float(lo), 4), round(float(hi), 4)


def summarize_threshold(per_image: Dict[str, PixelCounts], gsd_m: Optional[float],
                        pred_pixels: Dict[str, int], gt_pixels: Dict[str, int]) -> dict:
    total = PixelCounts(0, 0, 0)
    for c in per_image.values():
        total = total + c
    p, r = precision_recall_from(total)
    items = list(per_image.values())
    per_iou = [v for v in (iou_from(c) for c in items) if v is not None]
    pixel_area = gsd_m ** 2 if gsd_m else None
    pred_total, gt_total = sum(pred_pixels.values()), sum(gt_pixels.values())
    return {
        "micro_iou": _r(iou_from(total)),
        "micro_iou_ci95": bootstrap_ci(items, iou_from),
        "micro_dice_f1": _r(dice_from(total)),
        "micro_dice_ci95": bootstrap_ci(items, dice_from),
        "pixel_precision": _r(p),
        "pixel_recall": _r(r),
        "mean_image_iou": _r(float(np.mean(per_iou)) if per_iou else None),
        "area": {
            "predicted_damaged_pixels": pred_total,
            "ground_truth_damaged_pixels": gt_total,
            "predicted_damaged_m2": _r(pred_total * pixel_area, 2) if pixel_area else None,
            "ground_truth_damaged_m2": _r(gt_total * pixel_area, 2) if pixel_area else None,
            "relative_area_error": _r((pred_total - gt_total) / gt_total) if gt_total else None,
        },
        "per_image": {
            name: {"iou": _r(iou_from(c)), "dice": _r(dice_from(c)), "tp": c.tp, "fp": c.fp, "fn": c.fn,
                   "pred_pixels": pred_pixels[name], "gt_pixels": gt_pixels[name]}
            for name, c in per_image.items()
        },
    }


def _r(v, digits: int = 4):
    return None if v is None else round(float(v), digits)


# --------------------------------------------------------------------------- #
# Görsel panel
# --------------------------------------------------------------------------- #
def make_panel(pre_bgr: np.ndarray, post_bgr: np.ndarray, gt: np.ndarray, pred: np.ndarray,
               threshold: float) -> np.ndarray:
    def overlay(img, mask, color):
        out = img.copy()
        out[mask] = (0.5 * out[mask] + 0.5 * np.array(color)).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
        return out

    tiles = [
        (pre_bgr, "pre (girdi DEGIL)"),
        (post_bgr, "post (model girdisi)"),
        (overlay(post_bgr, gt, (0, 200, 0)), "ground truth"),
        (overlay(post_bgr, pred, (0, 0, 255)), f"tahmin conf>={threshold:.2f}"),
    ]
    rendered = []
    for img, label in tiles:
        tile = img.copy()
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(tile, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        rendered.append(tile)
    return cv2.hconcat(rendered)


# --------------------------------------------------------------------------- #
# Ana akış
# --------------------------------------------------------------------------- #
def run_evaluate(
    weights: Path,
    variant: str = VARIANT_MALATYA,
    split: str = "test",
    experiment: Optional[str] = None,
    thresholds: Sequence[float] = DEFAULT_PIXEL_THRESHOLDS,
    figure_threshold: float = 0.50,
    imgsz: int = 512,
    device: Optional[str] = None,
    audit_path: Path = Path("reports/data_audit.json"),
    processed_root: Path = Path("data/processed"),
    reports_dir: Path = Path("reports"),
    yolo_factory=None,
    run_ultralytics_val: bool = True,
) -> dict:
    audit_report = load_passed_audit(audit_path)
    if not Path(weights).is_file():
        raise PreparationError(f"Ağırlık dosyası yok: {weights}")
    dataset_dir = output_dir_for(variant, processed_root)
    image_dir = dataset_dir / "images" / split
    images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise PreparationError(f"Değerlendirme görüntüsü yok: {image_dir}")
    experiment = experiment or Path(weights).stem.replace("_best", "")
    gsd_values = audit_report["malatya"]["gsd_m_per_pixel_from_source_metadata"]
    gsd = gsd_values[0] if len(gsd_values) == 1 else None

    if yolo_factory is None:
        from ultralytics import YOLO as yolo_factory  # noqa: N811
    model = yolo_factory(str(weights))
    names = getattr(model, "names", {})
    class_names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    if class_names != ["damaged_region"]:
        raise PreparationError(f"Model sınıfları beklenen değil: {class_names}")

    result = {
        "experiment": experiment,
        "variant": variant,
        "split": split,
        "n_images": len(images),
        "weights": str(weights),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_cities": ["malatya"],
        "gsd_m_per_pixel": gsd,
        "caveats": [
            f"Değerlendirme kümesi küçük (n={len(images)}); metrikler belirsizlik aralığıyla yorumlanmalı.",
            "Piksel IoU/Dice, hasar maskesi ile tahmin maskesinin birleşimi üzerinden hesaplanır; "
            "bina sayısı veya hasarsız bina alanı ölçülmez.",
            "Model yalnızca deprem sonrası görüntüyü kullanır; deprem öncesi görüntü yalnızca panellerde gösterilir.",
        ],
    }

    if run_ultralytics_val:
        runtime_yaml = write_runtime_yaml(dataset_dir, Path("runs/segment") / f"eval_{experiment}" / "data.yaml")
        kwargs = dict(data=str(runtime_yaml), split=split, conf=EVAL_CONF_FOR_MAP, imgsz=imgsz,
                      plots=False, verbose=False)
        if device:
            kwargs["device"] = device
        metrics = model.val(**kwargs)
        seg = metrics.seg
        result["ultralytics_seg"] = {
            "conf_threshold": EVAL_CONF_FOR_MAP,
            "mask_map50": _r(seg.map50),
            "mask_map50_95": _r(seg.map),
            "mask_precision": _r(seg.mp),
            "mask_recall": _r(seg.mr),
        }
        box = getattr(metrics, "box", None)
        if box is not None:
            result["ultralytics_box"] = {"map50": _r(box.map50), "map50_95": _r(box.map)}

    lowest = min(thresholds)
    per_threshold: Dict[float, Dict[str, PixelCounts]] = {t: {} for t in thresholds}
    pred_px: Dict[float, Dict[str, int]] = {t: {} for t in thresholds}
    gt_px: Dict[str, int] = {}
    fig_dir = reports_dir / "figures" / experiment
    fig_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        gt = binarize(cv2.imread(str(dataset_dir / "masks" / split / image_path.name), cv2.IMREAD_UNCHANGED))
        post = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        pre = cv2.imread(str(dataset_dir / "pre" / split / image_path.name), cv2.IMREAD_COLOR)
        kwargs = dict(conf=lowest, imgsz=imgsz, retina_masks=True, verbose=False)
        if device:
            kwargs["device"] = device
        res = model.predict(post, **kwargs)[0]
        masks, confs = _extract(res, gt.shape)
        gt_px[image_path.name] = int(gt.sum())
        for t in thresholds:
            pred = union_mask(masks, confs, t, gt.shape)
            per_threshold[t][image_path.name] = pixel_counts(pred, gt)
            pred_px[t][image_path.name] = int(pred.sum())
            if abs(t - figure_threshold) < 1e-9 and pre is not None:
                cv2.imwrite(str(fig_dir / image_path.name), make_panel(pre, post, gt, pred, t))

    result["pixel_metrics_by_conf_threshold"] = {
        f"{t:.2f}": summarize_threshold(per_threshold[t], gsd, pred_px[t], gt_px) for t in thresholds
    }
    result["figures_dir"] = str(fig_dir)

    reports_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    (reports_dir / f"evaluation_{experiment}.json").write_text(text, encoding="utf-8")
    (reports_dir / "evaluation.json").write_text(text, encoding="utf-8")
    logger.info("Değerlendirme raporu: %s", reports_dir / f"evaluation_{experiment}.json")
    return result


def _extract(res, shape):
    """Ultralytics Results -> (bool maskeler, confidence'lar), görüntü boyutunda."""
    masks_obj, boxes = getattr(res, "masks", None), getattr(res, "boxes", None)
    if masks_obj is None or boxes is None:
        return [], []

    def to_np(v):
        for attr in ("detach", "cpu"):
            if hasattr(v, attr):
                v = getattr(v, attr)()
        return np.asarray(v.numpy() if hasattr(v, "numpy") else v)

    data = to_np(masks_obj.data)
    confs = to_np(boxes.conf).reshape(-1).tolist()
    out = []
    for m in data:
        m = m > 0.5
        if m.shape != shape:
            m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        out.append(m)
    return out, confs


def print_summary(result: dict) -> None:
    print("=" * 70)
    print(f"Deney: {result['experiment']}  |  split: {result['split']}  |  n = {result['n_images']}")
    if "ultralytics_seg" in result:
        s = result["ultralytics_seg"]
        print(f"Mask mAP50: {s['mask_map50']}  mAP50-95: {s['mask_map50_95']}  "
              f"P: {s['mask_precision']}  R: {s['mask_recall']}  (conf={s['conf_threshold']})")
    for t, m in result["pixel_metrics_by_conf_threshold"].items():
        print(f"conf>={t}: IoU {m['micro_iou']} {m['micro_iou_ci95']}  Dice {m['micro_dice_f1']} "
              f"{m['micro_dice_ci95']}  alan hatası {m['area']['relative_area_error']}")
    for c in result["caveats"]:
        print(f"NOT: {c}")
    print("=" * 70)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Modeli Malatya değerlendirme kümesinde ölç")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, default=VARIANT_MALATYA)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--name", dest="experiment")
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_PIXEL_THRESHOLDS))
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    try:
        result = run_evaluate(args.weights, args.variant, args.split, args.experiment, args.thresholds,
                              device=args.device)
    except PreparationError as exc:
        logger.error("%s", exc)
        return 2
    print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
