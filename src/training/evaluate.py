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

Eşik seçimi (test kümesine BAKMADAN):
    --select-threshold  doğrulama (val) kümesinde bir eşik ızgarası üzerinde mikro Dice'ı
                        en yüksek yapan confidence eşiğini seçer ve
                        reports/threshold_selection_<deney>.json olarak yazar.
    Test değerlendirmesi bu dosya varsa seçilen eşiği otomatik ekler ve panelleri o eşikle çizer.
    --export-app-weights ağırlığı weights/best_model.pt olarak kopyalar ve seçilen eşiği
                        weights/best_model.meta.json içine yazar; uygulama varsayılan eşiği oradan okur.

Çalıştırma:
    python -m src.training.evaluate --weights weights/<deney>_best.pt --variant malatya
    python -m src.training.evaluate --weights weights/<deney>_best.pt --variant kate7_eval_malatya \
        --select-threshold --export-app-weights
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
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
THRESHOLD_GRID = tuple(round(0.05 * i, 2) for i in range(1, 19))  # 0.05 ... 0.90
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


def select_best_threshold(per_threshold: Dict[float, Dict[str, PixelCounts]]):
    """Mikro Dice'ı en yüksek eşik; eşitlikte daha YÜKSEK eşik (daha az yanlış pozitif).

    Hiçbir eşikte pozitif Dice yoksa (None, None) döner.
    """
    best_t, best_dice = None, None
    for t in sorted(per_threshold):
        total = PixelCounts(0, 0, 0)
        for c in per_threshold[t].values():
            total = total + c
        dice = dice_from(total)
        if dice is None or dice <= 0:
            continue
        if best_dice is None or dice >= best_dice - 1e-12:
            best_t, best_dice = t, dice
    return best_t, best_dice


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
def _load_checked_model(weights: Path, yolo_factory):
    if not Path(weights).is_file():
        raise PreparationError(f"Ağırlık dosyası yok: {weights}")
    if yolo_factory is None:
        from ultralytics import YOLO as yolo_factory  # noqa: N811
    model = yolo_factory(str(weights))
    names = getattr(model, "names", {})
    class_names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    if class_names != ["damaged_region"]:
        raise PreparationError(f"Model sınıfları beklenen değil: {class_names}")
    return model


def _split_images(dataset_dir: Path, split: str) -> List[Path]:
    image_dir = dataset_dir / "images" / split
    images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise PreparationError(f"Değerlendirme görüntüsü yok: {image_dir}")
    return images


def _collect_pixel_counts(model, dataset_dir: Path, split: str, images: Sequence[Path],
                          thresholds: Sequence[float], imgsz: int, device: Optional[str],
                          fig_dir: Optional[Path] = None, figure_threshold: Optional[float] = None):
    """Her görüntü için tek tahmin; her eşikte piksel sayıları."""
    lowest = min(thresholds)
    per_threshold: Dict[float, Dict[str, PixelCounts]] = {t: {} for t in thresholds}
    pred_px: Dict[float, Dict[str, int]] = {t: {} for t in thresholds}
    gt_px: Dict[str, int] = {}
    for image_path in images:
        gt = binarize(cv2.imread(str(dataset_dir / "masks" / split / image_path.name), cv2.IMREAD_UNCHANGED))
        post = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
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
            if fig_dir is not None and figure_threshold is not None and abs(t - figure_threshold) < 1e-9:
                pre = cv2.imread(str(dataset_dir / "pre" / split / image_path.name), cv2.IMREAD_COLOR)
                if pre is not None:
                    cv2.imwrite(str(fig_dir / image_path.name), make_panel(pre, post, gt, pred, t))
    return per_threshold, pred_px, gt_px


def selection_path_for(experiment: str, reports_dir: Path = Path("reports")) -> Path:
    return reports_dir / f"threshold_selection_{experiment}.json"


def run_select_threshold(
    weights: Path,
    variant: str = VARIANT_MALATYA,
    experiment: Optional[str] = None,
    grid: Sequence[float] = THRESHOLD_GRID,
    imgsz: int = 512,
    device: Optional[str] = None,
    audit_path: Path = Path("reports/data_audit.json"),
    processed_root: Path = Path("data/processed"),
    reports_dir: Path = Path("reports"),
    yolo_factory=None,
) -> dict:
    """Doğrulama kümesinde eşik seçer. Test kümesi bu adımda HİÇ kullanılmaz."""
    load_passed_audit(audit_path)
    model = _load_checked_model(weights, yolo_factory)
    dataset_dir = output_dir_for(variant, processed_root)
    images = _split_images(dataset_dir, "val")
    experiment = experiment or Path(weights).stem.replace("_best", "")
    per_threshold, _, _ = _collect_pixel_counts(model, dataset_dir, "val", images, grid, imgsz, device)
    best_t, best_dice = select_best_threshold(per_threshold)

    table = []
    for t in sorted(per_threshold):
        total = PixelCounts(0, 0, 0)
        for c in per_threshold[t].values():
            total = total + c
        table.append({"threshold": t, "micro_dice": _r(dice_from(total)), "micro_iou": _r(iou_from(total))})
    result = {
        "experiment": experiment,
        "variant": variant,
        "weights": str(weights),
        "selected_on_split": "val",
        "n_images": len(images),
        "criterion": "doğrulama kümesinde en yüksek mikro Dice (eşitlikte daha yüksek eşik)",
        "selected_threshold": best_t,
        "selected_val_micro_dice": _r(best_dice),
        "grid": table,
        "caveats": [
            f"Eşik yalnızca {len(images)} doğrulama görüntüsüyle seçildi; kararsız olabilir.",
            "Test kümesi eşik seçiminde kullanılmadı.",
        ],
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if best_t is None:
        result["caveats"].append("Hiçbir eşikte pozitif Dice yok; model doğrulama kümesinde hasar bulamadı.")
    reports_dir.mkdir(parents=True, exist_ok=True)
    selection_path_for(experiment, reports_dir).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Seçilen eşik (val): %s, Dice %s", best_t, _r(best_dice))
    return result


def export_app_weights(weights: Path, selection: dict, weights_dir: Path = Path("weights"),
                       evaluation: Optional[dict] = None) -> Path:
    """Ağırlığı uygulamanın kullandığı yere kopyalar ve önerilen eşiği yanına yazar."""
    if selection.get("selected_threshold") is None:
        raise PreparationError("Seçilmiş eşik yok; model uygulamaya aktarılmadı.")
    weights_dir.mkdir(parents=True, exist_ok=True)
    target = weights_dir / "best_model.pt"
    if Path(weights).resolve() != target.resolve():
        shutil.copyfile(weights, target)
    meta = {
        "experiment": selection["experiment"],
        "variant": selection["variant"],
        "source_weights": str(weights),
        "weights_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "recommended_confidence_threshold": selection["selected_threshold"],
        "threshold_selected_on": selection["selected_on_split"],
        "val_micro_dice": selection["selected_val_micro_dice"],
        "n_val_images": selection["n_images"],
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if evaluation is not None:
        t = f"{selection['selected_threshold']:.2f}"
        m = evaluation.get("pixel_metrics_by_conf_threshold", {}).get(t)
        if m:
            meta["test_at_recommended_threshold"] = {
                "n_images": evaluation["n_images"], "micro_iou": m["micro_iou"],
                "micro_iou_ci95": m["micro_iou_ci95"], "micro_dice_f1": m["micro_dice_f1"],
            }
    meta_path = target.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Uygulama ağırlığı: %s (önerilen eşik %s)", target, meta["recommended_confidence_threshold"])
    return target


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
    use_selected_threshold: bool = True,
) -> dict:
    audit_report = load_passed_audit(audit_path)
    model = _load_checked_model(weights, yolo_factory)
    dataset_dir = output_dir_for(variant, processed_root)
    images = _split_images(dataset_dir, split)
    experiment = experiment or Path(weights).stem.replace("_best", "")
    gsd_values = audit_report["malatya"]["gsd_m_per_pixel_from_source_metadata"]
    gsd = gsd_values[0] if len(gsd_values) == 1 else None

    selected = None
    sel_path = selection_path_for(experiment, reports_dir)
    if use_selected_threshold and sel_path.is_file():
        selected = json.loads(sel_path.read_text(encoding="utf-8")).get("selected_threshold")
    thresholds = sorted({round(float(t), 2) for t in thresholds} | ({round(selected, 2)} if selected else set()))
    if selected:
        figure_threshold = round(selected, 2)

    result = {
        "experiment": experiment,
        "variant": variant,
        "split": split,
        "n_images": len(images),
        "weights": str(weights),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_cities": ["malatya"],
        "gsd_m_per_pixel": gsd,
        "selected_threshold_from_val": selected,
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

    fig_dir = reports_dir / "figures" / (experiment if split == "test" else f"{experiment}_{split}")
    fig_dir.mkdir(parents=True, exist_ok=True)
    per_threshold, pred_px, gt_px = _collect_pixel_counts(
        model, dataset_dir, split, images, thresholds, imgsz, device, fig_dir, figure_threshold)

    result["pixel_metrics_by_conf_threshold"] = {
        f"{t:.2f}": summarize_threshold(per_threshold[t], gsd, pred_px[t], gt_px) for t in thresholds
    }
    result["figures_dir"] = str(fig_dir)

    reports_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    suffix = "" if split == "test" else f"_{split}"
    (reports_dir / f"evaluation_{experiment}{suffix}.json").write_text(text, encoding="utf-8")
    if split == "test":
        (reports_dir / "evaluation.json").write_text(text, encoding="utf-8")
    logger.info("Değerlendirme raporu: %s", reports_dir / f"evaluation_{experiment}{suffix}.json")
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
    if result.get("selected_threshold_from_val") is not None:
        print(f"Doğrulama kümesinde seçilen eşik: {result['selected_threshold_from_val']}")
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
    parser.add_argument("--select-threshold", action="store_true",
                        help="Önce doğrulama kümesinde eşik seç (test kümesine bakmadan)")
    parser.add_argument("--export-app-weights", action="store_true",
                        help="Ağırlığı weights/best_model.pt yap, seçilen eşiği best_model.meta.json'a yaz")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    try:
        selection = None
        if args.select_threshold:
            selection = run_select_threshold(args.weights, args.variant, args.experiment, device=args.device)
            print(f"Doğrulama kümesinde seçilen eşik: {selection['selected_threshold']} "
                  f"(Dice {selection['selected_val_micro_dice']}, n={selection['n_images']})")
        result = run_evaluate(args.weights, args.variant, args.split, args.experiment, args.thresholds,
                              device=args.device)
        print_summary(result)
        if args.export_app_weights:
            if selection is None:
                path = selection_path_for(result["experiment"])
                if not path.is_file():
                    raise PreparationError("Önce --select-threshold ile eşik seçilmeli.")
                selection = json.loads(path.read_text(encoding="utf-8"))
            target = export_app_weights(args.weights, selection, evaluation=result)
            print(f"Uygulama modeli: {target} (önerilen eşik {selection['selected_threshold']})")
    except PreparationError as exc:
        logger.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
