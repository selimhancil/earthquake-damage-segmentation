"""İkili hasar maskelerini YOLOv8-seg biçimine dönüştür (şartname 2.3).

Girdi : prepare_malatya.py'nin ürettiği selection_<varyant>.json + indirilen PNG'ler
Çıktı : data/processed/<varyant>_yolo/
          images/{train,val,test}/   deprem SONRASI görüntü (modelin tek girdisi)
          labels/{train,val,test}/   YOLO-seg .txt  (class x1 y1 x2 y2 ... normalize)
          masks/{train,val,test}/    orijinal ikili maske (piksel IoU/Dice için)
          pre/{train,val,test}/      deprem ÖNCESİ görüntü (yalnızca görselleştirme)
          dataset.yaml               (spec biçimi, göreli yol)
        reports/conversion_fidelity_<varyant>.json

Kurallar:
- Tek sınıf: 0 = damaged_region. Maskede >0 olan her piksel hasardır (KATE-CD 0/255).
- Poligonlar cv2.findContours(RETR_EXTERNAL) ile çıkarılır; YOLO poligonu delik
  desteklemez. Her örnek için orijinal maske ile poligonlardan yeniden çizilen
  maske arasındaki IoU ölçülür; eşik altı örnekler uyarılır ama ÇIKARILMAZ.
- min_area_px altındaki küçük bileşenler atılır; sayı ve piksel toplamı raporlanır.
- Hasar etiketi olmayan görüntüler boş .txt ile tutulur (bunlar hasarsız bina
  etiketi DEĞİLDİR).

Çalıştırma:
    python -m src.training.convert_to_yolo
    python -m src.training.convert_to_yolo --variant kate7_eval_malatya
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.training.prepare_malatya import VARIANT_MALATYA, VARIANTS, PreparationError

logger = logging.getLogger("convert_to_yolo")

CLASS_ID = 0
CLASS_NAME = "damaged_region"
DEFAULT_MIN_AREA_PX = 16  # 0.5 m GSD'de 4 m²
DEFAULT_FIDELITY_WARN = 0.90  # config.POLYGON_FIDELITY_IOU_WARN ile aynı
SPLITS = ("train", "val", "test")

Polygon = np.ndarray  # (N, 2) piksel koordinatları, int


@dataclass
class SampleReport:
    split: str
    name: str
    city: str
    mask_pixels: int
    polygons: int
    dropped_components: int
    dropped_pixels: int
    fidelity_iou: Optional[float]  # boş maskede None
    warning: Optional[str] = None


@dataclass
class ConversionSummary:
    variant: str
    output_dir: str
    samples: List[SampleReport] = field(default_factory=list)
    min_area_px: int = DEFAULT_MIN_AREA_PX
    fidelity_warn_threshold: float = DEFAULT_FIDELITY_WARN


# --------------------------------------------------------------------------- #
# Saf dönüştürme fonksiyonları
# --------------------------------------------------------------------------- #
def binarize(mask: np.ndarray) -> np.ndarray:
    """0/1 veya 0/255 maskeyi bool'a çevirir; renkli kaydedilmiş gri maskeyi de kabul eder."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def mask_to_polygons(mask: np.ndarray, min_area_px: int = DEFAULT_MIN_AREA_PX,
                     epsilon_px: float = 0.0) -> Tuple[List[Polygon], int, int]:
    """Dış konturlar -> poligon listesi, (atılan bileşen sayısı, atılan piksel)."""
    binary = binarize(mask).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[Polygon] = []
    dropped, dropped_px = 0, 0
    for contour in contours:
        component = np.zeros_like(binary)
        cv2.drawContours(component, [contour], -1, 1, thickness=-1)
        area = int(np.count_nonzero(component & binary))
        if area < min_area_px:
            dropped += 1
            dropped_px += area
            continue
        if epsilon_px > 0:
            contour = cv2.approxPolyDP(contour, epsilon_px, True)
        pts = contour.reshape(-1, 2)
        if len(pts) < 3:  # çizgi/nokta biçimli bileşen: küçük bir kutuya genişlet
            x, y, w, h = cv2.boundingRect(contour)
            pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        polygons.append(pts.astype(np.int32))
    return polygons, dropped, dropped_px


def rasterize_polygons(polygons: Sequence[Polygon], height: int, width: int) -> np.ndarray:
    canvas = np.zeros((height, width), np.uint8)
    if polygons:
        cv2.fillPoly(canvas, [p.reshape(-1, 1, 2) for p in polygons], 1)
    return canvas.astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    union = np.count_nonzero(a | b)
    if union == 0:
        return None
    return float(np.count_nonzero(a & b) / union)


def polygons_to_yolo_lines(polygons: Sequence[Polygon], height: int, width: int,
                           class_id: int = CLASS_ID) -> List[str]:
    """'0 x1 y1 x2 y2 ...' satırları; koordinatlar [0, 1] aralığına kırpılır."""
    lines = []
    for poly in polygons:
        xs = np.clip(poly[:, 0] / width, 0.0, 1.0)
        ys = np.clip(poly[:, 1] / height, 0.0, 1.0)
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))
        lines.append(f"{class_id} {coords}")
    return lines


def yolo_lines_to_mask(lines: Sequence[str], height: int, width: int) -> np.ndarray:
    """Doğrulama için: YOLO satırlarını tekrar maskeye çizer."""
    polys = []
    for line in lines:
        values = [float(v) for v in line.split()[1:]]
        pts = np.array(values, dtype=np.float64).reshape(-1, 2)
        pts[:, 0] *= width
        pts[:, 1] *= height
        polys.append(np.round(pts).astype(np.int32))
    return rasterize_polygons(polys, height, width)


def convert_mask(mask: np.ndarray, min_area_px: int = DEFAULT_MIN_AREA_PX,
                 epsilon_px: float = 0.0):
    """Tek maske -> (YOLO satırları, poligon sayısı, atılan bileşen, atılan piksel, sadakat IoU)."""
    binary = binarize(mask)
    h, w = binary.shape
    polygons, dropped, dropped_px = mask_to_polygons(binary, min_area_px, epsilon_px)
    lines = polygons_to_yolo_lines(polygons, h, w)
    fidelity = iou(binary, yolo_lines_to_mask(lines, h, w)) if binary.any() else None
    return lines, len(polygons), dropped, dropped_px, fidelity


def dataset_yaml_text(relative_path: str) -> str:
    return (
        f"path: {relative_path}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"  {CLASS_ID}: {CLASS_NAME}\n"
    )


def output_dir_for(variant: str, processed_root: Path = Path("data/processed")) -> Path:
    return processed_root / f"{variant}_yolo"


# --------------------------------------------------------------------------- #
# Ana akış
# --------------------------------------------------------------------------- #
def run_convert(
    variant: str = VARIANT_MALATYA,
    download_dir: Path = Path("data/raw/kate_cd"),
    processed_root: Path = Path("data/processed"),
    reports_dir: Path = Path("reports"),
    min_area_px: int = DEFAULT_MIN_AREA_PX,
    epsilon_px: float = 0.0,
    fidelity_warn: float = DEFAULT_FIDELITY_WARN,
) -> ConversionSummary:
    selection_path = download_dir / f"selection_{variant}.json"
    if not selection_path.is_file():
        raise PreparationError(
            f"{selection_path} yok. Önce: python -m src.training.prepare_malatya --variant {variant}"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    pairs_dir = Path(selection["pairs_dir"])
    out_dir = output_dir_for(variant, processed_root)

    for sub in ("images", "labels", "masks", "pre"):
        for split in SPLITS:
            target = out_dir / sub / split
            if target.exists():
                for f in target.iterdir():
                    if f.name != ".gitkeep":
                        f.unlink()
            target.mkdir(parents=True, exist_ok=True)

    summary = ConversionSummary(variant=variant, output_dir=str(out_dir), min_area_px=min_area_px,
                                fidelity_warn_threshold=fidelity_warn)
    for pair in selection["selection"]:
        split, src_split, name, city = pair["split"], pair["source_split"], pair["name"], pair["city"]
        stem = Path(name).stem
        src = {f: pairs_dir / src_split / f / name for f in ("A", "B", "label")}
        missing = [str(p) for p in src.values() if not p.is_file()]
        if missing:
            raise PreparationError(f"Eksik dosya(lar): {missing}. prepare_malatya.py tekrar çalıştırılmalı.")

        mask = cv2.imread(str(src["label"]), cv2.IMREAD_UNCHANGED)
        post = cv2.imread(str(src["B"]), cv2.IMREAD_COLOR)
        if mask is None or post is None:
            raise PreparationError(f"Okunamayan görüntü/maske: {name}")
        if binarize(mask).shape != post.shape[:2]:
            raise PreparationError(f"Maske ve görüntü boyutu farklı: {name}")

        lines, n_poly, dropped, dropped_px, fidelity = convert_mask(mask, min_area_px, epsilon_px)
        out_name = f"{city}_{src_split}_{stem}" if variant != VARIANT_MALATYA else f"{src_split}_{stem}"
        shutil.copyfile(src["B"], out_dir / "images" / split / f"{out_name}.png")
        shutil.copyfile(src["A"], out_dir / "pre" / split / f"{out_name}.png")
        cv2.imwrite(str(out_dir / "masks" / split / f"{out_name}.png"), binarize(mask).astype(np.uint8) * 255)
        (out_dir / "labels" / split / f"{out_name}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

        warning = None
        if fidelity is not None and fidelity < fidelity_warn:
            warning = f"Poligon sadakati düşük (IoU {fidelity:.3f} < {fidelity_warn})"
            logger.warning("%s/%s: %s", split, out_name, warning)
        summary.samples.append(SampleReport(split, out_name, city, int(binarize(mask).sum()), n_poly,
                                            dropped, dropped_px, fidelity, warning))

    (out_dir / "dataset.yaml").write_text(dataset_yaml_text(out_dir.as_posix()), encoding="utf-8")
    write_fidelity_report(summary, reports_dir / f"conversion_fidelity_{variant}.json", selection)
    logger.info("Dönüştürme tamamlandı: %s (%d örnek)", out_dir, len(summary.samples))
    return summary


def write_fidelity_report(summary: ConversionSummary, path: Path, selection: dict) -> dict:
    fids = [s.fidelity_iou for s in summary.samples if s.fidelity_iou is not None]
    report = {
        "variant": summary.variant,
        "output_dir": summary.output_dir,
        "training_cities": selection.get("training_cities"),
        "evaluation_cities": selection.get("evaluation_cities"),
        "counts": {s: sum(1 for x in summary.samples if x.split == s) for s in SPLITS},
        "empty_label_images": sum(1 for s in summary.samples if s.polygons == 0),
        "polygon_method": "cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); delikler korunmaz",
        "min_area_px": summary.min_area_px,
        "dropped_components_total": sum(s.dropped_components for s in summary.samples),
        "dropped_pixels_total": sum(s.dropped_pixels for s in summary.samples),
        "fidelity_iou": {
            "mean": round(float(np.mean(fids)), 4) if fids else None,
            "min": round(float(np.min(fids)), 4) if fids else None,
            "warn_threshold": summary.fidelity_warn_threshold,
            "below_threshold": [s.name for s in summary.samples if s.warning],
        },
        "samples": [asdict(s) for s in summary.samples],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Hasar maskelerini YOLOv8-seg biçimine dönüştür")
    parser.add_argument("--variant", choices=VARIANTS, default=VARIANT_MALATYA)
    parser.add_argument("--download-dir", type=Path, default=Path("data/raw/kate_cd"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-area-px", type=int, default=DEFAULT_MIN_AREA_PX)
    parser.add_argument("--epsilon-px", type=float, default=0.0,
                        help="approxPolyDP toleransı (0 = kontur aynen)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    try:
        summary = run_convert(args.variant, args.download_dir, args.processed_root,
                              min_area_px=args.min_area_px, epsilon_px=args.epsilon_px)
    except PreparationError as exc:
        logger.error("%s", exc)
        return 2
    fids = [s.fidelity_iou for s in summary.samples if s.fidelity_iou is not None]
    print(f"Çıktı: {summary.output_dir}  |  örnek: {len(summary.samples)}  |  "
          f"ort. poligon sadakati: {np.mean(fids):.3f}" if fids else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
