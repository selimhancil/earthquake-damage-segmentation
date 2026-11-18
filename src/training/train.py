"""YOLOv8n-seg fine-tune (şartname 2.4). Colab T4 için tasarlandı.

Güvenlik kapıları:
- reports/data_audit.json durumu PASSED olmalı.
- data/processed/<varyant>_yolo içinde train/val görüntüleri olmalı.
- Başlangıç ağırlığı her zaman önceden eğitilmiş yolov8n-seg.pt'dir (fine-tune);
  sonuç modeli yalnızca 'damaged_region' sınıfını içerir.

Çıktılar:
- runs/segment/<deney>/            Ultralytics çıktıları
- weights/<deney>_best.pt          en iyi ağırlık kopyası
- weights/best_model.pt            yalnızca --export-weights ile (uygulamanın kullandığı)
- reports/experiment_config_<deney>.json ve reports/experiment_config.json

Çalıştırma:
    python -m src.training.train --variant malatya
    python -m src.training.train --variant kate7_eval_malatya --export-weights
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from src.training.convert_to_yolo import CLASS_NAME, SPLITS, output_dir_for
from src.training.prepare_malatya import VARIANT_MALATYA, VARIANTS, PreparationError, load_passed_audit

logger = logging.getLogger("train")

DEFAULTS = {
    "model": "yolov8n-seg.pt",
    "epochs": 30,
    "imgsz": 512,
    "batch": 8,
    "patience": 10,
    "seed": 42,
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def count_images(dataset_dir: Path) -> dict:
    return {
        split: sum(1 for p in (dataset_dir / "images" / split).glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if (dataset_dir / "images" / split).is_dir() else 0
        for split in SPLITS
    }


def check_dataset(dataset_dir: Path) -> dict:
    counts = count_images(dataset_dir)
    if counts["train"] == 0 or counts["val"] == 0:
        raise PreparationError(
            f"{dataset_dir} içinde train/val görüntüsü yok ({counts}). Önce prepare_malatya.py ve "
            "convert_to_yolo.py çalıştırılmalı."
        )
    for split in SPLITS:
        images = {p.stem for p in (dataset_dir / "images" / split).glob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
        labels = {p.stem for p in (dataset_dir / "labels" / split).glob("*.txt")}
        if images != labels:
            raise PreparationError(f"{split}: görüntü ve etiket dosyaları eşleşmiyor ({len(images)} / {len(labels)}).")
    return counts


def write_runtime_yaml(dataset_dir: Path, target: Path) -> Path:
    """Ultralytics göreli 'path'i kendi datasets klasörüne göre çözer; mutlak yol yazılır."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f"path: {dataset_dir.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n  0: {CLASS_NAME}\n",
        encoding="utf-8",
    )
    return target


def resolve_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def library_versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in ("ultralytics", "torch", "cv2", "numpy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "?")
        except ImportError:
            versions[name] = None
    return versions


def build_config(variant: str, experiment: str, params: dict, counts: dict, device: str,
                 audit_report: dict, runtime_yaml: Path) -> dict:
    selection_note = (
        "Eğitim yalnızca Malatya örnekleriyle yapılır."
        if variant == VARIANT_MALATYA
        else "Eğitimde KATE-CD'nin tüm train bölümü (7 kaynak şehir) kullanılır; val/test yalnızca Malatya."
    )
    return {
        "experiment": experiment,
        "variant": variant,
        "selection_note": selection_note,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "params": {**params, "device": device, "deterministic": True},
        "dataset": {"yaml": str(runtime_yaml), "image_counts": counts},
        "audit": {
            "status": audit_report["status"],
            "malatya_pairs": audit_report["malatya"]["pair_count"],
            "gsd_m_per_pixel": audit_report["malatya"]["gsd_m_per_pixel_from_source_metadata"],
        },
        "versions": library_versions(),
        "note": "Bu değerler deney başlangıcıdır; performans garantisi değildir.",
    }


def run_train(
    variant: str = VARIANT_MALATYA,
    experiment: Optional[str] = None,
    audit_path: Path = Path("reports/data_audit.json"),
    processed_root: Path = Path("data/processed"),
    runs_dir: Path = Path("runs/segment"),
    weights_dir: Path = Path("weights"),
    reports_dir: Path = Path("reports"),
    device: Optional[str] = None,
    export_weights: bool = False,
    yolo_factory=None,
    **overrides,
) -> dict:
    audit_report = load_passed_audit(audit_path)
    dataset_dir = output_dir_for(variant, processed_root)
    counts = check_dataset(dataset_dir)
    params = {**DEFAULTS, **{k: v for k, v in overrides.items() if v is not None}}
    experiment = experiment or f"{variant}_{Path(params['model']).stem}_e{params['epochs']}_s{params['seed']}"
    device = resolve_device(device)
    runtime_yaml = write_runtime_yaml(dataset_dir, runs_dir / experiment / "data.yaml")

    config = build_config(variant, experiment, params, counts, device, audit_report, runtime_yaml)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Deney: %s | varyant: %s | cihaz: %s | görüntüler: %s", experiment, variant, device, counts)

    if yolo_factory is None:
        from ultralytics import YOLO as yolo_factory  # noqa: N811

    model = yolo_factory(params["model"])
    results = model.train(
        data=str(runtime_yaml),
        epochs=params["epochs"],
        imgsz=params["imgsz"],
        batch=params["batch"],
        patience=params["patience"],
        seed=params["seed"],
        deterministic=True,
        device=device,
        project=str(runs_dir.resolve()),
        name=experiment,
        exist_ok=True,
        verbose=True,
    )
    save_dir = Path(getattr(results, "save_dir", None) or getattr(model.trainer, "save_dir", runs_dir / experiment))
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        raise PreparationError(f"Eğitim bitti ama best.pt bulunamadı: {best}")

    weights_dir.mkdir(parents=True, exist_ok=True)
    best_copy = weights_dir / f"{experiment}_best.pt"
    shutil.copyfile(best, best_copy)
    config["outputs"] = {"run_dir": str(save_dir), "best_weights": str(best_copy)}
    if export_weights:
        shutil.copyfile(best, weights_dir / "best_model.pt")
        config["outputs"]["app_weights"] = str(weights_dir / "best_model.pt")

    text = json.dumps(config, ensure_ascii=False, indent=2)
    (reports_dir / f"experiment_config_{experiment}.json").write_text(text, encoding="utf-8")
    (reports_dir / "experiment_config.json").write_text(text, encoding="utf-8")
    logger.info("En iyi ağırlık: %s", best_copy)
    return config


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="YOLOv8n-seg fine-tune")
    parser.add_argument("--variant", choices=VARIANTS, default=VARIANT_MALATYA)
    parser.add_argument("--name", dest="experiment")
    parser.add_argument("--model", default=None, help="Başlangıç ağırlığı (varsayılan yolov8n-seg.pt)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default=None, help="'0' (GPU), 'mps', 'cpu'; boşsa otomatik")
    parser.add_argument("--export-weights", action="store_true",
                        help="best.pt'yi weights/best_model.pt olarak da kopyala (uygulama bunu kullanır)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    try:
        config = run_train(args.variant, args.experiment, device=args.device, export_weights=args.export_weights,
                           model=args.model, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                           patience=args.patience, seed=args.seed)
    except PreparationError as exc:
        logger.error("%s", exc)
        return 2
    print(f"Tamamlandı: {config['outputs']['best_weights']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
