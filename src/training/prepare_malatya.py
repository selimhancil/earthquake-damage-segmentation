"""Eğitim verisini seç ve indir (şartname 2.3 öncesi adım).

Yalnızca veri denetimi (audit_dataset.py) PASSED ise çalışır.

Veri varyantları:
  malatya (VARSAYILAN)
      train = Malatya train (53), val = Malatya val (5), test = Malatya test (4)
  kate7_eval_malatya (AÇIKÇA seçilmelidir)
      train = KATE-CD'nin tüm train bölümü (404; Malatya'nın 53'ü dahil)
      val   = yalnızca Malatya val (5), test = yalnızca Malatya test (4)
      Diğer şehirlerin val/test örnekleri hiç kullanılmaz. Raporlarda eğitim
      verisinin birden çok şehirden geldiği açıkça yazılmalıdır.

KATE-CD'nin orijinal split'i korunur: denetim, split'ler arasında coğrafi
örtüşme olmadığını doğrulamıştır (cross_split_overlaps = 0).

Çalıştırma:
    python -m src.training.prepare_malatya
    python -m src.training.prepare_malatya --variant kate7_eval_malatya
    python -m src.training.prepare_malatya --offline   # yalnızca yerel dosyalar (ör. Colab)

Not: CodeOcean dosya sunucusu bulut sağlayıcılarının (ör. Google Colab) isteklerini
403 ile reddedebilir. O durumda veriyi kendi bilgisayarınızda bu script ile indirip
data/raw/kate_cd klasörünü Colab'a taşıyın ve --offline ile çalıştırın.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from src.training import audit_dataset as audit

logger = logging.getLogger("prepare_malatya")

VARIANT_MALATYA = "malatya"
VARIANT_KATE7 = "kate7_eval_malatya"
VARIANTS = (VARIANT_MALATYA, VARIANT_KATE7)
FOLDERS = ("A", "B", "label")  # A = deprem öncesi, B = deprem sonrası, label = maske

Fetcher = Callable[[str], bytes]


class PreparationError(RuntimeError):
    """Veri hazırlama güvenlik kapısı veya girdi hatası."""


@dataclass(frozen=True)
class SelectedPair:
    split: str  # çıktıdaki split (train/val/test)
    source_split: str  # KATE-CD'deki split
    name: str  # train_14.png vb.
    city: str  # kaynak sahneden çıkarılan şehir


# --------------------------------------------------------------------------- #
# Güvenlik kapısı
# --------------------------------------------------------------------------- #
def load_passed_audit(report_path: Path) -> dict:
    """Denetim raporunu okur; PASSED değilse PreparationError fırlatır."""
    if not report_path.is_file():
        raise PreparationError(
            f"Denetim raporu bulunamadı: {report_path}. Önce şunu çalıştırın: "
            "python -m src.training.audit_dataset"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != audit.STATUS_PASSED:
        raise PreparationError(
            f"Veri denetimi geçmedi (durum: {report.get('status')}). Eğitim verisi hazırlanmaz. "
            f"Engeller: {report.get('blockers')}"
        )
    if report.get("malatya", {}).get("cross_split_overlaps"):
        raise PreparationError("Malatya alt kümesinde split'ler arası coğrafi örtüşme var; split yeniden yapılmalı.")
    return report


def verify_local_metadata(report: dict, download_dir: Path, required: bool = False) -> List[str]:
    """Yerel metadata dosyalarının SHA-256'sı denetim raporundakilerle aynı mı?

    Veri başka bir makinede indirilip taşındığında, denetlenen dosyalarla aynı
    olduğunu kanıtlar. Döndürülen liste doğrulanan dosyalardır.
    """
    expected = report.get("dataset", {}).get("metadata_files", {})
    verified, missing = [], []
    for rel, info in expected.items():
        path = download_dir / rel
        if not path.is_file():
            missing.append(rel)
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != info.get("sha256"):
            raise PreparationError(
                f"{rel} denetlenen dosyadan farklı (SHA-256 uyuşmuyor). Denetimi yeniden çalıştırın."
            )
        verified.append(rel)
    if required and missing:
        raise PreparationError(f"Denetim metadata'sı eksik: {missing}")
    return verified


def offline_fetch(url: str) -> bytes:
    raise PreparationError(
        f"Çevrimdışı modda eksik dosya: {url.split('/data/')[-1]}. Veriyi internete erişebilen bir "
        "bilgisayarda 'python -m src.training.prepare_malatya --variant <varyant>' ile indirip "
        "data/raw/kate_cd klasörünü buraya kopyalayın."
    )


# --------------------------------------------------------------------------- #
# Seçim
# --------------------------------------------------------------------------- #
def read_malatya_pairs(csv_path: Path) -> List[SelectedPair]:
    import csv

    if not csv_path.is_file():
        raise PreparationError(f"Malatya listesi bulunamadı: {csv_path} (audit_dataset.py üretir).")
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [SelectedPair(r["split"], r["split"], r["final_name"], audit.TARGET_CITY) for r in rows]


def all_pairs_from_metadata(download_dir: Path) -> List[SelectedPair]:
    """Denetimin indirdiği metadata'dan 486 çiftin tamamı (şehir bilgisiyle)."""
    pre = download_dir / "preprocessing"
    try:
        history = json.loads((pre / "history_maxar.json").read_text(encoding="utf-8")) + json.loads(
            (pre / "history_pleidas.json").read_text(encoding="utf-8")
        )
        pairs = json.loads((pre / "history_pre_post_pairs.json").read_text(encoding="utf-8"))
        mapping = audit.parse_filename_mapping((pre / "filename_mapping.txt").read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PreparationError(
            f"Denetim metadata'sı eksik ({exc.filename}). Önce audit_dataset.py çalıştırılmalı."
        ) from exc
    records = audit.build_pair_records(mapping, history, pairs)
    selected = []
    for rec in records:
        is_malatya, _ = audit.classify_target_city(rec)
        city = audit.TARGET_CITY if is_malatya else (rec.post_city or "unknown")
        selected.append(SelectedPair(rec.split, rec.split, rec.final_name, city))
    return selected


def select_pairs(variant: str, malatya: Sequence[SelectedPair],
                 all_pairs: Optional[Sequence[SelectedPair]] = None) -> List[SelectedPair]:
    """Varyanta göre eğitim/val/test listesini kurar."""
    if variant == VARIANT_MALATYA:
        return sorted(malatya, key=_sort_key)
    if variant == VARIANT_KATE7:
        if all_pairs is None:
            raise PreparationError("kate7_eval_malatya için tüm çiftlerin listesi gerekli.")
        malatya_names = {(p.source_split, p.name) for p in malatya}
        train = [p for p in all_pairs if p.source_split == "train"]
        evaluation = [p for p in all_pairs if p.source_split in ("val", "test")
                      and (p.source_split, p.name) in malatya_names]
        # tutarlılık: metadata'dan çıkan Malatya listesi audit CSV'siyle aynı olmalı
        derived = {(p.source_split, p.name) for p in all_pairs if p.city == audit.TARGET_CITY}
        if derived != malatya_names:
            raise PreparationError("Metadata'dan çıkan Malatya listesi malatya_pairs.csv ile uyuşmuyor.")
        return sorted(train + evaluation, key=_sort_key)
    raise PreparationError(f"Bilinmeyen varyant: {variant}. Seçenekler: {VARIANTS}")


def _sort_key(p: SelectedPair):
    order = {"train": 0, "val": 1, "test": 2}
    digits = "".join(ch for ch in p.name if ch.isdigit())
    return order.get(p.split, 9), int(digits or 0)


def summarize(selection: Sequence[SelectedPair]) -> Dict[str, Dict[str, int]]:
    table: Dict[str, Dict[str, int]] = {}
    for p in selection:
        table.setdefault(p.split, {})
        table[p.split][p.city] = table[p.split].get(p.city, 0) + 1
    return table


# --------------------------------------------------------------------------- #
# İndirme
# --------------------------------------------------------------------------- #
def download_pairs(selection: Sequence[SelectedPair], fetch: Fetcher, base_url: str,
                   out_dir: Path, workers: int = 8) -> Dict[str, dict]:
    """Her çift için A/B/label PNG'lerini indirir. Var olan dosyalar yeniden indirilmez."""
    jobs = []
    for pair in selection:
        for folder in FOLDERS:
            rel = f"{pair.source_split}/{folder}/{pair.name}"
            jobs.append(rel)

    def work(rel: str):
        target = out_dir / rel
        if target.is_file() and target.stat().st_size > 0:
            data = target.read_bytes()
        else:
            data = fetch(f"{base_url}{audit.PAIRS_DIR}/{rel}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return rel, {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    manifest: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, (rel, info) in enumerate(pool.map(work, jobs), start=1):
            manifest[rel] = info
            if i % 100 == 0 or i == len(jobs):
                logger.info("İndirilen dosya: %d/%d", i, len(jobs))
    return manifest


# --------------------------------------------------------------------------- #
# Ana akış
# --------------------------------------------------------------------------- #
def run_prepare(
    variant: str = VARIANT_MALATYA,
    audit_report: Path = Path("reports/data_audit.json"),
    malatya_csv: Path = Path("reports/malatya_pairs.csv"),
    download_dir: Path = Path("data/raw/kate_cd"),
    fetch: Fetcher = audit.http_fetch,
    base_url: str = audit.FILES_BASE_URL,
    workers: int = 8,
) -> dict:
    report = load_passed_audit(audit_report)
    malatya = read_malatya_pairs(malatya_csv)
    if len(malatya) != report["malatya"]["pair_count"]:
        raise PreparationError("malatya_pairs.csv ile denetim raporundaki Malatya sayısı farklı.")

    verified = verify_local_metadata(report, download_dir, required=(variant == VARIANT_KATE7))
    if verified:
        logger.info("Yerel metadata denetim raporuyla aynı (SHA-256): %d dosya", len(verified))
    all_pairs = all_pairs_from_metadata(download_dir) if variant == VARIANT_KATE7 else None
    if variant == VARIANT_KATE7:
        logger.warning(
            "VARYANT kate7_eval_malatya: eğitimde Malatya DIŞI şehirler de kullanılacak. "
            "Değerlendirme yalnızca Malatya val/test üzerinde yapılır; raporlarda belirtin."
        )
    selection = select_pairs(variant, malatya, all_pairs)
    pairs_dir = download_dir / "pairs"
    manifest = download_pairs(selection, fetch, base_url, pairs_dir, workers=workers)

    out = {
        "variant": variant,
        "audit_status": report["status"],
        "metadata_verified_sha256": verified,
        "training_cities": sorted({p.city for p in selection if p.split == "train"}),
        "evaluation_cities": sorted({p.city for p in selection if p.split in ("val", "test")}),
        "counts": summarize(selection),
        "pairs_dir": str(pairs_dir),
        "selection": [asdict(p) for p in selection],
        "files": manifest,
        "notes": [
            "A = deprem öncesi, B = deprem sonrası (modelin tek girdisi), label = hasar maskesi",
            "KATE-CD orijinal split'i korunmuştur (denetimde split'ler arası örtüşme yok).",
        ],
    }
    if variant == VARIANT_KATE7:
        out["notes"].append(
            "Eğitim verisi birden çok şehirden gelir; sonuçlar 'Malatya modeli' olarak değil, "
            "'7 şehirle eğitilmiş, Malatya'da değerlendirilmiş model' olarak raporlanmalıdır."
        )
    selection_path = download_dir / f"selection_{variant}.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Seçim kaydedildi: %s", selection_path)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Malatya eğitim verisini seç ve indir")
    parser.add_argument("--variant", choices=VARIANTS, default=VARIANT_MALATYA)
    parser.add_argument("--audit-report", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--malatya-csv", type=Path, default=Path("reports/malatya_pairs.csv"))
    parser.add_argument("--download-dir", type=Path, default=Path("data/raw/kate_cd"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--offline", action="store_true",
                        help="İndirme yapma; eksik dosya varsa açıklayıcı hata ver")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    try:
        out = run_prepare(args.variant, args.audit_report, args.malatya_csv, args.download_dir,
                          fetch=offline_fetch if args.offline else audit.http_fetch,
                          workers=args.workers)
    except PreparationError as exc:
        logger.error("%s", exc)
        return 2
    print(f"Varyant: {out['variant']}")
    for split, cities in out["counts"].items():
        print(f"  {split}: {sum(cities.values())}  {cities}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
