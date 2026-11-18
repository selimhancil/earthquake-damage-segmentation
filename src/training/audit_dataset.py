"""KATE-CD veri denetimi: Malatya alt kümesini kanıtla (şartname 2.2).

EĞİTİMDEN ÖNCE ÇALIŞTIRILMASI ZORUNLUDUR. Bu script:

1. KATE-CD'nin birincil kaynağından (CodeOcean kapsülü, CC0 veri klasörü) ara
   ön işleme dosyalarını indirir ve SHA-256 özetlerini kaydeder.
2. Son veri setindeki her çifti (train_N.png / val_N.png / test_N.png) geri
   izler:  yeni ad -> eski ad (filename_mapping.txt) -> patch -> kaynak uydu
   sahnesi (history_*.json) ve UTM sınırları + piksel boyutu.
3. Malatya'yı İKİ BAĞIMSIZ kanıtla belirler:
     a) kaynak sahne dosya adı "malatya" içerir  (post ve pre)
     b) patch'in UTM (EPSG:32637) sınırları Malatya zarfının içindedir
   İkisi birden sağlanmazsa örnek Malatya sayılmaz. Görselin görünüşünden
   ASLA tahmin yapılmaz.
4. Malatya maskelerini indirip sınıf semantiğini doğrular (yalnızca 0/1).
5. Split'ler arasında coğrafi örtüşme (sızıntı) olup olmadığını kontrol eder.
6. Sonucu reports/data_audit.json (+ malatya_pairs.csv) olarak yazar.
   Kanıt yoksa durum DATASET_BLOCKED olur ve eğitim başlatılmamalıdır.

Çalıştırma (Colab veya yerel, internet gerekir):
    python -m src.training.audit_dataset
    python -m src.training.audit_dataset --skip-masks     # yalnızca metadata
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("audit_dataset")

# --------------------------------------------------------------------------- #
# Kaynak ve sabitler
# --------------------------------------------------------------------------- #
CAPSULE_URL = "https://codeocean.com/capsule/9061546/tree/v2"
CAPSULE_DOI = "https://doi.org/10.24433/CO.3747729.v1"
FILES_BASE_URL = (
    "https://files.codeocean.com/files/verified/"
    "82765786-a936-438c-a75a-84e2817294c5_v2.0/data/"
)
PREPROCESSING_FILES = (
    "preprocessing/README.md",
    "preprocessing/history_maxar.json",
    "preprocessing/history_pleidas.json",
    "preprocessing/history_pre_post_pairs.json",
    "preprocessing/filename_mapping.txt",
    "preprocessing/label_studio_files.txt",
    "preprocessing/label_studio_annotations.csv",
)
DATA_LICENSE_FILE = "LICENSE"
PAIRS_DIR = "label_studio_pre_post"
TARGET_CITY = "malatya"
SOURCE_CRS = "EPSG:32637"

# Malatya kent merkezi ve çevresi için UTM 37N zarfı (metre).
# Malatya merkez ~ (38.35 K, 38.31 D) ≈ (E 440 km, N 4245 km). Zarf ±~40 km;
# en yakın diğer KATE-CD kaynakları (Adıyaman/Gölbaşı N≈4180 km,
# Elbistan E≈342 km, Kahramanmaraş E≈320 km) bu zarfın dışında kalır.
MALATYA_UTM_ENVELOPE = (400_000.0, 4_210_000.0, 480_000.0, 4_280_000.0)

# Güvenilir test için asgari örnek sayıları (yalnızca uyarı amaçlı eşik).
MIN_TRAIN_PAIRS = 50
MIN_EVAL_PAIRS = 30

STATUS_PASSED = "PASSED"
STATUS_BLOCKED = "DATASET_BLOCKED"

Fetcher = Callable[[str], bytes]


# --------------------------------------------------------------------------- #
# Veri yapıları
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairRecord:
    split: str  # train | val | test
    final_name: str  # ör. train_14.png (A/B/label klasörlerinde aynı ad)
    original_name: str  # ör. trn_tile_1680_patch_965__pre_tile_1680.png
    patch_key: str  # ör. tile_1680_patch_965
    post_source: Optional[str]
    pre_source: Optional[str]
    tile_bounds: Optional[Tuple[float, float, float, float]]
    gsd_m_per_pixel: Optional[float]

    @property
    def post_city(self) -> Optional[str]:
        return city_from_source(self.post_source)

    @property
    def pre_city(self) -> Optional[str]:
        return city_from_source(self.pre_source)


@dataclass
class MaskStats:
    checked: int = 0
    unique_values: List[int] = field(default_factory=list)
    binary_encoding: Optional[str] = None  # "0/1" | "0/255" | "non-binary"
    empty_masks: int = 0
    positive_pixel_ratio_mean: Optional[float] = None
    connected_regions_mean: Optional[float] = None
    wrong_size: int = 0
    errors: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Saf ayrıştırma fonksiyonları (ağ gerektirmez, test edilebilir)
# --------------------------------------------------------------------------- #
_CITY_RE = re.compile(r"(?:^|/)(?:post|pre)_([a-z]+)")
_SPLIT_PREFIX = re.compile(r"^(trn|val|tst)_")


def city_from_source(source: Optional[str]) -> Optional[str]:
    """'post_malatya_subset_PHR1A_...JP2' -> 'malatya'. Bulunamazsa None."""
    if not source:
        return None
    match = _CITY_RE.search(source.lower())
    return match.group(1) if match else None


def parse_filename_mapping(text: str) -> List[Tuple[str, str]]:
    """'./test/label/<eski>.png   ./test/label/test_1.png' satırlarını çözer."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            rows.append((parts[0], parts[1]))
    return rows


def patch_key_from_original(original_name: str) -> str:
    """'trn_tile_1680_patch_965__pre_tile_1680.png' -> 'tile_1680_patch_965'."""
    post_part = original_name.split("__")[0]
    post_part = _SPLIT_PREFIX.sub("", post_part)
    return post_part.replace(".png", "").replace(".tif", "")


def build_pair_records(
    mapping_rows: Sequence[Tuple[str, str]],
    post_history: Iterable[dict],
    pre_pairs: Iterable[dict],
) -> List[PairRecord]:
    """Son veri setindeki her çift için kaynak bilgisini birleştirir.

    Her çift A/B/label klasörlerinde üç kez geçer; 'label' satırları tek kayıt
    olarak kullanılır.
    """
    post_by_key: Dict[str, dict] = {}
    for row in post_history:
        key = str(row.get("patch_name", "")).replace(".tif", "")
        if key:
            post_by_key[key] = row
    pre_by_pair = {str(r.get("pre_post_pair_name")): r for r in pre_pairs}

    records: List[PairRecord] = []
    for old_path, new_path in mapping_rows:
        old_parts = old_path.strip("./").split("/")
        if len(old_parts) < 3 or old_parts[-2] != "label":
            continue
        split = old_parts[0]
        original = old_parts[-1]
        key = patch_key_from_original(original)
        post = post_by_key.get(key)
        pre = pre_by_pair.get(original)
        bounds = tuple(float(v) for v in post["tile_bounds"]) if post and post.get("tile_bounds") else None
        gsd = None
        if post and post.get("tif_transform"):
            a = float(post["tif_transform"][0])
            e = float(post["tif_transform"][4])
            gsd = abs(a) if abs(abs(a) - abs(e)) < 1e-9 else None
        records.append(
            PairRecord(
                split=split,
                final_name=new_path.split("/")[-1],
                original_name=original,
                patch_key=key,
                post_source=(post or {}).get("jp2_file"),
                pre_source=(pre or {}).get("jp2_file"),
                tile_bounds=bounds,  # type: ignore[arg-type]
                gsd_m_per_pixel=gsd,
            )
        )
    return records


def bounds_inside(bounds: Optional[Sequence[float]], envelope: Sequence[float]) -> bool:
    if bounds is None:
        return False
    x0, y0, x1, y1 = bounds
    ex0, ey0, ex1, ey1 = envelope
    return ex0 <= x0 and ey0 <= y0 and x1 <= ex1 and y1 <= ey1


def classify_target_city(
    record: PairRecord,
    city: str = TARGET_CITY,
    envelope: Sequence[float] = MALATYA_UTM_ENVELOPE,
) -> Tuple[bool, List[str]]:
    """Kayıt hedef şehre ait mi? (karar, kanıt/çelişki listesi)

    Malatya sayılmak için: post kaynak adı + UTM sınırları birlikte uymalı.
    Pre kaynak adı varsa o da aynı şehri göstermeli.
    """
    notes: List[str] = []
    name_ok = record.post_city == city
    geo_ok = bounds_inside(record.tile_bounds, envelope)
    pre_ok = record.pre_city in (None, city) if record.pre_source else True
    if name_ok:
        notes.append(f"post kaynak adı: {record.post_source}")
    if geo_ok:
        notes.append(f"UTM sınırları Malatya zarfında: {record.tile_bounds}")
    if name_ok != geo_ok:
        notes.append("ÇELİŞKİ: kaynak adı ile koordinat uyuşmuyor")
    if name_ok and record.pre_source and record.pre_city != city:
        notes.append(f"ÇELİŞKİ: pre kaynak başka şehir: {record.pre_source}")
    return (name_ok and geo_ok and pre_ok), notes


def find_cross_split_overlaps(records: Sequence[PairRecord]) -> List[Tuple[str, str]]:
    """Farklı split'lerdeki patch'ler coğrafi olarak örtüşüyor mu? (sızıntı)

    Sınırlar yarı açık kabul edilir: yalnızca kenarı paylaşan komşu grid
    hücreleri örtüşme sayılmaz.
    """
    overlaps = []
    items = [r for r in records if r.tile_bounds is not None]
    for i, a in enumerate(items):
        ax0, ay0, ax1, ay1 = a.tile_bounds  # type: ignore[misc]
        for b in items[i + 1:]:
            if a.split == b.split:
                continue
            bx0, by0, bx1, by1 = b.tile_bounds  # type: ignore[misc]
            if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                overlaps.append((f"{a.split}/{a.final_name}", f"{b.split}/{b.final_name}"))
    return overlaps


def summarize_cities(records: Sequence[PairRecord]) -> Dict[str, Dict[str, int]]:
    """{şehir: {split: adet}} (kaynak adından)."""
    table: Dict[str, Dict[str, int]] = {}
    for r in records:
        city = r.post_city or "UNKNOWN"
        table.setdefault(city, {})
        table[city][r.split] = table[city].get(r.split, 0) + 1
    return dict(sorted(table.items()))


def decide_gate(
    target_records: Sequence[PairRecord],
    unresolved: int,
    contradictions: int,
    mask_stats: Optional[MaskStats],
) -> Tuple[str, List[str], List[str]]:
    """Eğitim kapısı: (durum, engelleyici nedenler, uyarılar)."""
    blockers: List[str] = []
    warnings: List[str] = []

    if not target_records:
        blockers.append("Hiçbir çift Malatya olarak kanıtlanamadı (kaynak adı + koordinat).")
    if contradictions:
        blockers.append(f"{contradictions} kayıtta kaynak adı / koordinat çelişkisi var.")
    if mask_stats is not None:
        if mask_stats.binary_encoding == "non-binary":
            blockers.append(
                f"Maskeler ikili değil: {mask_stats.unique_values} (0/1 veya 0/255 beklenir)."
            )
        if mask_stats.wrong_size:
            blockers.append(f"{mask_stats.wrong_size} maske 512x512 değil.")
        if mask_stats.errors:
            blockers.append(f"{len(mask_stats.errors)} maske indirilemedi/okunamadı.")
    else:
        warnings.append("Maske semantiği bu çalıştırmada kontrol edilmedi (--skip-masks).")
    if unresolved:
        warnings.append(f"{unresolved} çiftin kaynağı çözülemedi; bunlar Malatya sayılmadı.")

    counts = {s: sum(1 for r in target_records if r.split == s) for s in ("train", "val", "test")}
    if counts["train"] < MIN_TRAIN_PAIRS:
        warnings.append(f"Malatya eğitim örneği az ({counts['train']} < {MIN_TRAIN_PAIRS}).")
    if counts["val"] + counts["test"] < MIN_EVAL_PAIRS:
        warnings.append(
            f"Malatya değerlendirme örneği az (val+test = {counts['val'] + counts['test']} "
            f"< {MIN_EVAL_PAIRS}); metrikler güvenilir test sonucu olarak sunulmamalı, "
            "belirsizlik aralığıyla raporlanmalı."
        )
    if target_records and not any(r.gsd_m_per_pixel for r in target_records):
        warnings.append("Piksel boyutu (GSD) kaynak metadata'dan çıkarılamadı.")

    status = STATUS_BLOCKED if blockers else STATUS_PASSED
    return status, blockers, warnings


# --------------------------------------------------------------------------- #
# Maske denetimi
# --------------------------------------------------------------------------- #
def inspect_masks(blobs: Iterable[Tuple[str, bytes]], expected_size: int = 512) -> MaskStats:
    """PNG maskelerinin benzersiz değerlerini ve pozitif oranlarını ölçer."""
    import cv2
    import numpy as np

    stats = MaskStats()
    values = set()
    ratios: List[float] = []
    regions: List[int] = []
    for name, data in blobs:
        try:
            arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise ValueError("çözülemedi")
            if arr.ndim == 3:
                if not (arr == arr[..., :1]).all():
                    raise ValueError("kanallar farklı (renkli maske)")
                arr = arr[..., 0]
        except Exception as exc:  # noqa: BLE001 - ayrıntı rapora yazılır
            stats.errors.append(f"{name}: {exc}")
            continue
        stats.checked += 1
        if arr.shape != (expected_size, expected_size):
            stats.wrong_size += 1
        uniq = np.unique(arr)
        values.update(int(v) for v in uniq)
        # 0/255 olarak kaydedilmiş ikili maskeler de 0/1 semantiğindedir.
        binary = arr > 0
        ratio = float(binary.mean())
        ratios.append(ratio)
        if ratio == 0:
            stats.empty_masks += 1
        else:
            n, _ = cv2.connectedComponents(binary.astype(np.uint8))
            regions.append(n - 1)
    stats.unique_values = sorted(values)
    if values <= {0, 1}:
        stats.binary_encoding = "0/1"
    elif values <= {0, 255}:
        stats.binary_encoding = "0/255"
    elif values:
        stats.binary_encoding = "non-binary"
    stats.positive_pixel_ratio_mean = round(sum(ratios) / len(ratios), 6) if ratios else None
    stats.connected_regions_mean = round(sum(regions) / len(regions), 3) if regions else None
    return stats


# --------------------------------------------------------------------------- #
# Ağ ve dosya işlemleri
# --------------------------------------------------------------------------- #
# files.codeocean.com, tarayıcı dışı istemci kimliklerini (ör. Python-urllib)
# 403 ile reddediyor. Kapsül sayfasının tarayıcıda gönderdiği başlıklar kullanılır.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://codeocean.com",
    "Referer": "https://codeocean.com/",
}


def _urllib_fetch(url: str, timeout: int) -> bytes:
    req = urllib.request.Request(url, headers=BROWSER_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _curl_fetch(url: str, timeout: int) -> bytes:
    """Yedek: sistemdeki curl (macOS ve Colab'da kurulu)."""
    import shutil
    import subprocess

    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl bulunamadı")
    cmd = [curl, "-fsSL", "--max-time", str(timeout)]
    for key, value in BROWSER_HEADERS.items():
        cmd += ["-H", f"{key}: {value}"]
    cmd.append(url)
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"curl hata kodu {proc.returncode}: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def http_fetch(url: str, retries: int = 3, timeout: int = 60) -> bytes:
    last: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        for name, method in (("urllib", _urllib_fetch), ("curl", _curl_fetch)):
            try:
                return method(url, timeout)
            except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
                last = exc
                logger.warning("İndirme (%s) denemesi %d/%d başarısız: %s (%s)",
                               name, attempt, retries, url, exc)
        time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(
        f"İndirilemedi: {url}: {last}. Tarayıcıda açılabiliyorsa dosyaları elle indirip "
        "--download-dir altına koyun ve --offline ile çalıştırın."
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_metadata(fetch: Fetcher, base_url: str, out_dir: Path) -> Dict[str, dict]:
    """Ön işleme dosyalarını indirir, diske yazar; {yol: {sha256, bytes}} döndürür."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, dict] = {}
    for rel in PREPROCESSING_FILES + (DATA_LICENSE_FILE,):
        data = fetch(base_url + rel)
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest[rel] = {"sha256": sha256(data), "bytes": len(data)}
        logger.info("İndirildi: %s (%d bayt)", rel, len(data))
    return manifest


# --------------------------------------------------------------------------- #
# Ana akış
# --------------------------------------------------------------------------- #
def run_audit(
    fetch: Fetcher = http_fetch,
    base_url: str = FILES_BASE_URL,
    download_dir: Path = Path("data/raw/kate_cd"),
    report_path: Path = Path("reports/data_audit.json"),
    check_masks: bool = True,
) -> dict:
    started = datetime.now(timezone.utc)
    manifest = download_metadata(fetch, base_url, download_dir)

    def read(rel: str) -> str:
        return (download_dir / rel).read_text(encoding="utf-8")

    history = json.loads(read("preprocessing/history_maxar.json")) + json.loads(
        read("preprocessing/history_pleidas.json")
    )
    pre_pairs = json.loads(read("preprocessing/history_pre_post_pairs.json"))
    mapping = parse_filename_mapping(read("preprocessing/filename_mapping.txt"))
    records = build_pair_records(mapping, history, pre_pairs)
    unresolved = [r for r in records if r.post_source is None]

    target: List[PairRecord] = []
    contradictions: List[dict] = []
    evidence: Dict[str, List[str]] = {}
    for rec in records:
        is_target, notes = classify_target_city(rec)
        if any(n.startswith("ÇELİŞKİ") for n in notes):
            contradictions.append({"pair": f"{rec.split}/{rec.final_name}", "notes": notes})
        if is_target:
            target.append(rec)
            evidence[f"{rec.split}/{rec.final_name}"] = notes

    mask_stats: Optional[MaskStats] = None
    if check_masks and target:
        def blobs():
            for rec in target:
                rel = f"{PAIRS_DIR}/{rec.split}/label/{rec.final_name}"
                try:
                    yield rel, fetch(base_url + rel)
                except Exception as exc:  # noqa: BLE001
                    yield rel, f"HATA {exc}".encode()
        mask_stats = inspect_masks(blobs())

    overlaps_target = find_cross_split_overlaps(target)
    overlaps_all = find_cross_split_overlaps(records)
    status, blockers, warnings = decide_gate(target, len(unresolved), len(contradictions), mask_stats)
    if overlaps_target:
        warnings.append(f"Malatya alt kümesinde split'ler arası {len(overlaps_target)} coğrafi örtüşme var.")

    counts = {s: sum(1 for r in target if r.split == s) for s in ("train", "val", "test")}
    gsds = sorted({r.gsd_m_per_pixel for r in target if r.gsd_m_per_pixel})
    report = {
        "status": status,
        "blockers": blockers,
        "warnings": warnings,
        "generated_at_utc": started.isoformat(),
        "dataset": {
            "name": "KATE-CD (Kahramanmaraş Türkiye Earthquake-Change Detection)",
            "primary_source": CAPSULE_URL,
            "doi": CAPSULE_DOI,
            "files_base_url": base_url,
            "data_license_first_line": read(DATA_LICENSE_FILE).splitlines()[0].strip(),
            "imagery_note": (
                "Alttaki Maxar Open Data / Airbus Pleiades görüntülerinin kullanım koşulları "
                "yayından önce ayrıca kontrol edilmelidir."
            ),
            "metadata_files": manifest,
            "total_final_pairs": len(records),
            "pairs_by_source_city": summarize_cities(records),
            "unresolved_pairs": len(unresolved),
        },
        "malatya": {
            "selection_method": (
                "Bir çift yalnızca şu iki koşul birlikte sağlanırsa Malatya sayılır: (a) post "
                "kaynak sahne adı 'malatya' içerir (pre kaynak varsa o da Malatya), (b) patch'in "
                f"{SOURCE_CRS} sınırları Malatya zarfı {MALATYA_UTM_ENVELOPE} içindedir. "
                "Görsel görünüşe dayalı tahmin yapılmaz."
            ),
            "pair_count": len(target),
            "by_split": counts,
            "post_sources": sorted({r.post_source for r in target if r.post_source}),
            "pre_sources": sorted({r.pre_source for r in target if r.pre_source}),
            "utm_bbox": (
                [min(r.tile_bounds[0] for r in target), min(r.tile_bounds[1] for r in target),
                 max(r.tile_bounds[2] for r in target), max(r.tile_bounds[3] for r in target)]
                if target else None
            ),
            "gsd_m_per_pixel_from_source_metadata": gsds,
            "contradictions": contradictions,
            "cross_split_overlaps": overlaps_target,
            "evidence": evidence,
        },
        "all_pairs_cross_split_overlaps": len(overlaps_all),
        "mask_semantics": {
            "documented": "1 = hasar poligonu içi (damaged), 0 = işaretlenmemiş (hasarsız bina DEĞİL)",
            "label_unit": "hasar poligonu / bölge (tekil bina garantisi yok)",
            "checked": asdict(mask_stats) if mask_stats else None,
        },
        "calibration": {
            "patch_files_carry_georeference": False,
            "note": (
                "PNG patch'ler coğrafi dönüşüm taşımaz; GSD kaynak sahnenin tif_transform "
                "değerinden alınır (kaynak metadata ile doğrulanmış)."
            ),
        },
        "training_gate": {
            "training_allowed": status == STATUS_PASSED,
            "yolo_class": "0: damaged_region",
            "undamaged_building_class": "pasif (doğrulanmış footprint/etiket yok)",
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_pairs_csv(target, report_path.with_name("malatya_pairs.csv"))
    logger.info("Denetim raporu: %s (durum: %s)", report_path, status)
    return report


def write_pairs_csv(records: Sequence[PairRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "final_name", "original_name", "post_source", "pre_source",
                         "utm_x_min", "utm_y_min", "utm_x_max", "utm_y_max", "gsd_m_per_pixel"])
        for r in sorted(records, key=lambda x: (x.split, int(re.sub(r"\D", "", x.final_name) or 0))):
            b = r.tile_bounds or (None, None, None, None)
            writer.writerow([r.split, r.final_name, r.original_name, r.post_source, r.pre_source,
                             *b, r.gsd_m_per_pixel])


def print_summary(report: dict) -> None:
    mal = report["malatya"]
    print("=" * 70)
    print(f"DURUM: {report['status']}")
    print(f"Toplam çift: {report['dataset']['total_final_pairs']}  |  Malatya: {mal['pair_count']} "
          f"(train {mal['by_split']['train']}, val {mal['by_split']['val']}, test {mal['by_split']['test']})")
    print(f"GSD (kaynak metadata): {mal['gsd_m_per_pixel_from_source_metadata']} m/piksel")
    checked = report["mask_semantics"]["checked"]
    if checked:
        print(f"Maske değerleri: {checked['unique_values']} ({checked['binary_encoding']})  |  "
              f"boş maske: {checked['empty_masks']}")
    for b in report["blockers"]:
        print(f"ENGEL: {b}")
    for w in report["warnings"]:
        print(f"UYARI: {w}")
    print("=" * 70)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="KATE-CD Malatya veri denetimi")
    parser.add_argument("--download-dir", type=Path, default=Path("data/raw/kate_cd"))
    parser.add_argument("--report", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--skip-masks", action="store_true", help="Maske semantiği kontrolünü atla")
    parser.add_argument("--base-url", default=FILES_BASE_URL)
    parser.add_argument(
        "--offline", action="store_true",
        help="İndirme yapma; dosyaları --download-dir altından oku (ör. tarayıcıdan elle "
             "indirilmiş preprocessing/ klasörü). Maskeler yoksa --skip-masks ekleyin.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    if args.offline:
        local_root = args.download_dir.resolve()
        fetch: Fetcher = lambda url: Path(url).read_bytes()  # noqa: E731
        base_url = str(local_root) + "/"
        logger.info("Çevrimdışı mod: dosyalar %s altından okunuyor", local_root)
    else:
        fetch, base_url = http_fetch, args.base_url
    report = run_audit(fetch=fetch, base_url=base_url, download_dir=args.download_dir,
                       report_path=args.report, check_masks=not args.skip_masks)
    print_summary(report)
    return 0 if report["status"] == STATUS_PASSED else 2


if __name__ == "__main__":
    sys.exit(main())
