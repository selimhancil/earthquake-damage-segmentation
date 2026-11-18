"""Veri hazırlama testleri (şartname bölüm 8 - test_dataset_conversion.py).

Bu dosyada şimdilik veri DENETİMİ (audit_dataset.py) testleri var:
- KATE-CD'nin gerçek dosya biçimlerini taklit eden küçük sabitlerle (ağsız)
- Malatya kanıtı yoksa fail-closed (DATASET_BLOCKED)
- Split'ler arası coğrafi örtüşme (sızıntı) tespiti
- İsteğe bağlı: gerçek KATE-CD metadata'sıyla ağ testi (EV_NETWORK_TESTS=1)

YOLO dönüştürme testleri convert_to_yolo.py ile birlikte eklenecek.
"""

import json
import os

import cv2
import numpy as np
import pytest

from src.training import audit_dataset as audit
from src.training.audit_dataset import (
    STATUS_BLOCKED,
    STATUS_PASSED,
    MaskStats,
    PairRecord,
    build_pair_records,
    city_from_source,
    classify_target_city,
    decide_gate,
    find_cross_split_overlaps,
    inspect_masks,
    parse_filename_mapping,
    patch_key_from_original,
    run_audit,
)

MALATYA_POST = "post_malatya_subset_PHR1A_20230209_utm_jpeg_merged_shifted.JP2"
MALATYA_PRE = "pre_malatya_subset_PHR1B_20200719_utm_jpeg_merged_aligned.JP2"
HATAY_POST = "post_hatay_merkez_PHR1B_20230208_utm_jpeg_merged_shifted.JP2"
HATAY_PRE = "pre_hatay_merkez_PHR1B_20200921_utm_jpeg_merged_aligned.JP2"
MALATYA_XY = (431313.25, 4240750.25)  # gerçek Malatya patch'lerinin köşe bölgesi
HATAY_XY = (243665.25, 4013166.25)
PATCH_M = 256.0  # 512 piksel × 0.5 m


def bounds(x, y, size=PATCH_M):
    return [x, y, x + size, y + size]


def history_row(tile, patch, source, x, y):
    return {
        "tile_number": None,
        "patch_name": f"tile_{tile}_patch_{patch}.tif",
        "jp2_file": source,
        "tile_bounds": bounds(x, y),
        "tif_transform": [0.5, 0, x - 1000, 0, -0.5, y + 5000, 0, 0, 1],
    }


def mapping_lines(split, prefix, tile, patch, index):
    old = f"{prefix}_tile_{tile}_patch_{patch}__pre_tile_{tile}.png"
    return [f"./{split}/{folder}/{old}    ./{split}/{folder}/{split}_{index}.png"
            for folder in ("A", "B", "label")], old


def build_fixture(malatya_specs, other_specs):
    """(split, prefix, tile, patch, index, xoff) listelerinden KATE-CD benzeri metadata."""
    history, pairs, lines = [], [], []
    for specs, post_src, pre_src, (x0, y0) in (
        (malatya_specs, MALATYA_POST, MALATYA_PRE, MALATYA_XY),
        (other_specs, HATAY_POST, HATAY_PRE, HATAY_XY),
    ):
        for split, prefix, tile, patch, index, xoff in specs:
            x = x0 + xoff
            history.append(history_row(tile, patch, post_src, x, y0))
            rows, old = mapping_lines(split, prefix, tile, patch, index)
            lines.extend(rows)
            pairs.append({"pre_post_pair_name": old, "jp2_file": pre_src})
    return history, pairs, "\n".join(lines) + "\n"


MALATYA_SPECS = [
    ("train", "trn", 1680, 965, 14, 0.0),
    ("train", "trn", 1681, 966, 28, 256.0),
    ("val", "val", 1700, 980, 28, 2048.0),
    ("test", "tst", 1724, 1009, 1, 4096.0),
]
HATAY_SPECS = [
    ("train", "trn", 570, 317, 1, 0.0),
    ("test", "tst", 571, 318, 2, 512.0),
]


# --------------------------------------------------------------------------- #
# Ayrıştırma
# --------------------------------------------------------------------------- #
class TestParsing:
    @pytest.mark.parametrize("source, city", [
        (MALATYA_POST, "malatya"),
        (MALATYA_PRE, "malatya"),
        ("pre_malatya_visual_merged_aligned.JP2", "malatya"),
        ("bedas-dataset/maxar_post/reprojected_and_shifted_maxar_files/"
         "post_bahce_visual_reprojected_shifted.JP2", "bahce"),
        (HATAY_POST, "hatay"),
        (None, None),
        ("random_file.JP2", None),
    ])
    def test_city_from_source(self, source, city):
        assert city_from_source(source) == city

    def test_patch_key(self):
        assert patch_key_from_original("trn_tile_1680_patch_965__pre_tile_1680.png") == "tile_1680_patch_965"
        assert patch_key_from_original("tst_tile_maxar_213_patch_2__pre_tile_maxar_213.png") == \
            "tile_maxar_213_patch_2"

    def test_parse_mapping_ignores_blank_lines(self):
        rows = parse_filename_mapping("./a/label/x.png   ./a/label/a_1.png  \n\n  \n")
        assert rows == [("./a/label/x.png", "./a/label/a_1.png")]

    def test_build_records_uses_label_rows_only(self):
        history, pairs, text = build_fixture(MALATYA_SPECS, HATAY_SPECS)
        records = build_pair_records(parse_filename_mapping(text), history, pairs)
        assert len(records) == len(MALATYA_SPECS) + len(HATAY_SPECS)  # A/B/label tek kayıt
        first = next(r for r in records if r.final_name == "train_14.png")
        assert first.split == "train"
        assert first.patch_key == "tile_1680_patch_965"
        assert first.post_source == MALATYA_POST
        assert first.pre_source == MALATYA_PRE
        assert first.gsd_m_per_pixel == 0.5
        assert first.tile_bounds == tuple(bounds(*MALATYA_XY))

    def test_missing_history_leaves_source_unresolved(self):
        history, pairs, text = build_fixture(MALATYA_SPECS, [])
        records = build_pair_records(parse_filename_mapping(text), history[1:], pairs)
        unresolved = [r for r in records if r.post_source is None]
        assert len(unresolved) == 1
        assert unresolved[0].tile_bounds is None


# --------------------------------------------------------------------------- #
# Malatya sınıflandırması (iki bağımsız kanıt)
# --------------------------------------------------------------------------- #
def record(post=MALATYA_POST, pre=MALATYA_PRE, xy=MALATYA_XY, split="train", name="train_1.png"):
    return PairRecord(split, name, "x.png", "tile_1_patch_1", post, pre,
                      tuple(bounds(*xy)) if xy else None, 0.5)


class TestClassification:
    def test_name_and_coordinates_agree(self):
        ok, notes = classify_target_city(record())
        assert ok
        assert not any(n.startswith("ÇELİŞKİ") for n in notes)

    def test_name_without_coordinates_is_rejected(self):
        ok, notes = classify_target_city(record(xy=HATAY_XY))
        assert not ok
        assert any("ÇELİŞKİ" in n for n in notes)

    def test_coordinates_without_name_is_rejected(self):
        ok, notes = classify_target_city(record(post=HATAY_POST, pre=HATAY_PRE))
        assert not ok
        assert any("ÇELİŞKİ" in n for n in notes)

    def test_missing_bounds_is_rejected(self):
        ok, _ = classify_target_city(record(xy=None))
        assert not ok

    def test_pre_image_from_other_city_is_rejected(self):
        ok, notes = classify_target_city(record(pre=HATAY_PRE))
        assert not ok
        assert any("pre kaynak" in n for n in notes)

    def test_other_city_is_rejected_cleanly(self):
        ok, notes = classify_target_city(record(post=HATAY_POST, pre=HATAY_PRE, xy=HATAY_XY))
        assert not ok
        assert notes == []

    @pytest.mark.parametrize("label, xy", [
        ("Adıyaman/Gölbaşı (N≈4182 km)", (436000.0, 4182000.0)),
        ("Elbistan (E≈342 km)", (342000.0, 4232000.0)),
        ("Kahramanmaraş (E≈320 km)", (320000.0, 4160000.0)),
    ])
    def test_envelope_excludes_nearby_cities(self, label, xy):
        assert not audit.bounds_inside(bounds(*xy), audit.MALATYA_UTM_ENVELOPE), label


# --------------------------------------------------------------------------- #
# Sızıntı
# --------------------------------------------------------------------------- #
class TestLeakage:
    def test_adjacent_grid_cells_are_not_overlaps(self):
        a = record(split="train", name="a.png")
        b = record(split="test", name="b.png", xy=(MALATYA_XY[0] + PATCH_M, MALATYA_XY[1]))
        assert find_cross_split_overlaps([a, b]) == []

    def test_overlapping_patches_across_splits_are_reported(self):
        a = record(split="train", name="a.png")
        b = record(split="test", name="b.png", xy=(MALATYA_XY[0] + 100, MALATYA_XY[1] + 100))
        assert find_cross_split_overlaps([a, b]) == [("train/a.png", "test/b.png")]

    def test_overlaps_within_same_split_are_ignored(self):
        a = record(split="train", name="a.png")
        b = record(split="train", name="b.png", xy=(MALATYA_XY[0] + 100, MALATYA_XY[1]))
        assert find_cross_split_overlaps([a, b]) == []


# --------------------------------------------------------------------------- #
# Maske semantiği
# --------------------------------------------------------------------------- #
def png(array):
    ok, buf = cv2.imencode(".png", array)
    assert ok
    return buf.tobytes()


class TestMaskInspection:
    def test_binary_01_masks(self):
        m = np.zeros((512, 512), np.uint8)
        m[10:20, 10:20] = 1
        m[100:110, 100:110] = 1
        stats = inspect_masks([("a", png(m)), ("b", png(np.zeros((512, 512), np.uint8)))])
        assert stats.checked == 2
        assert stats.unique_values == [0, 1]
        assert stats.binary_encoding == "0/1"
        assert stats.empty_masks == 1
        assert stats.connected_regions_mean == 2.0
        assert stats.positive_pixel_ratio_mean == pytest.approx(200 / (512 * 512) / 2, abs=1e-6)

    def test_binary_0255_masks(self):
        m = np.zeros((512, 512), np.uint8)
        m[0:5, 0:5] = 255
        stats = inspect_masks([("a", png(m))])
        assert stats.binary_encoding == "0/255"

    def test_non_binary_masks(self):
        m = np.zeros((512, 512), np.uint8)
        m[0, 0], m[0, 1] = 1, 2
        assert inspect_masks([("a", png(m))]).binary_encoding == "non-binary"

    def test_wrong_size_and_corrupt(self):
        stats = inspect_masks([("small", png(np.zeros((256, 256), np.uint8))), ("bad", b"HATA")])
        assert stats.wrong_size == 1
        assert len(stats.errors) == 1

    def test_gray_saved_as_rgb_is_accepted(self):
        m = np.zeros((512, 512, 3), np.uint8)
        m[0:5, 0:5] = 1
        assert inspect_masks([("rgb", png(m))]).binary_encoding == "0/1"


# --------------------------------------------------------------------------- #
# Eğitim kapısı
# --------------------------------------------------------------------------- #
def ok_masks():
    return MaskStats(checked=4, unique_values=[0, 1], binary_encoding="0/1")


class TestGate:
    def test_no_target_records_blocks(self):
        status, blockers, _ = decide_gate([], 0, 0, ok_masks())
        assert status == STATUS_BLOCKED
        assert any("kanıtlanamadı" in b for b in blockers)

    def test_contradictions_block(self):
        status, blockers, _ = decide_gate([record()], 0, 1, ok_masks())
        assert status == STATUS_BLOCKED

    def test_non_binary_masks_block(self):
        bad = MaskStats(checked=1, unique_values=[0, 1, 2], binary_encoding="non-binary")
        status, blockers, _ = decide_gate([record()], 0, 0, bad)
        assert status == STATUS_BLOCKED

    def test_mask_download_errors_block(self):
        bad = MaskStats(checked=0, errors=["x: HATA"])
        assert decide_gate([record()], 0, 0, bad)[0] == STATUS_BLOCKED

    def test_small_but_valid_subset_passes_with_warnings(self):
        recs = [record(split="train")] * 53 + [record(split="val")] * 5 + [record(split="test")] * 4
        status, blockers, warnings = decide_gate(recs, 0, 0, ok_masks())
        assert status == STATUS_PASSED
        assert blockers == []
        assert any("değerlendirme örneği az" in w for w in warnings)

    def test_skipped_masks_are_a_warning_not_a_pass_of_semantics(self):
        status, _, warnings = decide_gate([record()] * 60, 0, 0, None)
        assert status == STATUS_PASSED
        assert any("--skip-masks" in w for w in warnings)


# --------------------------------------------------------------------------- #
# Uçtan uca (sahte indirici ile, ağsız)
# --------------------------------------------------------------------------- #
def fake_fetcher(history, pairs, mapping_text, mask_value=1, missing=()):
    maxar = [r for r in history if "maxar" in r["jp2_file"]]
    pleiades = [r for r in history if "maxar" not in r["jp2_file"]]
    files = {
        "preprocessing/README.md": b"readme",
        "preprocessing/history_maxar.json": json.dumps(maxar).encode(),
        "preprocessing/history_pleidas.json": json.dumps(pleiades).encode(),
        "preprocessing/history_pre_post_pairs.json": json.dumps(pairs).encode(),
        "preprocessing/filename_mapping.txt": mapping_text.encode(),
        "preprocessing/label_studio_files.txt": b"",
        "preprocessing/label_studio_annotations.csv": b"",
        "LICENSE": b"CC0 1.0 Universal\n...",
    }
    mask = np.zeros((512, 512), np.uint8)
    mask[50:80, 50:80] = mask_value
    calls = []

    def fetch(url):
        rel = url.split("/data/", 1)[1] if "/data/" in url else url
        calls.append(rel)
        if rel in missing:
            raise RuntimeError("404")
        if rel in files:
            return files[rel]
        if rel.startswith("label_studio_pre_post/") and "/label/" in rel:
            return png(mask)
        raise RuntimeError(f"beklenmeyen istek: {rel}")

    fetch.calls = calls
    return fetch


class TestRunAudit:
    def run(self, tmp_path, fetch, **kw):
        return run_audit(fetch=fetch, base_url="https://example.test/data/",
                         download_dir=tmp_path / "raw", report_path=tmp_path / "reports" / "data_audit.json",
                         **kw)

    def test_passes_and_writes_report(self, tmp_path):
        history, pairs, text = build_fixture(MALATYA_SPECS, HATAY_SPECS)
        report = self.run(tmp_path, fake_fetcher(history, pairs, text))
        assert report["status"] == STATUS_PASSED
        mal = report["malatya"]
        assert mal["pair_count"] == 4
        assert mal["by_split"] == {"train": 2, "val": 1, "test": 1}
        assert mal["post_sources"] == [MALATYA_POST]
        assert mal["gsd_m_per_pixel_from_source_metadata"] == [0.5]
        assert mal["cross_split_overlaps"] == []
        assert report["dataset"]["total_final_pairs"] == 6
        assert report["dataset"]["pairs_by_source_city"] == {
            "hatay": {"train": 1, "test": 1},
            "malatya": {"train": 2, "val": 1, "test": 1},
        }
        assert report["dataset"]["data_license_first_line"] == "CC0 1.0 Universal"
        assert report["mask_semantics"]["checked"]["binary_encoding"] == "0/1"
        assert report["training_gate"]["training_allowed"] is True

        saved = json.loads((tmp_path / "reports" / "data_audit.json").read_text(encoding="utf-8"))
        assert saved["status"] == STATUS_PASSED
        csv_text = (tmp_path / "reports" / "malatya_pairs.csv").read_text(encoding="utf-8")
        assert csv_text.splitlines()[0].startswith("split,final_name")
        assert len(csv_text.strip().splitlines()) == 5
        for rel in audit.PREPROCESSING_FILES:
            assert (tmp_path / "raw" / rel).is_file()
            assert len(saved["dataset"]["metadata_files"][rel]["sha256"]) == 64

    def test_only_malatya_masks_are_downloaded(self, tmp_path):
        history, pairs, text = build_fixture(MALATYA_SPECS, HATAY_SPECS)
        fetch = fake_fetcher(history, pairs, text)
        self.run(tmp_path, fetch)
        mask_calls = [c for c in fetch.calls if "/label/" in c]
        assert len(mask_calls) == 4
        assert all(c.split("/")[-1] in {"train_14.png", "train_28.png", "val_28.png", "test_1.png"}
                   for c in mask_calls)

    def test_no_malatya_is_dataset_blocked(self, tmp_path):
        history, pairs, text = build_fixture([], HATAY_SPECS)
        report = self.run(tmp_path, fake_fetcher(history, pairs, text))
        assert report["status"] == STATUS_BLOCKED
        assert report["malatya"]["pair_count"] == 0
        assert report["training_gate"]["training_allowed"] is False
        assert (tmp_path / "reports" / "data_audit.json").is_file()  # rapor yine yazılır

    def test_non_binary_masks_block(self, tmp_path):
        history, pairs, text = build_fixture(MALATYA_SPECS, HATAY_SPECS)
        report = self.run(tmp_path, fake_fetcher(history, pairs, text, mask_value=7))
        # 0/7 -> ikili ama 0/1 veya 0/255 değil -> engel
        assert report["status"] == STATUS_BLOCKED

    def test_mask_download_failure_blocks(self, tmp_path):
        history, pairs, text = build_fixture(MALATYA_SPECS, HATAY_SPECS)
        fetch = fake_fetcher(history, pairs, text, missing={"label_studio_pre_post/test/label/test_1.png"})
        report = self.run(tmp_path, fetch)
        assert report["status"] == STATUS_BLOCKED

    def test_skip_masks(self, tmp_path):
        history, pairs, text = build_fixture(MALATYA_SPECS, HATAY_SPECS)
        fetch = fake_fetcher(history, pairs, text)
        report = self.run(tmp_path, fetch, check_masks=False)
        assert report["mask_semantics"]["checked"] is None
        assert not any("/label/" in c for c in fetch.calls)

    def test_main_exit_code(self, tmp_path, monkeypatch):
        history, pairs, text = build_fixture([], HATAY_SPECS)
        monkeypatch.setattr(audit, "http_fetch", fake_fetcher(history, pairs, text))
        monkeypatch.setattr(audit.run_audit, "__defaults__",
                            (audit.http_fetch, *audit.run_audit.__defaults__[1:]))
        code = audit.main(["--download-dir", str(tmp_path / "raw"),
                           "--report", str(tmp_path / "r.json"), "--skip-masks"])
        assert code == 2  # DATASET_BLOCKED


# --------------------------------------------------------------------------- #
# Gerçek KATE-CD (ağ gerekir): EV_NETWORK_TESTS=1 pytest -m network
# --------------------------------------------------------------------------- #
@pytest.mark.network
@pytest.mark.skipif(os.environ.get("EV_NETWORK_TESTS") != "1",
                    reason="ağ testi: EV_NETWORK_TESTS=1 ile çalıştırın")
def test_real_kate_cd_malatya_subset(tmp_path):
    report = run_audit(download_dir=tmp_path / "raw", report_path=tmp_path / "data_audit.json",
                       check_masks=True)
    mal = report["malatya"]
    assert report["dataset"]["total_final_pairs"] == 486
    assert report["dataset"]["unresolved_pairs"] == 0
    assert mal["pair_count"] == 62
    assert mal["by_split"] == {"train": 53, "val": 5, "test": 4}
    assert mal["gsd_m_per_pixel_from_source_metadata"] == [0.5]
    assert mal["contradictions"] == []
    assert report["mask_semantics"]["checked"]["binary_encoding"] in ("0/1", "0/255")
    assert report["status"] == STATUS_PASSED
