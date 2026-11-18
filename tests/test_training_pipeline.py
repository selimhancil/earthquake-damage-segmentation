"""Eğitim akışı testleri: denetim -> hazırlama -> dönüştürme -> eğitim -> değerlendirme.

Ağ ve gerçek Ultralytics kullanılmaz: indirici ve YOLO sahtedir. Amaç güvenlik
kapılarının, dosya düzeninin ve metrik hesaplarının doğruluğudur.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.training import convert_to_yolo, evaluate, prepare_malatya, train
from src.training.audit_dataset import run_audit
from src.training.evaluate import (
    PixelCounts,
    bootstrap_ci,
    dice_from,
    iou_from,
    pixel_counts,
    precision_recall_from,
    union_mask,
)
from src.training.prepare_malatya import PreparationError, select_pairs, SelectedPair
from tests.fakes import FakeBoxes, FakeMasks, FakeResult
from tests.test_dataset_conversion import (
    HATAY_SPECS,
    MALATYA_SPECS,
    build_fixture,
    fake_fetcher,
    png,
)


# --------------------------------------------------------------------------- #
# Ortak kurulum: sahte KATE-CD ile gerçek denetim
# --------------------------------------------------------------------------- #
def pair_image_fetcher(base_fetch):
    """Denetim indiricisine A/B görüntülerini de ekler (label zaten var)."""
    def fetch(url):
        rel = url.split("/data/", 1)[1]
        if rel.startswith("label_studio_pre_post/") and ("/A/" in rel or "/B/" in rel):
            img = np.full((512, 512, 3), 90 if "/A/" in rel else 160, np.uint8)
            return png(img)
        return base_fetch(url)
    return fetch


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    history, pairs, text = build_fixture(MALATYA_SPECS, HATAY_SPECS)
    fetch = pair_image_fetcher(fake_fetcher(history, pairs, text))
    report = run_audit(fetch=fetch, base_url="https://example.test/data/",
                       download_dir=Path("data/raw/kate_cd"), report_path=Path("reports/data_audit.json"))
    assert report["status"] == "PASSED"
    return {"fetch": fetch, "base": "https://example.test/data/"}


def prepare(ws, variant):
    return prepare_malatya.run_prepare(variant=variant, fetch=ws["fetch"], base_url=ws["base"], workers=2)


# --------------------------------------------------------------------------- #
# Hazırlama
# --------------------------------------------------------------------------- #
class TestPrepare:
    def test_refuses_without_audit(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(PreparationError, match="audit_dataset"):
            prepare_malatya.run_prepare()

    def test_refuses_blocked_audit(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        Path("reports").mkdir()
        Path("reports/data_audit.json").write_text(json.dumps({"status": "DATASET_BLOCKED", "blockers": ["x"]}))
        with pytest.raises(PreparationError, match="geçmedi"):
            prepare_malatya.run_prepare()

    def test_malatya_variant_uses_only_malatya(self, workspace):
        out = prepare(workspace, "malatya")
        assert out["training_cities"] == ["malatya"]
        assert out["evaluation_cities"] == ["malatya"]
        assert out["counts"] == {"train": {"malatya": 2}, "val": {"malatya": 1}, "test": {"malatya": 1}}
        assert len(out["files"]) == 4 * 3  # A/B/label
        assert Path("data/raw/kate_cd/pairs/train/B/train_14.png").is_file()
        assert Path("data/raw/kate_cd/selection_malatya.json").is_file()

    def test_kate7_variant_trains_on_all_but_evaluates_on_malatya(self, workspace):
        out = prepare(workspace, "kate7_eval_malatya")
        assert out["training_cities"] == ["hatay", "malatya"]
        assert out["evaluation_cities"] == ["malatya"]
        # Hatay'ın test örneği hiç kullanılmaz
        assert out["counts"]["test"] == {"malatya": 1}
        assert out["counts"]["train"] == {"malatya": 2, "hatay": 1}
        assert any("birden çok şehirden" in n for n in out["notes"])

    def test_existing_files_are_not_redownloaded(self, workspace):
        prepare(workspace, "malatya")
        calls = []
        def counting(url):
            calls.append(url)
            return workspace["fetch"](url)
        prepare_malatya.run_prepare(variant="malatya", fetch=counting, base_url=workspace["base"], workers=1)
        assert calls == []

    def test_select_pairs_detects_inconsistent_lists(self):
        malatya = [SelectedPair("test", "test", "test_1.png", "malatya")]
        all_pairs = [SelectedPair("test", "test", "test_1.png", "malatya"),
                     SelectedPair("test", "test", "test_2.png", "malatya")]
        with pytest.raises(PreparationError, match="uyuşmuyor"):
            select_pairs("kate7_eval_malatya", malatya, all_pairs)

    def test_unknown_variant(self):
        with pytest.raises(PreparationError):
            select_pairs("everything", [], [])


# --------------------------------------------------------------------------- #
# Dönüştürme (dosya düzeni)
# --------------------------------------------------------------------------- #
class TestConvert:
    def test_requires_selection(self, workspace):
        with pytest.raises(PreparationError, match="prepare_malatya"):
            convert_to_yolo.run_convert("malatya")

    def test_layout_labels_and_report(self, workspace):
        prepare(workspace, "malatya")
        summary = convert_to_yolo.run_convert("malatya")
        root = Path("data/processed/malatya_yolo")
        for split, n in {"train": 2, "val": 1, "test": 1}.items():
            for sub, ext in (("images", ".png"), ("labels", ".txt"), ("masks", ".png"), ("pre", ".png")):
                assert len(list((root / sub / split).glob(f"*{ext}"))) == n, (sub, split)
        label = (root / "labels/train/train_train_14.txt").read_text().strip()
        assert label.startswith("0 ")
        assert (root / "dataset.yaml").read_text().startswith("path: data/processed/malatya_yolo")
        # model girdisi deprem SONRASI görüntüdür (B = 160), öncesi pre/ altında (A = 90)
        assert cv2.imread(str(root / "images/train/train_train_14.png"))[0, 0, 0] == 160
        assert cv2.imread(str(root / "pre/train/train_train_14.png"))[0, 0, 0] == 90
        report = json.loads(Path("reports/conversion_fidelity_malatya.json").read_text(encoding="utf-8"))
        assert report["counts"] == {"train": 2, "val": 1, "test": 1}
        assert report["fidelity_iou"]["mean"] == pytest.approx(1.0)
        assert len(summary.samples) == 4

    def test_kate7_names_include_city(self, workspace):
        prepare(workspace, "kate7_eval_malatya")
        convert_to_yolo.run_convert("kate7_eval_malatya")
        names = sorted(p.name for p in Path("data/processed/kate7_eval_malatya_yolo/images/train").glob("*.png"))
        assert any(n.startswith("hatay_") for n in names)
        assert any(n.startswith("malatya_") for n in names)

    def test_rerun_replaces_old_outputs(self, workspace):
        prepare(workspace, "malatya")
        convert_to_yolo.run_convert("malatya")
        stale = Path("data/processed/malatya_yolo/images/train/stale.png")
        stale.write_bytes(b"x")
        convert_to_yolo.run_convert("malatya")
        assert not stale.exists()


# --------------------------------------------------------------------------- #
# Eğitim (sahte YOLO)
# --------------------------------------------------------------------------- #
class FakeTrainer:
    def __init__(self, save_dir):
        self.save_dir = save_dir


class FakeTrainYolo:
    calls = []

    def __init__(self, model):
        self.model = model
        self.trainer = None

    def train(self, **kwargs):
        FakeTrainYolo.calls.append(kwargs)
        save_dir = Path(kwargs["project"]) / kwargs["name"]
        (save_dir / "weights").mkdir(parents=True, exist_ok=True)
        (save_dir / "weights" / "best.pt").write_bytes(b"best-weights")
        self.trainer = FakeTrainer(save_dir)
        return type("R", (), {"save_dir": save_dir})()


class TestTrain:
    def test_refuses_without_converted_dataset(self, workspace):
        with pytest.raises(PreparationError, match="convert_to_yolo"):
            train.run_train("malatya", yolo_factory=FakeTrainYolo)

    def test_trains_with_spec_defaults_and_writes_config(self, workspace):
        prepare(workspace, "malatya")
        convert_to_yolo.run_convert("malatya")
        FakeTrainYolo.calls = []
        config = train.run_train("malatya", "A_test", device="cpu", yolo_factory=FakeTrainYolo)
        kwargs = FakeTrainYolo.calls[0]
        assert (kwargs["epochs"], kwargs["imgsz"], kwargs["batch"], kwargs["patience"], kwargs["seed"]) == \
            (30, 512, 8, 10, 42)
        assert kwargs["deterministic"] is True
        data_yaml = Path(kwargs["data"]).read_text()
        assert "path: /" in data_yaml  # Ultralytics için mutlak yol
        assert "0: damaged_region" in data_yaml
        assert Path("weights/A_test_best.pt").read_bytes() == b"best-weights"
        assert not Path("weights/best_model.pt").exists()  # --export-weights olmadan uygulamaya dokunmaz
        saved = json.loads(Path("reports/experiment_config_A_test.json").read_text(encoding="utf-8"))
        assert saved["params"]["model"] == "yolov8n-seg.pt"  # önceden eğitilmiş ağırlıktan fine-tune
        assert saved["dataset"]["image_counts"] == {"train": 2, "val": 1, "test": 1}
        assert saved["audit"]["status"] == "PASSED"
        assert config["variant"] == "malatya"

    def test_export_weights_and_overrides(self, workspace):
        prepare(workspace, "malatya")
        convert_to_yolo.run_convert("malatya")
        FakeTrainYolo.calls = []
        train.run_train("malatya", "A2", device="cpu", export_weights=True, yolo_factory=FakeTrainYolo,
                        epochs=50, batch=4)
        assert FakeTrainYolo.calls[0]["epochs"] == 50 and FakeTrainYolo.calls[0]["batch"] == 4
        assert Path("weights/best_model.pt").is_file()

    def test_mismatched_labels_are_rejected(self, workspace):
        prepare(workspace, "malatya")
        convert_to_yolo.run_convert("malatya")
        next(Path("data/processed/malatya_yolo/labels/train").glob("*.txt")).unlink()
        with pytest.raises(PreparationError, match="eşleşmiyor"):
            train.run_train("malatya", yolo_factory=FakeTrainYolo)


# --------------------------------------------------------------------------- #
# Piksel metrikleri
# --------------------------------------------------------------------------- #
class TestPixelMetrics:
    def test_counts_and_metrics(self):
        gt = np.zeros((10, 10), bool)
        gt[0:4, 0:5] = True  # 20 px
        pred = np.zeros((10, 10), bool)
        pred[0:4, 2:7] = True  # 20 px, 12 px kesişim
        c = pixel_counts(pred, gt)
        assert (c.tp, c.fp, c.fn) == (12, 8, 8)
        assert iou_from(c) == pytest.approx(12 / 28)
        assert dice_from(c) == pytest.approx(24 / 40)
        assert precision_recall_from(c) == (pytest.approx(0.6), pytest.approx(0.6))

    def test_empty_both_is_undefined(self):
        c = PixelCounts(0, 0, 0)
        assert iou_from(c) is None and dice_from(c) is None

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            pixel_counts(np.zeros((2, 2), bool), np.zeros((3, 3), bool))

    def test_union_mask_respects_threshold(self):
        a = np.zeros((4, 4), bool); a[0, 0] = True
        b = np.zeros((4, 4), bool); b[3, 3] = True
        assert union_mask([a, b], [0.9, 0.3], 0.5, (4, 4)).sum() == 1
        assert union_mask([a, b], [0.9, 0.3], 0.25, (4, 4)).sum() == 2

    def test_bootstrap_is_deterministic_and_bounded(self):
        items = [PixelCounts(50, 10, 10), PixelCounts(30, 20, 5), PixelCounts(0, 5, 40), PixelCounts(70, 0, 0)]
        ci1 = bootstrap_ci(items, iou_from, samples=500)
        ci2 = bootstrap_ci(items, iou_from, samples=500)
        assert ci1 == ci2
        lo, hi = ci1
        assert 0.0 <= lo <= hi <= 1.0
        assert bootstrap_ci([], iou_from) == (None, None)


# --------------------------------------------------------------------------- #
# Değerlendirme (sahte YOLO)
# --------------------------------------------------------------------------- #
class FakeSeg:
    map50, map, mp, mr = 0.61, 0.33, 0.7, 0.55


class FakeMetrics:
    seg = FakeSeg()
    box = type("B", (), {"map50": 0.5, "map": 0.3})()


class FakeEvalYolo:
    """Her görüntüde ground truth karesinin sol yarısını 0.8, uzak bir alanı 0.3 ile tahmin eder."""

    names = {0: "damaged_region"}
    val_kwargs = None

    def __init__(self, weights):
        self.weights = weights

    def val(self, **kwargs):
        FakeEvalYolo.val_kwargs = kwargs
        return FakeMetrics()

    def predict(self, image, **kwargs):
        good = np.zeros((512, 512), np.float32); good[50:80, 50:65] = 1  # GT (50:80,50:80) içinde
        far = np.zeros((512, 512), np.float32); far[400:420, 400:420] = 1
        boxes = FakeBoxes(np.zeros((2, 4)), [0.8, 0.3], [0, 0])
        return [FakeResult(FakeMasks(np.stack([good, far])), boxes)]


class TestEvaluate:
    def setup_dataset(self, workspace):
        prepare(workspace, "malatya")
        convert_to_yolo.run_convert("malatya")
        Path("weights").mkdir(exist_ok=True)
        Path("weights/A_best.pt").write_bytes(b"w")

    def test_reports_map_at_low_conf_and_pixel_metrics_per_threshold(self, workspace):
        self.setup_dataset(workspace)
        result = evaluate.run_evaluate(Path("weights/A_best.pt"), "malatya", "test", "A",
                                       yolo_factory=FakeEvalYolo)
        assert FakeEvalYolo.val_kwargs["conf"] == 0.001  # mAP, arayüz eşiğiyle hesaplanmaz
        assert FakeEvalYolo.val_kwargs["split"] == "test"
        assert result["ultralytics_seg"]["mask_map50"] == 0.61
        assert result["n_images"] == 1
        m50 = result["pixel_metrics_by_conf_threshold"]["0.50"]
        m25 = result["pixel_metrics_by_conf_threshold"]["0.25"]
        # GT 30x30 = 900 px; conf>=0.5 tahmin 30x15 = 450 px, tamamı GT içinde
        assert m50["micro_iou"] == pytest.approx(0.5)
        assert m50["pixel_precision"] == pytest.approx(1.0)
        assert m50["area"]["ground_truth_damaged_m2"] == pytest.approx(900 * 0.25)
        assert m50["area"]["relative_area_error"] == pytest.approx(-0.5)
        # conf>=0.25 uzak yanlış pozitifi de içerir -> IoU düşer
        assert m25["micro_iou"] < m50["micro_iou"]
        assert set(result["pixel_metrics_by_conf_threshold"]) == {"0.25", "0.50", "0.70"}
        assert Path("reports/evaluation_A.json").is_file() and Path("reports/evaluation.json").is_file()
        figures = list(Path("reports/figures/A").glob("*.png"))
        assert len(figures) == 1
        assert cv2.imread(str(figures[0])).shape == (512, 512 * 4, 3)
        assert any("küçük" in c for c in result["caveats"])

    def test_rejects_model_with_wrong_classes(self, workspace):
        self.setup_dataset(workspace)

        class Coco(FakeEvalYolo):
            names = {0: "person", 1: "car"}

        with pytest.raises(PreparationError, match="sınıfları"):
            evaluate.run_evaluate(Path("weights/A_best.pt"), "malatya", yolo_factory=Coco)

    def test_missing_weights(self, workspace):
        self.setup_dataset(workspace)
        with pytest.raises(PreparationError, match="Ağırlık"):
            evaluate.run_evaluate(Path("weights/yok.pt"), "malatya", yolo_factory=FakeEvalYolo)


# --------------------------------------------------------------------------- #
# Colab not defteri
# --------------------------------------------------------------------------- #
def test_colab_notebook_is_valid_and_gated():
    nb = json.loads((Path(__file__).resolve().parents[1] / "notebooks" / "colab_train.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    source = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    order = [source.index(s) for s in ('assert audit["status"] == "PASSED"', "prepare_malatya --variant",
                                       "convert_to_yolo", "src.training.train", "src.training.evaluate")]
    assert order == sorted(order)  # denetim kapısı her zaman önce
    assert "--offline" in source  # Colab'da CodeOcean'a istek atılmaz (403)
    assert "returncode" in source  # başarısız komut hücreyi durdurur
    assert "getpass" in source and "ghp_" not in source and "github_pat_" not in source


def test_prepare_offline_mode_fails_clearly_when_files_missing(workspace):
    with pytest.raises(PreparationError, match="Çevrimdışı modda eksik dosya"):
        prepare_malatya.run_prepare(variant="malatya", fetch=prepare_malatya.offline_fetch,
                                    base_url=workspace["base"], workers=1)


def test_prepare_offline_mode_uses_local_files(workspace):
    prepare(workspace, "malatya")  # "bilgisayarda" indir
    out = prepare_malatya.run_prepare(variant="malatya", fetch=prepare_malatya.offline_fetch,
                                      base_url=workspace["base"], workers=1)
    assert out["counts"]["train"] == {"malatya": 2}
    assert len(out["metadata_verified_sha256"]) >= 7


def test_tampered_metadata_is_rejected(workspace):
    path = Path("data/raw/kate_cd/preprocessing/filename_mapping.txt")
    path.write_text(path.read_text() + "\n./train/label/x.png ./train/label/train_999.png\n")
    with pytest.raises(PreparationError, match="SHA-256"):
        prepare(workspace, "kate7_eval_malatya")


# --------------------------------------------------------------------------- #
# Eşik seçimi (doğrulama kümesi) ve uygulamaya aktarma
# --------------------------------------------------------------------------- #
from src.training.evaluate import (  # noqa: E402
    export_app_weights,
    run_select_threshold,
    select_best_threshold,
    selection_path_for,
)


class TestThresholdSelection:
    def test_picks_threshold_with_best_dice(self):
        per = {
            0.25: {"a": PixelCounts(450, 400, 450)},  # uzak yanlış pozitif dahil
            0.50: {"a": PixelCounts(450, 0, 450)},
            0.90: {"a": PixelCounts(0, 0, 900)},
        }
        t, dice = select_best_threshold(per)
        assert t == 0.50
        assert dice == pytest.approx(900 / 1350)

    def test_tie_prefers_higher_threshold(self):
        same = {"a": PixelCounts(10, 2, 3)}
        assert select_best_threshold({0.2: same, 0.4: same})[0] == 0.4

    def test_no_positive_dice_returns_none(self):
        assert select_best_threshold({0.5: {"a": PixelCounts(0, 0, 10)}}) == (None, None)

    def test_selection_uses_val_only_and_test_eval_picks_it_up(self, workspace):
        prepare(workspace, "malatya")
        convert_to_yolo.run_convert("malatya")
        Path("weights").mkdir(exist_ok=True)
        Path("weights/A_best.pt").write_bytes(b"w")

        seen_images = []

        class Recording(FakeEvalYolo):
            def predict(self, image, **kwargs):
                seen_images.append(float(image.mean()))
                return super().predict(image, **kwargs)

        sel = run_select_threshold(Path("weights/A_best.pt"), "malatya", "A", yolo_factory=Recording)
        # fake: conf 0.8 doğru, 0.3 uzak yanlış pozitif -> 0.30'un üstü, 0.80'e kadar en iyi; eşitlikte yüksek
        assert sel["selected_threshold"] == 0.80
        assert sel["selected_on_split"] == "val" and sel["n_images"] == 1
        assert len(seen_images) == 1  # yalnızca val görüntüsü tahmin edildi
        assert selection_path_for("A").is_file()

        result = evaluate.run_evaluate(Path("weights/A_best.pt"), "malatya", "test", "A",
                                       yolo_factory=FakeEvalYolo)
        assert result["selected_threshold_from_val"] == 0.80
        assert "0.80" in result["pixel_metrics_by_conf_threshold"]

    def test_val_evaluation_does_not_overwrite_test_report(self, workspace):
        prepare(workspace, "malatya")
        convert_to_yolo.run_convert("malatya")
        Path("weights").mkdir(exist_ok=True)
        Path("weights/A_best.pt").write_bytes(b"w")
        evaluate.run_evaluate(Path("weights/A_best.pt"), "malatya", "test", "A", yolo_factory=FakeEvalYolo)
        evaluate.run_evaluate(Path("weights/A_best.pt"), "malatya", "val", "A", yolo_factory=FakeEvalYolo)
        assert json.loads(Path("reports/evaluation_A.json").read_text())["split"] == "test"
        assert json.loads(Path("reports/evaluation_A_val.json").read_text())["split"] == "val"
        assert json.loads(Path("reports/evaluation.json").read_text())["split"] == "test"

    def test_export_writes_weights_and_meta(self, tmp_path):
        src = tmp_path / "B_best.pt"
        src.write_bytes(b"model-bytes")
        selection = {"experiment": "B", "variant": "kate7_eval_malatya", "selected_threshold": 0.25,
                     "selected_on_split": "val", "selected_val_micro_dice": 0.7, "n_images": 5}
        evaluation = {"n_images": 4, "pixel_metrics_by_conf_threshold": {
            "0.25": {"micro_iou": 0.55, "micro_iou_ci95": [0.26, 0.71], "micro_dice_f1": 0.71}}}
        target = export_app_weights(src, selection, tmp_path / "weights", evaluation)
        assert target.read_bytes() == b"model-bytes"
        meta = json.loads((tmp_path / "weights" / "best_model.meta.json").read_text())
        assert meta["recommended_confidence_threshold"] == 0.25
        assert meta["threshold_selected_on"] == "val"
        assert meta["test_at_recommended_threshold"]["micro_iou"] == 0.55
        assert len(meta["weights_sha256"]) == 64

    def test_export_refuses_without_threshold(self, tmp_path):
        src = tmp_path / "B_best.pt"
        src.write_bytes(b"x")
        with pytest.raises(PreparationError, match="eşik"):
            export_app_weights(src, {"selected_threshold": None}, tmp_path / "w")


class TestAppThresholdFromMeta:
    def test_reads_recommended_threshold(self, tmp_path):
        import config

        meta = tmp_path / "best_model.meta.json"
        meta.write_text(json.dumps({"recommended_confidence_threshold": 0.25}))
        value, source, data = config.recommended_threshold(meta)
        assert (value, source) == (0.25, "model_meta")

    @pytest.mark.parametrize("content", [None, "not json", json.dumps({"x": 1}),
                                         json.dumps({"recommended_confidence_threshold": 1.5})])
    def test_falls_back_to_070(self, tmp_path, content):
        import config

        meta = tmp_path / "best_model.meta.json"
        if content is not None:
            meta.write_text(content)
        value, source, _ = config.recommended_threshold(meta)
        assert (value, source) == (0.70, "default")

    def test_slider_help_explains_source(self):
        from src.presentation.components import threshold_help

        assert "doğrulama kümesinde" in threshold_help("model_meta", 0.25)
        assert "0.25" in threshold_help("model_meta", 0.25)
        assert "0.70 başlangıç" in threshold_help("default", 0.70)


def test_notebook_streams_logs_and_selects_threshold_on_val():
    nb = json.loads((Path(__file__).resolve().parents[1] / "notebooks" / "colab_train.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "subprocess.Popen" in code and "for line in proc.stdout" in code  # canlı epoch çıktısı
    assert "--epochs 100" in code
    assert "--select-threshold" in code and "--export-app-weights" in code
    assert code.index("src.training.train") < code.index("--select-threshold")
