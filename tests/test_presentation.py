"""Presentation katmanı testleri.

- components.py: Streamlit gerektirmez, her ortamda çalışır.
- Arayüz (AppTest): streamlit kurulu değilse atlanır.
"""

import pytest

from src.presentation import components as ui
from tests.presentation_support import sample_output

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover
    AppTest = None

requires_streamlit = pytest.mark.skipif(AppTest is None, reason="streamlit kurulu değil")


@pytest.fixture(scope="module")
def output():
    return sample_output(gsd=0.5)


@pytest.fixture(scope="module")
def output_no_gsd():
    return sample_output(gsd=None)


# --------------------------------------------------------------------------- #
# Biçimlendirme
# --------------------------------------------------------------------------- #
class TestFormatting:
    @pytest.mark.parametrize("value, expected", [(0, "0 px"), (636, "636 px"),
                                                 (1234567, "1.234.567 px"), (None, "N/A")])
    def test_format_pixels(self, value, expected):
        assert ui.format_pixels(value) == expected

    @pytest.mark.parametrize("value, expected", [
        (None, "N/A"),
        (159.0, "159,00 m²"),
        (1234.5, "1.234,50 m²"),
        (2_500_000.0, "2.500.000,00 m² (2,500 km²)"),
    ])
    def test_format_m2(self, value, expected):
        assert ui.format_m2(value) == expected

    def test_format_ratio(self):
        assert ui.format_ratio(0.103515625) == "%10,35"
        assert ui.format_ratio(0.0) == "%0,00"

    def test_format_ms(self):
        assert ui.format_ms(12.6) == "13 ms"


# --------------------------------------------------------------------------- #
# Metrikler
# --------------------------------------------------------------------------- #
class TestMetricCards:
    def test_cards_with_calibration(self, output):
        cards = {c.label: c for c in ui.metric_cards(output)}
        assert cards["Hasar bölgesi sayısı"].value == "3"
        assert "bina sayısı DEĞİLDİR" in cards["Hasar bölgesi sayısı"].help
        assert cards["Hasar alanı (piksel)"].value == "636 px"
        assert cards["Hasar alanı (m²)"].value == "159,00 m²"
        assert "Hasarlı bina oranı DEĞİLDİR" in cards["Hasar oranı"].help
        assert "Hasarsız bina alanı DEĞİLDİR" in cards[
            "Hasar olarak işaretlenmeyen alan (piksel)"].help

    def test_cards_without_calibration_show_na(self, output_no_gsd):
        cards = {c.label: c for c in ui.metric_cards(output_no_gsd)}
        assert cards["Hasar alanı (m²)"].value == "N/A"
        assert cards["Hasar olarak işaretlenmeyen alan (m²)"].value == "N/A"
        assert cards["Hasar alanı (piksel)"].value == "636 px"

    def test_no_building_metrics_are_shown_as_values(self, output):
        labels = [c.label.lower() for c in ui.metric_cards(output)]
        assert not any("bina" in label for label in labels)

    def test_unavailable_metrics_listed(self, output):
        names = [name for name, _ in ui.unavailable_metrics(output)]
        assert names == ["Hasarlı bina sayısı", "Hasarsız bina alanı"]

    def test_calibration_labels(self, output, output_no_gsd):
        assert "Kullanıcı GSD" in ui.calibration_label(output)
        assert "0.25" in ui.calibration_label(output).replace(",", ".")
        assert "Kalibrasyon yok" in ui.calibration_label(output_no_gsd)

    def test_aoi_label(self, output):
        assert ui.aoi_label(output).startswith("Görüntünün tamamı")


# --------------------------------------------------------------------------- #
# Tablo ve uyarılar
# --------------------------------------------------------------------------- #
class TestRegionTableAndWarnings:
    def test_region_rows(self, output, output_no_gsd):
        rows = ui.region_table_rows(output)
        assert [r["ID"] for r in rows] == [1, 2, 3]
        assert rows[0] == {"ID": 1, "Confidence": 0.95, "Piksel alanı": 400,
                           "Alan (m²)": 100.0, "Kenar / kısmi": "Hayır"}
        assert rows[2]["Kenar / kısmi"] == "Evet"
        assert all(r["Alan (m²)"] == "N/A" for r in ui.region_table_rows(output_no_gsd))

    def test_classify_warnings_orders_warnings_first(self):
        parsed = ui.classify_warnings([
            "SINGLE_DATE_MODEL: tek tarih",
            "CALIBRATION_UNAVAILABLE: kalibrasyon yok",
            "serbest metin",
        ])
        assert parsed[0] == ("warning", "CALIBRATION_UNAVAILABLE", "kalibrasyon yok")
        assert ("info", "SINGLE_DATE_MODEL", "tek tarih") in parsed
        assert ("info", "", "serbest metin") in parsed

    def test_every_use_case_warning_code_has_a_level(self):
        from src.application import use_cases

        codes = {v for k, v in vars(use_cases).items() if k.startswith("W_")}
        assert codes <= set(ui.WARNING_LEVELS)

    def test_low_confidence_regions(self, output):
        # eşik 0.70, conf'lar 0.95 / 0.72 / 0.80 -> yalnızca 0.72 eşiğe yakın
        assert ui.low_confidence_regions(output) == 1


# --------------------------------------------------------------------------- #
# İstek oluşturma
# --------------------------------------------------------------------------- #
class TestBuildRequest:
    def test_basic_request(self):
        req = ui.build_request(b"x", "a.png", 0.6, 0.5, True)
        assert req.post_image_filename == "a.png"
        assert req.confidence_threshold == 0.6
        assert req.user_gsd_m_per_pixel == 0.5
        assert req.exclude_black_padding is True
        assert req.pre_image_bytes is None and req.pre_image_filename is None

    @pytest.mark.parametrize("gsd", [None, 0, 0.0])
    def test_empty_or_zero_gsd_means_no_calibration(self, gsd):
        assert ui.build_request(b"x", "a.png", 0.7, gsd, False).user_gsd_m_per_pixel is None

    def test_pre_image_passed_through(self):
        req = ui.build_request(b"x", "a.png", 0.7, None, False, b"pre", "pre.png")
        assert req.pre_image_bytes == b"pre" and req.pre_image_filename == "pre.png"

    def test_empty_pre_image_ignored(self):
        req = ui.build_request(b"x", "a.png", 0.7, None, False, b"", "pre.png")
        assert req.pre_image_bytes is None and req.pre_image_filename is None

    def test_uploader_types(self):
        assert ui.uploader_types(frozenset({".png", ".jpg", ".tif"})) == ["jpg", "png", "tif"]


# --------------------------------------------------------------------------- #
# Arayüz (Streamlit AppTest)
# --------------------------------------------------------------------------- #
def _texts(at, kind):
    return " ".join(str(e.value) for e in getattr(at, kind))


@requires_streamlit
class TestStreamlitApp:
    def test_main_shows_training_required_without_weights(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)  # yanlışlıkla çıktı yazılırsa repo kirlenmesin
        from pathlib import Path

        main_py = Path(__file__).resolve().parents[1] / "main.py"
        at = AppTest.from_file(str(main_py), default_timeout=60).run()
        assert not at.exception
        assert "Earthquake Vision" in at.title[0].value
        assert "Eğitim gerekli" in _texts(at, "error")
        assert "best_model.pt" in _texts(at, "markdown")
        assert len(at.button) == 0  # model yokken analiz düğmesi yok

    def test_ready_app_shows_controls(self):
        def script():
            from src.presentation.app import run_app
            from tests.presentation_support import build_fake_context

            run_app(build_fake_context)

        at = AppTest.from_function(script, default_timeout=30).run()
        assert not at.exception
        assert at.slider[0].value == pytest.approx(0.70)
        assert at.number_input[0].value is None  # GSD varsayılan boş
        assert at.checkbox[0].value is False  # padding dışlama varsayılan kapalı
        assert at.button[0].label == "Analiz et"
        assert at.button[0].disabled  # dosya yüklenmeden pasif

    def test_results_are_rendered_from_session_state(self):
        def script():
            from src.presentation.app import run_app
            from tests.presentation_support import build_fake_context

            run_app(build_fake_context)

        at = AppTest.from_function(script, default_timeout=30)
        at.session_state["analysis_output"] = sample_output(gsd=0.5, pre=True)
        at.run()
        assert not at.exception
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Hasar bölgesi sayısı"] == "3"
        assert metrics["Hasar alanı (piksel)"] == "636 px"
        assert metrics["Hasar alanı (m²)"] == "159,00 m²"
        warnings = _texts(at, "warning")
        assert "PRE_IMAGE_DISPLAY_ONLY" in warnings
        assert "LOCATION_NOT_VERIFIED" in warnings
        assert "eşiğe çok yakın" in _texts(at, "info")
        assert len(at.dataframe) == 1
        captions = _texts(at, "caption")
        assert "Hasarsız bina alanı" in captions and "Veri yetersiz" in captions

    def test_error_message_is_shown(self):
        def script():
            from src.presentation.app import run_app
            from tests.presentation_support import build_fake_context

            run_app(build_fake_context)

        at = AppTest.from_function(script, default_timeout=30)
        at.session_state["analysis_error"] = "Desteklenmeyen dosya uzantısı: '.gif'."
        at.run()
        assert "Desteklenmeyen dosya uzantısı" in _texts(at, "error")
