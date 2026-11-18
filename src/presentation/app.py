"""Streamlit arayüzü (şartname bölüm 7).

Bu modül yalnızca Application katmanını (use case, DTO, port) bilir.
Somut adapter'lar main.py'deki composition root'ta kurulur ve AppContext
olarak buraya verilir.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import streamlit as st

from src.application.dtos import AnalysisOutputDTO
from src.domain.exceptions import EarthquakeVisionBaseError
from src.presentation import components as ui
from src.presentation.context import AppContext

logger = logging.getLogger(__name__)

PAGE_TITLE = "Earthquake Vision — Malatya Damage & Area Analysis"
RESULT_KEY = "analysis_output"
ERROR_KEY = "analysis_error"


# --------------------------------------------------------------------------- #
# Streamlit sürüm uyumluluğu
# --------------------------------------------------------------------------- #
def _stretch(render: Callable, *args, **kwargs):
    """Öğeyi kapsayıcı genişliğinde çizer.

    Yeni Streamlit sürümleri width="stretch", eskiler use_container_width=True bekler.
    """
    try:
        return render(*args, width="stretch", **kwargs)
    except TypeError:
        return render(*args, use_container_width=True, **kwargs)
    except Exception as exc:  # StreamlitAPIException: width değeri tanınmadı
        if "width" not in str(exc).lower():
            raise
        return render(*args, use_container_width=True, **kwargs)


# --------------------------------------------------------------------------- #
# Giriş noktası
# --------------------------------------------------------------------------- #
def run_app(get_context: Callable[[], AppContext]) -> None:
    st.set_page_config(page_title="Earthquake Vision", layout="wide")
    st.title(PAGE_TITLE)
    st.caption(
        "Deprem sonrası uydu görüntüsünde hasar olarak etiketlenen bölgelerin "
        "segmentasyonu ve alan ölçümü. Sonuçlar görüntü tabanlı tahminlerdir; "
        "resmî hasar tespiti veya mühendislik raporu değildir."
    )

    context = get_context()
    if context.demo_mode:
        st.warning(ui.DEMO_BANNER)
    if not context.segmentation_service.is_ready():
        _render_model_missing(context)
        return

    settings = _render_sidebar(context)
    post_file, pre_file = _render_uploaders(context)

    if st.button("Analiz et", type="primary", disabled=post_file is None):
        _run_analysis(context, post_file, pre_file, settings)
    if context.demo_mode and context.demo_sample_provider is not None:
        if st.button("Sentetik örnek sahneyle dene (DEMO)"):
            pre_png, post_png = context.demo_sample_provider()
            _run_analysis(
                context,
                _InMemoryFile("demo_sentetik_sonrasi.png", post_png),
                _InMemoryFile("demo_sentetik_oncesi.png", pre_png),
                settings,
            )

    error: Optional[str] = st.session_state.get(ERROR_KEY)
    if error:
        st.error(error)
    output: Optional[AnalysisOutputDTO] = st.session_state.get(RESULT_KEY)
    if output is not None:
        render_results(output)


class _InMemoryFile:
    """st.file_uploader dosyasının kullandığımız arayüzü (name, getvalue)."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


# --------------------------------------------------------------------------- #
# Model yok durumu (şartname 7.8)
# --------------------------------------------------------------------------- #
def _render_model_missing(context: AppContext) -> None:
    st.error("Eğitim gerekli: kullanılabilir bir hasar modeli yok.")
    reason = getattr(context.segmentation_service, "unavailable_reason", None)
    if reason:
        st.caption(f"Ayrıntı: {reason}")
    st.markdown(ui.TRAINING_INSTRUCTIONS.format(weights_path=context.weights_path))


# --------------------------------------------------------------------------- #
# Kontroller
# --------------------------------------------------------------------------- #
def _render_sidebar(context: AppContext) -> dict:
    with st.sidebar:
        st.header("Ayarlar")
        threshold = st.slider(
            "Confidence eşiği",
            min_value=0.05,
            max_value=1.0,
            value=float(context.default_confidence_threshold),
            step=0.05,
            help=ui.threshold_help(context.confidence_threshold_source,
                                   context.default_confidence_threshold),
        )
        st.subheader("Kalibrasyon")
        gsd = st.number_input(
            "GSD (metre/piksel)",
            min_value=0.0,
            value=None,
            step=0.05,
            format="%.4f",
            placeholder="Boş bırakın: yalnızca piksel",
            help="Görüntünün bilinen yer örnekleme aralığı. Boş bırakılırsa m² hesaplanmaz. "
            "GeoTIFF georeference bilgisi varsa otomatik kullanılır ve bu değer yok sayılır.",
        )
        if gsd:
            st.caption("Rapor kalibrasyonu `USER_SUPPLIED_GSD` olarak işaretlenecek.")
        st.subheader("İnceleme alanı")
        exclude_padding = st.checkbox(
            "Tam siyah kenar boşluğunu alan dışı say",
            value=False,
            help="Yalnızca görüntüde siyah dolgu olduğundan eminseniz açın. Gerçek gölgeler "
            "de tam siyah olabilir. GeoTIFF'te nodata maskesi her zaman kullanılır.",
        )
        st.divider()
        if context.demo_mode:
            st.caption("Model: **DEMO (eğitilmiş model yok)**")
            st.caption(ui.DEMO_HINT)
        else:
            st.caption(f"Model: `{context.weights_path.name}`")
        if not context.geotiff_enabled:
            st.caption("GeoTIFF desteği kapalı (rasterio kurulu değil).")
    return {"threshold": threshold, "gsd": gsd, "exclude_padding": exclude_padding}


def _render_uploaders(context: AppContext):
    types = ui.uploader_types(context.allowed_extensions)
    col_post, col_pre = st.columns(2)
    with col_post:
        post_file = st.file_uploader(
            "Deprem sonrası uydu görüntüsü (zorunlu)",
            type=types,
            help=f"Azami {context.max_image_mb:g} MB.",
        )
    with col_pre:
        pre_file = st.file_uploader(
            "Deprem öncesi görüntü (isteğe bağlı, yalnızca görsel karşılaştırma)",
            type=types,
            help="Modele verilmez. Hizalama ve tarih doğrulanmadıkça değişim haritası "
            "olarak yorumlamayın.",
        )
    return post_file, pre_file


# --------------------------------------------------------------------------- #
# Analiz
# --------------------------------------------------------------------------- #
def _run_analysis(context: AppContext, post_file, pre_file, settings: dict) -> None:
    st.session_state[RESULT_KEY] = None
    st.session_state[ERROR_KEY] = None
    request = ui.build_request(
        post_bytes=post_file.getvalue(),
        post_name=post_file.name,
        confidence_threshold=settings["threshold"],
        gsd_m_per_pixel=settings["gsd"],
        exclude_black_padding=settings["exclude_padding"],
        pre_bytes=pre_file.getvalue() if pre_file is not None else None,
        pre_name=pre_file.name if pre_file is not None else None,
    )
    with st.spinner("Analiz ediliyor..."):
        try:
            st.session_state[RESULT_KEY] = context.use_case.execute(request)
        except EarthquakeVisionBaseError as exc:
            logger.warning("Analiz başarısız: %s", exc)
            st.session_state[ERROR_KEY] = str(exc)
        except Exception:  # beklenmeyen hata: kullanıcıya genel mesaj, ayrıntı logda
            logger.exception("Beklenmeyen analiz hatası")
            st.session_state[ERROR_KEY] = (
                "Beklenmeyen bir hata oluştu. Ayrıntılar terminaldeki logda."
            )


# --------------------------------------------------------------------------- #
# Sonuçlar
# --------------------------------------------------------------------------- #
def render_results(out: AnalysisOutputDTO) -> None:
    st.divider()
    st.subheader("Sonuçlar (DEMO ONLY)" if out.demo_mode else "Sonuçlar")
    if out.demo_mode:
        st.error("DEMO ONLY: Bu sonuçlar eğitilmiş bir modelden gelmiyor; deprem hasarı değildir.")

    # Görseller (şartname 7.3)
    col_orig, col_pred = st.columns(2)
    with col_orig:
        _stretch(st.image, out.original_image, caption="Deprem sonrası görüntü")
    with col_pred:
        _stretch(
            st.image,
            out.annotated_image,
            caption="Model tahmini (kırmızı: hasar, sarı: kenara değen bölge)",
        )
    if out.pre_image is not None:
        with st.expander("Deprem öncesi görüntü (yalnızca görsel karşılaştırma)"):
            _stretch(st.image, out.pre_image)

    # Metrikler (şartname 7.4)
    cards = ui.metric_cards(out)
    for row_start in range(0, len(cards), 4):
        columns = st.columns(4)
        for column, card in zip(columns, cards[row_start:row_start + 4]):
            column.metric(card.label, card.value, help=card.help)
    for label, reason in ui.unavailable_metrics(out):
        st.caption(f"**{label}:** {reason}")
    st.caption(f"**Kalibrasyon:** {ui.calibration_label(out)}")
    st.caption(f"**İnceleme alanı:** {ui.aoi_label(out)}")
    st.caption(f"**Confidence eşiği:** {out.confidence_threshold:.2f}")

    # Uyarılar (şartname 7.7)
    near = ui.low_confidence_regions(out)
    if near:
        st.info(f"{near} bölgenin confidence değeri eşiğe çok yakın; sonuçları dikkatle yorumlayın.")
    for level, code, message in ui.classify_warnings(out.warnings):
        text = f"**{code}** — {message}" if code else message
        (st.warning if level == "warning" else st.info)(text)

    # Bölge tablosu (şartname 7.5)
    st.subheader("Bölge tablosu")
    rows = ui.region_table_rows(out)
    if rows:
        _stretch(st.dataframe, rows, hide_index=True)
    else:
        st.caption("Eşiği geçen bölge yok.")

    # İndirmeler (şartname 7.6)
    st.subheader("İndir")
    _render_downloads(out)


def _render_downloads(out: AnalysisOutputDTO) -> None:
    files = [
        ("İşaretli görüntü (.png)", out.saved_image_path, "image/png"),
        ("Hasar maskesi (.png)", out.saved_mask_path, "image/png"),
        ("Bölge metrikleri (.csv)", out.saved_metrics_path, "text/csv"),
        ("Özet ve metadata (.json)", out.saved_summary_path, "application/json"),
    ]
    columns = st.columns(len(files))
    for column, (label, path, mime) in zip(columns, files):
        path = Path(path)
        if not path.is_file():
            column.caption(f"{label}: dosya bulunamadı")
            continue
        column.download_button(
            label, data=path.read_bytes(), file_name=path.name, mime=mime,
            key=f"download_{path.name}",
        )
    st.caption(f"Dosyalar ayrıca şu klasöre kaydedildi: `{Path(out.saved_summary_path).parent}`")
