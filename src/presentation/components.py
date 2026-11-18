"""Arayüzün Streamlit'ten bağımsız yardımcıları.

Bu modül Streamlit import ETMEZ; böylece biçimlendirme, uyarı sınıflandırma ve
istek oluşturma mantığı arayüz açılmadan test edilebilir. Ekrana çizim
src/presentation/app.py içindedir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.application.dtos import AnalysisOutputDTO, AnalysisRequestDTO

NOT_AVAILABLE = "N/A"

# --------------------------------------------------------------------------- #
# Biçimlendirme
# --------------------------------------------------------------------------- #
def format_pixels(value: Optional[int]) -> str:
    if value is None:
        return NOT_AVAILABLE
    return f"{value:,} px".replace(",", ".")


def format_m2(value: Optional[float]) -> str:
    """m² değeri; kalibrasyon yoksa N/A. 1 km² üstü değerlerde km² de gösterilir."""
    if value is None:
        return NOT_AVAILABLE
    text = f"{value:,.2f} m²".replace(",", "X").replace(".", ",").replace("X", ".")
    if value >= 1_000_000:
        km2 = f"{value / 1_000_000:.3f}".replace(".", ",")
        text += f" ({km2} km²)"
    return text


def format_ratio(value: float) -> str:
    return f"%{value * 100:.2f}".replace(".", ",")


def format_ms(value: float) -> str:
    return f"{value:.0f} ms"


# --------------------------------------------------------------------------- #
# Metrik kartları
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetricCard:
    label: str
    value: str
    help: str


def metric_cards(out: AnalysisOutputDTO) -> List[MetricCard]:
    """Şartname 7.4: gösterilecek metrikler ve ne anlama geldikleri."""
    return [
        MetricCard(
            "Hasar bölgesi sayısı",
            str(out.counts.get("damaged_region_count", 0)),
            "Modelin işaretlediği segment sayısı. Bir segment birden fazla binayı "
            "kapsayabilir; bina sayısı DEĞİLDİR. "
            f"Kenara değen (kısmi): {out.counts.get('partial_region_count', 0)}",
        ),
        MetricCard(
            "Hasar alanı (piksel)",
            format_pixels(out.damaged_union_pixels),
            "Çakışan maskelerin birleşimi (union); her piksel bir kez sayılır.",
        ),
        MetricCard(
            "Hasar alanı (m²)",
            format_m2(out.damaged_area_m2),
            "Yalnızca doğrulanmış georeference veya kullanıcı GSD'si varsa hesaplanır.",
        ),
        MetricCard(
            "Hasar oranı",
            format_ratio(out.damaged_area_ratio),
            "Hasar union pikselleri / geçerli inceleme alanı pikselleri. "
            "Hasarlı bina oranı DEĞİLDİR.",
        ),
        MetricCard(
            "Hasar olarak işaretlenmeyen alan (piksel)",
            format_pixels(out.not_predicted_damaged_pixels),
            "İnceleme alanında modelin hasar demediği kısım. Hasarsız bina alanı DEĞİLDİR.",
        ),
        MetricCard(
            "Hasar olarak işaretlenmeyen alan (m²)",
            format_m2(out.not_predicted_damaged_area_m2),
            "Kalibrasyon yoksa N/A. Hasarsız bina alanı DEĞİLDİR.",
        ),
        MetricCard(
            "Çıkarım süresi",
            format_ms(out.inference_time_ms),
            "Yalnızca model çıkarımı; okuma ve kayıt hariç.",
        ),
    ]


def unavailable_metrics(out: AnalysisOutputDTO) -> List[Tuple[str, str]]:
    """Veri yetersizliği nedeniyle gösterilmeyen metrikler (Kural 7)."""
    items = [
        ("Hasarlı bina sayısı", "Veri yetersiz: bina bazlı etiket / footprint yok."),
    ]
    if out.undamaged_building_area_m2 is None:
        items.append(
            ("Hasarsız bina alanı", "Veri yetersiz: doğrulanmış bina footprint'i yok.")
        )
    return items


def calibration_label(out: AnalysisOutputDTO) -> str:
    source = out.calibration_source
    if source == "verified_georeference":
        return f"Doğrulanmış georeference (piksel = {out.pixel_area_m2:.4f} m²)"
    if source == "user_supplied_gsd":
        return (
            f"Kullanıcı GSD'si: {out.gsd_m_per_pixel:g} m/piksel "
            f"(piksel = {out.pixel_area_m2:.4f} m²)"
        )
    return "Kalibrasyon yok: yalnızca piksel alanları raporlanır"


AOI_LABELS = {
    "full_image": "Görüntünün tamamı (geçerli alan maskesi yok)",
    "geotiff_nodata": "GeoTIFF nodata maskesi hariç tutuldu",
    "user_black_padding_excluded": "Tam siyah kenar boşluğu hariç tutuldu (kullanıcı seçimi)",
}


def aoi_label(out: AnalysisOutputDTO) -> str:
    return AOI_LABELS.get(out.aoi_assumption, out.aoi_assumption)


# --------------------------------------------------------------------------- #
# Bölge tablosu
# --------------------------------------------------------------------------- #
def region_table_rows(out: AnalysisOutputDTO) -> List[Dict[str, Any]]:
    """Şartname 7.5: ID, confidence, piksel alanı, m² / N/A, kısmi bayrağı."""
    return [
        {
            "ID": r.id,
            "Confidence": round(r.confidence, 3),
            "Piksel alanı": r.pixel_area,
            "Alan (m²)": round(r.area_m2, 2) if r.area_m2 is not None else NOT_AVAILABLE,
            "Kenar / kısmi": "Evet" if r.is_partial else "Hayır",
        }
        for r in out.objects
    ]


# --------------------------------------------------------------------------- #
# Uyarılar
# --------------------------------------------------------------------------- #
# Uyarı kodu -> gösterim seviyesi ("warning" sarı, "info" mavi)
WARNING_LEVELS = {
    "DEMO_ONLY": "warning",
    "NO_DAMAGE_DETECTED": "warning",
    "CALIBRATION_UNAVAILABLE": "warning",
    "GEOREFERENCE_NOT_METRIC": "warning",
    "USER_GSD_IGNORED": "info",
    "PARTIAL_REGIONS": "info",
    "OVERLAPPING_REGIONS": "info",
    "EMPTY_MASK_SKIPPED": "info",
    "PRE_IMAGE_DISPLAY_ONLY": "warning",
    "AOI_FULL_IMAGE_ASSUMED": "info",
    "SINGLE_DATE_MODEL": "info",
    "LOCATION_NOT_VERIFIED": "warning",
}


def classify_warnings(warnings: List[str]) -> List[Tuple[str, str, str]]:
    """Uyarıları (seviye, kod, mesaj) üçlüsüne çevirir; önce 'warning'lar gelir."""
    parsed = []
    for raw in warnings:
        code, sep, message = raw.partition(":")
        if not sep:
            code, message = "", raw
        level = WARNING_LEVELS.get(code.strip(), "info")
        parsed.append((level, code.strip(), message.strip()))
    order = {"warning": 0, "info": 1}
    return sorted(parsed, key=lambda item: order[item[0]])


def low_confidence_regions(out: AnalysisOutputDTO, margin: float = 0.05) -> int:
    """Eşiğe çok yakın (eşik + margin altında) bölge sayısı."""
    limit = out.confidence_threshold + margin
    return sum(1 for r in out.objects if r.confidence < limit)


# --------------------------------------------------------------------------- #
# İstek oluşturma
# --------------------------------------------------------------------------- #
def build_request(
    post_bytes: bytes,
    post_name: str,
    confidence_threshold: float,
    gsd_m_per_pixel: Optional[float],
    exclude_black_padding: bool,
    pre_bytes: Optional[bytes] = None,
    pre_name: Optional[str] = None,
) -> AnalysisRequestDTO:
    """Arayüz girdilerini use case isteğine çevirir. 0 veya boş GSD -> kalibrasyon yok."""
    gsd = gsd_m_per_pixel if gsd_m_per_pixel else None
    return AnalysisRequestDTO(
        post_image_bytes=post_bytes,
        post_image_filename=post_name,
        confidence_threshold=confidence_threshold,
        user_gsd_m_per_pixel=gsd,
        exclude_black_padding=exclude_black_padding,
        pre_image_bytes=pre_bytes or None,
        pre_image_filename=pre_name if pre_bytes else None,
    )


def threshold_help(source: str, value: float) -> str:
    """Confidence kaydırıcısının açıklaması: varsayılanın nereden geldiği."""
    base = "Bu değerin altındaki tahminler sonuca girmez. "
    if source == "model_meta":
        return base + (
            f"Varsayılan ({value:.2f}), model seçilirken doğrulama kümesinde belirlenen eşiktir "
            "(test kümesine bakılmadan)."
        )
    return base + "0.70 başlangıç değeridir, doğrulanmış en iyi değer değildir."


def uploader_types(allowed_extensions) -> List[str]:
    """st.file_uploader için uzantı listesi (noktasız, sıralı)."""
    return sorted(ext.lstrip(".") for ext in allowed_extensions)


DEMO_BANNER = (
    "**DEMO ONLY:** Bu oturum eğitilmiş bir model KULLANMIYOR. İşaretlenen alanlar basit "
    "bir doku sezgiselinden gelir ve deprem hasarı değildir. Yalnızca arayüzü denemek "
    "içindir; sonuçları bilimsel, resmî veya makale amaçlı kullanmayın. Kaydedilen "
    "dosyalar `DEMO_` ön ekiyle ve filigranla işaretlenir."
)

DEMO_HINT = (
    "Demo modunu kapatmak için uygulamayı `EV_DEMO_MODE` olmadan başlatın: "
    "`streamlit run main.py`"
)

TRAINING_INSTRUCTIONS = """\
**Model ağırlıkları bulunamadı veya kullanılamıyor.**

Bu uygulama yalnızca Malatya hasar verisiyle eğitilmiş bir YOLOv8-Seg modeliyle
çalışır. Genel (COCO) bir model deprem hasarı olarak sunulmaz.

1. Veri denetimini çalıştırın: `python -m src.training.audit_dataset`
2. Denetim geçerse veriyi hazırlayın ve YOLO biçimine dönüştürün.
3. Modeli Colab'da eğitin: `python -m src.training.train`
4. Oluşan `best_model.pt` dosyasını `{weights_path}` konumuna koyun.
5. Bu sayfayı yenileyin.

Arayüzü model olmadan denemek için demo modunu kullanabilirsiniz (sonuçlar DEMO ONLY):
`EV_DEMO_MODE=1 streamlit run main.py`
"""
