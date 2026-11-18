"""Domain nesneleri ve iş kuralları.

Bu modül yalnızca Python standart kütüphanesini kullanır. NumPy, OpenCV,
Ultralytics veya Streamlit import ETMEZ. Piksel sayımı gibi dizi işlemleri
Infrastructure katmanında yapılır; domain yalnızca sayılarla ve kurallarla çalışır.

Uygulanan kurallar (şartname 3.2):
    1  Confidence eşiği       -> MeasurementSettings, AnalysisResult.add_region
    2  Pozitif değerler        -> DamageRegion, Calibration
    4  Kenar (partial) bölge   -> DamageRegion.is_touching_edge, AnalysisResult
    5  Union alanı             -> AnalysisResult.set_pixel_counts
    6  Kalibrasyon kaynağı     -> Calibration
    7  Sınıf semantiği         -> ObjectClass (UNDAMAGED_BUILDING yok)
    8  AOI paydası             -> AoiAssumption, AnalysisResult.damaged_area_ratio
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import List, Optional, Tuple

from src.domain.exceptions import DomainRuleViolationError

BoundingBox = Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max), piksel, dahil


# --------------------------------------------------------------------------- #
# Enum'lar
# --------------------------------------------------------------------------- #
class ObjectClass(Enum):
    """Modelin ürettiği sınıflar.

    Yalnızca DAMAGED_REGION vardır. KATE-CD maskesinde 0 değeri "hasarsız bina"
    anlamına gelmez; bu yüzden UNDAMAGED_BUILDING sınıfı, doğrulanmış harici
    bina etiketleri sağlanmadan EKLENMEZ (Kural 7).
    """

    DAMAGED_REGION = 0

    @property
    def label(self) -> str:
        return self.name.lower()


class CalibrationSource(Enum):
    """Fiziksel ölçeğin nereden geldiği (Kural 6)."""

    UNKNOWN = "unknown"
    USER_SUPPLIED_GSD = "user_supplied_gsd"
    VERIFIED_GEOREFERENCE = "verified_georeference"


class AoiAssumption(Enum):
    """Oran paydasındaki geçerli alanın nasıl belirlendiği (Kural 8)."""

    FULL_IMAGE = "full_image"
    GEOTIFF_NODATA = "geotiff_nodata"
    USER_BLACK_PADDING_EXCLUDED = "user_black_padding_excluded"


class AreaSemantic(Enum):
    """Raporlanan alanların anlamı. Arayüz ve çıktılarda bu adlar kullanılır."""

    DAMAGED_REGION = "damaged_region"
    NOT_PREDICTED_DAMAGED = "not_predicted_damaged"  # hasarsız bina DEĞİL


# --------------------------------------------------------------------------- #
# Value object'ler
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Calibration:
    """Tek fiziksel ölçek kaynağı: piksel başına m² (pixel_area_m2).

    - USER_SUPPLIED_GSD: pixel_area_m2 = gsd²
    - VERIFIED_GEOREFERENCE: pixel_area_m2 = |a·e − b·d| (projected, metrik CRS)
    - UNKNOWN: pixel_area_m2 = None, m²/km² üretilmez.
    """

    source: CalibrationSource = CalibrationSource.UNKNOWN
    pixel_area_m2: Optional[float] = None
    gsd_m_per_pixel: Optional[float] = None  # yalnızca USER_SUPPLIED_GSD için bilgi amaçlı

    def __post_init__(self) -> None:
        if self.source == CalibrationSource.UNKNOWN:
            if self.pixel_area_m2 is not None or self.gsd_m_per_pixel is not None:
                raise DomainRuleViolationError(
                    "Kalibrasyon bilinmiyorsa fiziksel ölçek verilemez"
                )
            return
        if self.pixel_area_m2 is None or self.pixel_area_m2 <= 0:
            raise DomainRuleViolationError("Piksel başına alan (m²) pozitif olmalıdır")
        if self.source == CalibrationSource.USER_SUPPLIED_GSD:
            if self.gsd_m_per_pixel is None or self.gsd_m_per_pixel <= 0:
                raise DomainRuleViolationError("GSD pozitif olmalıdır")
        elif self.gsd_m_per_pixel is not None:
            raise DomainRuleViolationError(
                "Georeference kalibrasyonunda GSD alanı kullanılmaz; pixel_area_m2 kullanılır"
            )

    @classmethod
    def unknown(cls) -> "Calibration":
        return cls()

    @classmethod
    def from_user_gsd(cls, gsd_m_per_pixel: float) -> "Calibration":
        if gsd_m_per_pixel is None or gsd_m_per_pixel <= 0:
            raise DomainRuleViolationError("GSD pozitif olmalıdır")
        return cls(
            source=CalibrationSource.USER_SUPPLIED_GSD,
            pixel_area_m2=gsd_m_per_pixel**2,
            gsd_m_per_pixel=gsd_m_per_pixel,
        )

    @classmethod
    def from_affine(
        cls, a: float, b: float, d: float, e: float, crs_is_projected_metric: bool
    ) -> "Calibration":
        """GeoTIFF affine katsayılarından piksel alanı.

        Affine: x = a·col + b·row + c ; y = d·col + e·row + f
        Dönüklük, shear ve kare olmayan pikselleri doğru ele alır.
        """
        if not crs_is_projected_metric:
            raise DomainRuleViolationError(
                "Coğrafi (derece tabanlı) CRS ile m² hesaplanamaz; metrik projected CRS gerekir"
            )
        area = abs(a * e - b * d)
        if area <= 0:
            raise DomainRuleViolationError("Affine dönüşümü geçersiz (determinant sıfır)")
        return cls(source=CalibrationSource.VERIFIED_GEOREFERENCE, pixel_area_m2=area)

    @property
    def is_available(self) -> bool:
        return self.source != CalibrationSource.UNKNOWN

    def to_m2(self, pixel_count: int) -> Optional[float]:
        """Piksel sayısını m²'ye çevirir; kalibrasyon yoksa None."""
        if pixel_count < 0:
            raise DomainRuleViolationError("Piksel sayısı negatif olamaz")
        if self.pixel_area_m2 is None:
            return None
        return pixel_count * self.pixel_area_m2


@dataclass(frozen=True)
class MeasurementSettings:
    """Bir analizin ayarları. Sonuç metadata'sına aynen yazılır."""

    calibration: Calibration = field(default_factory=Calibration)
    confidence_threshold: float = 0.70

    def __post_init__(self) -> None:
        if not 0 < self.confidence_threshold <= 1:
            raise DomainRuleViolationError("Confidence eşiği (0, 1] aralığında olmalıdır")

    def accepts_confidence(self, confidence: float) -> bool:
        """Kural 1: eşik altı bölgeler aggregate'e girmez."""
        _validate_confidence(confidence)
        return confidence >= self.confidence_threshold


# --------------------------------------------------------------------------- #
# Entity
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DamageRegion:
    """Modelin hasar olarak işaretlediği tek bir segment.

    Bir segment birden fazla binayı ya da bina dışı bir alanı kapsayabilir;
    bu yüzden "bina" değil "bölge" olarak adlandırılır. pixel_area, bounding box
    alanı DEĞİL, tam çözünürlüklü maskedeki piksel sayısıdır.
    """

    id: int
    object_class: ObjectClass
    confidence: float
    bounding_box: BoundingBox
    pixel_area: int
    is_partial: bool = False

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)
        if not isinstance(self.pixel_area, int) or self.pixel_area <= 0:
            raise DomainRuleViolationError("Bölge piksel alanı pozitif tam sayı olmalıdır")
        if len(self.bounding_box) != 4:
            raise DomainRuleViolationError("Bounding box 4 değer içermelidir")
        x_min, y_min, x_max, y_max = self.bounding_box
        if x_min < 0 or y_min < 0 or x_max < x_min or y_max < y_min:
            raise DomainRuleViolationError("Bounding box koordinatları geçersiz")

    def is_touching_edge(self, image_width: int, image_height: int) -> bool:
        """Kural 4: görüntü sınırına değen bölge kısmi (partial) kabul edilir."""
        x_min, y_min, x_max, y_max = self.bounding_box
        return (
            x_min <= 0
            or y_min <= 0
            or x_max >= image_width - 1
            or y_max >= image_height - 1
        )

    def get_physical_area(self, calibration: Calibration) -> Optional[float]:
        """Görüntü içindeki gözlenen alanın m² karşılığı; kalibrasyon yoksa None."""
        return calibration.to_m2(self.pixel_area)


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
@dataclass
class AnalysisResult:
    """Tek bir görüntü analizinin aggregate kökü.

    Kullanım sırası:
        1. add_region(...) ile eşiği geçen bölgeler eklenir.
        2. set_pixel_counts(...) ile Infrastructure'ın hesapladığı union ve
           geçerli AOI piksel sayıları verilir.
        3. Alan ve oran metotları çağrılır.
    """

    image_width: int
    image_height: int
    settings: MeasurementSettings
    aoi_assumption: AoiAssumption = AoiAssumption.FULL_IMAGE
    regions: List[DamageRegion] = field(default_factory=list)
    _damaged_union_pixels: Optional[int] = field(default=None, repr=False)
    _valid_aoi_pixels: Optional[int] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise DomainRuleViolationError("Görüntü boyutları pozitif olmalıdır")

    # ---- bölge yönetimi ------------------------------------------------- #
    def add_region(
        self,
        region_id: int,
        object_class: ObjectClass,
        confidence: float,
        bounding_box: BoundingBox,
        pixel_area: int,
    ) -> Optional[DamageRegion]:
        """Eşiği geçen bölgeyi ekler ve döndürür; eşik altındaysa None döner."""
        if self._damaged_union_pixels is not None:
            raise DomainRuleViolationError(
                "Piksel sayıları belirlendikten sonra yeni bölge eklenemez"
            )
        if not self.settings.accepts_confidence(confidence):
            return None
        if any(r.id == region_id for r in self.regions):
            raise DomainRuleViolationError(f"Bölge kimliği tekrar ediyor: {region_id}")
        x_min, y_min, x_max, y_max = bounding_box
        if x_max >= self.image_width or y_max >= self.image_height:
            raise DomainRuleViolationError("Bounding box görüntü sınırlarının dışında")

        region = DamageRegion(region_id, object_class, confidence, bounding_box, pixel_area)
        region = replace(
            region, is_partial=region.is_touching_edge(self.image_width, self.image_height)
        )
        self.regions.append(region)
        return region

    @property
    def complete_regions(self) -> List[DamageRegion]:
        return [r for r in self.regions if not r.is_partial]

    @property
    def partial_regions(self) -> List[DamageRegion]:
        """Kenara değen bölgeler. Ayrı raporlanır ama union alanından ÇIKARILMAZ."""
        return [r for r in self.regions if r.is_partial]

    @property
    def region_count(self) -> int:
        """Bölge (segment) sayısı. Bina sayısı DEĞİLDİR."""
        return len(self.regions)

    @property
    def sum_of_region_pixels(self) -> int:
        """Tekil maske alanlarının toplamı (çakışmalar iki kez sayılır)."""
        return sum(r.pixel_area for r in self.regions)

    # ---- piksel sayıları ------------------------------------------------ #
    def set_pixel_counts(self, damaged_union_pixels: int, valid_aoi_pixels: int) -> None:
        """Kural 5 ve 8: union ve AOI piksel sayılarını doğrulayarak kaydeder.

        damaged_union_pixels, kabul edilen bölgelerin logical_or birleşiminin
        AOI ile kesişimindeki piksel sayısıdır (Infrastructure hesaplar).
        """
        total_pixels = self.image_width * self.image_height
        if valid_aoi_pixels <= 0:
            raise DomainRuleViolationError("Geçerli AOI piksel sayısı pozitif olmalıdır")
        if valid_aoi_pixels > total_pixels:
            raise DomainRuleViolationError("Geçerli AOI görüntüden büyük olamaz")
        if damaged_union_pixels < 0:
            raise DomainRuleViolationError("Hasar union piksel sayısı negatif olamaz")
        if damaged_union_pixels > valid_aoi_pixels:
            raise DomainRuleViolationError("Hasar union alanı geçerli AOI'den büyük olamaz")
        if damaged_union_pixels > self.sum_of_region_pixels:
            raise DomainRuleViolationError(
                "Union alanı tekil bölge alanlarının toplamından büyük olamaz"
            )
        self._damaged_union_pixels = damaged_union_pixels
        self._valid_aoi_pixels = valid_aoi_pixels

    @property
    def damaged_union_pixels(self) -> int:
        self._require_counts()
        return self._damaged_union_pixels  # type: ignore[return-value]

    @property
    def valid_aoi_pixels(self) -> int:
        self._require_counts()
        return self._valid_aoi_pixels  # type: ignore[return-value]

    @property
    def not_predicted_damaged_pixels(self) -> int:
        """AOI içinde modelin hasar olarak işaretlemediği alan. Hasarsız bina DEĞİL."""
        return self.valid_aoi_pixels - self.damaged_union_pixels

    @property
    def overlap_pixels(self) -> int:
        """Tekil alanlar toplamı ile union arasındaki fark (çakışma + AOI dışı kısım)."""
        return self.sum_of_region_pixels - self.damaged_union_pixels

    # ---- alan ve oranlar ------------------------------------------------ #
    def damaged_area_ratio(self) -> float:
        """union_damaged_pixels / valid_AOI_pixels. Hasarlı bina oranı DEĞİLDİR."""
        return self.damaged_union_pixels / self.valid_aoi_pixels

    def calculate_union_damaged_area(self) -> Optional[float]:
        return self.settings.calibration.to_m2(self.damaged_union_pixels)

    def calculate_not_predicted_damaged_area(self) -> Optional[float]:
        return self.settings.calibration.to_m2(self.not_predicted_damaged_pixels)

    def calculate_average_region_area_pixels(self) -> Optional[float]:
        """Bölgelerin ortalama maske alanı (piksel); bölge yoksa None.

        Tekil maskeler üzerinden hesaplanır; çakışan bölgeler varsa ortalama
        union alanı / bölge sayısı ile aynı olmayabilir.
        """
        if not self.regions:
            return None
        return self.sum_of_region_pixels / len(self.regions)

    def calculate_average_region_area_m2(self) -> Optional[float]:
        avg = self.calculate_average_region_area_pixels()
        if avg is None or self.settings.calibration.pixel_area_m2 is None:
            return None
        return avg * self.settings.calibration.pixel_area_m2

    @property
    def undamaged_building_area_m2(self) -> Optional[float]:
        """Doğrulanmış bina footprint'i ve sınıf etiketi olmadan hesaplanamaz.

        MVP'de bu veri yoktur; değer her zaman None'dır (Kural 7).
        """
        return None

    @property
    def has_detections(self) -> bool:
        return bool(self.regions)

    # ---- yardımcılar ---------------------------------------------------- #
    def _require_counts(self) -> None:
        if self._damaged_union_pixels is None or self._valid_aoi_pixels is None:
            raise DomainRuleViolationError(
                "Önce set_pixel_counts ile union ve AOI piksel sayıları belirlenmelidir"
            )


def _validate_confidence(confidence: float) -> None:
    if not 0 <= confidence <= 1:
        raise DomainRuleViolationError("Confidence [0, 1] aralığında olmalıdır")
