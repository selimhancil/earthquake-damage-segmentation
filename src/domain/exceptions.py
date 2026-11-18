"""Earthquake Vision hata hiyerarşisi.

Domain katmanı dış kütüphanelere bağımlı değildir; tüm katmanlar bu
istisnaları kullanabilir. Mesajlar kullanıcıya gösterilebileceği için Türkçedir.
"""


class EarthquakeVisionBaseError(Exception):
    """Projedeki tüm özel hataların ortak atası."""


class InvalidImageError(EarthquakeVisionBaseError):
    """Yüklenen dosya geçersiz: boş, desteklenmeyen uzantı, çok büyük veya okunamıyor."""


class DomainRuleViolationError(EarthquakeVisionBaseError):
    """Bir domain kuralı ihlal edildi (ör. negatif alan, geçersiz confidence)."""


class DimensionMismatchError(EarthquakeVisionBaseError):
    """Maske boyutu analiz görüntüsünün H×W boyutuyla eşleşmiyor."""


class DatasetMetadataMissingError(EarthquakeVisionBaseError):
    """Malatya ayrımı veya eğitim için gerekli veri seti metadata'sı eksik."""


class CalibrationUnavailableError(EarthquakeVisionBaseError):
    """Fiziksel alan (m²) açıkça istendi ancak geçerli kalibrasyon yok."""


class ModelWeightsUnavailableError(EarthquakeVisionBaseError):
    """Eğitilmiş model ağırlık dosyası (.pt) bulunamadı veya yüklenemedi."""
