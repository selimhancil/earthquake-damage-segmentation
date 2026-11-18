"""Testlerde paylaşılan sahte Ultralytics nesneleri.

Gerçek ultralytics.YOLO'nun kullandığımız yüzeyini taklit eder:
    model.names, model.predict(source, **kwargs) -> [Result]
    Result.masks.data, Result.boxes.{xyxy, conf, cls} -> .cpu().numpy()
"""

from typing import List, Optional

import numpy as np


class FakeTensor:
    def __init__(self, array):
        self._a = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = FakeTensor(xyxy), FakeTensor(conf), FakeTensor(cls)


class FakeMasks:
    def __init__(self, data):
        self.data = FakeTensor(data)


class FakeResult:
    def __init__(self, masks, boxes):
        self.masks, self.boxes = masks, boxes


class FakeYolo:
    """predict() verilen maskeleri ve confidence değerlerini döndürür.

    Gerçek YOLO gibi `conf` eşiğinin altındakileri de filtreler, böylece
    use case'in kendi eşik filtresiyle birlikte test edilebilir.
    """

    def __init__(self, names=None, masks: Optional[List[np.ndarray]] = None, conf=None,
                 filter_by_conf: bool = False):
        self.names = names if names is not None else {0: "damaged_region"}
        self._masks = masks
        self._conf = conf
        self._filter = filter_by_conf
        self.predict_kwargs = None
        self.received = None
        self.predict_calls = 0

    def predict(self, source, **kwargs):
        self.predict_calls += 1
        self.received = source
        self.predict_kwargs = kwargs
        if not self._masks:
            return [FakeResult(None, None)]
        masks = list(self._masks)
        conf = list(self._conf)
        if self._filter:
            keep = [i for i, c in enumerate(conf) if c >= kwargs.get("conf", 0)]
            masks = [masks[i] for i in keep]
            conf = [conf[i] for i in keep]
            if not masks:
                return [FakeResult(None, None)]
        n = len(masks)
        xyxy = np.tile([0.0, 0.0, 1.0, 1.0], (n, 1))
        return [FakeResult(FakeMasks(np.stack(masks).astype(np.float32)),
                           FakeBoxes(xyxy, conf, np.zeros(n)))]


def rect_mask(x0: int, y0: int, x1: int, y1: int, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.float32)
    m[y0:y1, x0:x1] = 1.0
    return m
