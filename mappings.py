import cv2
from dataclasses import dataclass

stat_metrics = ["mean", "std", "skew", "kurtosis", "entropy"]


@dataclass(frozen=True)
class ColorSpec:
    cvt_code: int
    ch: str
    idx: int
    hist_range: tuple[int, int]

CS = {
    "LAB": ColorSpec(cv2.COLOR_BGR2LAB, "L", 0, (0, 256)),
    "HSV": ColorSpec(cv2.COLOR_BGR2HSV,   "V", 2, (0, 256)),
    "YCRCB": ColorSpec(cv2.COLOR_BGR2YCrCb, "Y", 0, (0, 256)),
}