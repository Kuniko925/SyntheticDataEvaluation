import os
from pathlib import Path
import cv2
import numpy as np
import scipy.stats as stats

def load_color_images(dir_path, colorspace="LAB"):
    cs = colorspace.upper()
    if cs not in _CS_MAP:
        raise ValueError(f"colorspace must be one of {tuple(_CS_MAP)}, got {colorspace}")
    cvt, keys = _CS_MAP[cs]

    data = []
    cls = Path(dir_path).name
    for filename in os.listdir(dir_path):
        img_path = os.path.join(dir_path, filename)
        img = cv2.imread(img_path)
        cimg = cv2.cvtColor(img, cvt)
        data.append({
            "class": cls,
            keys[0]: cimg[:, :, 0].astype(np.float32),
            keys[1]: cimg[:, :, 1].astype(np.float32),
            keys[2]: cimg[:, :, 2].astype(np.float32),
        })
    return data

def concatenate_channels(samples, ch):
    return np.concatenate([s[ch].ravel() for s in samples])

# This is for L, S, V and Y channels.
def l_stats(samples, ch="L"):
    L_all = concatenate_channels(samples, ch=ch)
    return {
        "mean": L_all.mean(),
        "std": L_all.std(),
        "skew": ((L_all - L_all.mean())**3).mean() / (L_all.std()**3 + 1e-8)
    }

_CS_MAP = {
    "LAB":   (cv2.COLOR_BGR2LAB,   ("L", "A", "B")),
    "HSV":   (cv2.COLOR_BGR2HSV,   ("H", "S", "V")),
    "YCRCB": (cv2.COLOR_BGR2YCrCb, ("Y", "CR", "CB")),
}

def get_channel_class_stats(dir_path, color_space, ch):
    data = load_color_images(dir_path, colorspace=color_space)
    c_data = np.concatenate([d[ch].ravel() for d in data])
    bins = np.linspace(0, 1, 256)
    hist_range = (bins[0], bins[-1])
    hist, _ = np.histogram(c_data, bins=bins, range=hist_range)
    hist = hist.astype(float)
    pdf = hist / hist.sum() if hist.sum() else hist
    return {
        "mean": c_data.mean(),
        "std": c_data.std(),
        "skew": stats.skew(c_data),
        "kurtosis": stats.kurtosis(c_data),
        "entropy": stats.entropy(pdf),
    }


def get_channel_stats(dir_path, color_space='LAB', ch='L'):
    cs = color_space.upper()
    if cs not in _CS_MAP:
        raise ValueError(f"colorspace must be one of {tuple(_CS_MAP)}, got {color_space}")
    cvt, keys = _CS_MAP[cs]

    ch_u = ch.upper()
    if ch_u not in keys:
        raise ValueError(f"ch must be one of {keys}, got {ch}")

    cidx = keys.index(ch_u)

    data = []
    for filename in os.listdir(dir_path):
        img_path = os.path.join(dir_path, filename)
        img_bgr = cv2.imread(img_path)
        img_cvt = cv2.cvtColor(img_bgr, cvt)
        C = img_cvt[:, :, cidx]
        pixels = C.ravel()

        hist = np.bincount(pixels, minlength=256).astype(float)
        pdf = hist / hist.sum()
        data.append({
                "mean": pixels.mean(),
                "std": pixels.std(),
                "skew": stats.skew(pixels),
                "entropy": stats.entropy(pdf),
            })
    return data

# This is for a, b, Cb and Cr channels.
def ab_stats(samples, ch1="a", ch2="b"):
    a_all = concatenate_channels(samples, ch=ch1)
    b_all = concatenate_channels(samples, ch=ch2)
    return {
        f"var_{ch1}": a_all.var(),
        f"var_{ch2}": b_all.var(),
        f"cov_{ch1+ch2}": np.cov(a_all, b_all)[0,1]
    }

