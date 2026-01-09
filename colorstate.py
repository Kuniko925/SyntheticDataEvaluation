import numpy as np
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance


def hist1d(values, bins, vmin, vmax):
    h, _ = np.histogram(values, bins=bins, range=(vmin, vmax))
    h = h.astype(np.float64)
    h /= (h.sum() + 1e-12)
    return h

def hist2d(x, y, bins_x, bins_y, xmin, xmax, ymin, ymax):
    h, _, _ = np.histogram2d(x, y, bins=[bins_x, bins_y], range=[[xmin, xmax], [ymin, ymax]])
    h = h.astype(np.float64)
    h /= (h.sum() + 1e-12)
    return h

def kl_div(p, q, eps=1e-12):
    # KL(p||q) with epsilon smoothing
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(rel_entr(p, q))

def js_div(p, q, eps=1e-12):
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
    p = p / p.sum(); q = q / q.sum()
    m = 0.5*(p+q)
    return 0.5*kl_div(p, m, eps) + 0.5*kl_div(q, m, eps)

def wasserstein_from_hist(p, q):
    # 1D Wasserstein distance computed on bin indices
    # Better: use bin centers with weights
    n = len(p)
    x = np.arange(n, dtype=np.float64)
    return wasserstein_distance(x, x, u_weights=p, v_weights=q)