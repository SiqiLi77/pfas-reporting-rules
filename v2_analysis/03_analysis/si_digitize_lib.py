"""Helpers for reading values off the original SI V1.62 / manuscript V1.64 figure bitmaps (2,401 px wide).

Used only for plotted elements that no table reports (see si_redraw_data.py). Axis calibration uses tick marks found on
the spines; marks are located by colour. Every digitized value is checked against tabulated values drawn in the same
figure wherever such values exist.
"""
import numpy as np
from PIL import Image

INK_RGB = (32, 48, 64)


def load(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(int)


def near(im, rgb, tol=40):
    return np.abs(im - np.array(rgb)).max(axis=2) <= tol


def dark(im, thr=110):
    return im.max(axis=2) <= thr


def runs(mask1d):
    """(start, end) inclusive of True runs."""
    m = np.r_[False, mask1d, False].astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def vspines(im, x0, x1, y0, y1, min_len=150, thr=110):
    """Columns in [x0, x1) holding a vertical dark run of at least min_len px within rows [y0, y1)."""
    D = dark(im, thr)[y0:y1, x0:x1]
    out = []
    for j in range(D.shape[1]):
        best = max((b - a + 1 for a, b in runs(D[:, j])), default=0)
        if best >= min_len:
            out.append((x0 + j, best))
    return out


def hspines(im, x0, x1, y0, y1, min_len=150, thr=110):
    D = dark(im, thr)[y0:y1, x0:x1]
    out = []
    for i in range(D.shape[0]):
        best = max((b - a + 1 for a, b in runs(D[i, :])), default=0)
        if best >= min_len:
            out.append((y0 + i, best))
    return out


def centres(mask1d, offset=0):
    return [offset + (a + b) / 2 for a, b in runs(mask1d)]


def xticks(im, y_spine, x0, x1, depth=(3, 9), thr=110):
    """Centres of tick marks hanging below a horizontal spine at row y_spine."""
    D = dark(im, thr)[y_spine + depth[0]:y_spine + depth[1], x0:x1]
    return centres(D.all(axis=0), x0)


def yticks(im, x_spine, y0, y1, depth=(3, 9), thr=110):
    """Centres of tick marks left of a vertical spine at column x_spine."""
    D = dark(im, thr)[y0:y1, x_spine - depth[1]:x_spine - depth[0]]
    return centres(D.all(axis=1), y0)


class Axis:
    """Linear (or log10) map from pixel to data, fitted by least squares on tick pixels and values."""

    def __init__(self, px, vals, log=False):
        self.log = log
        v = np.log10(vals) if log else np.asarray(vals, float)
        self.a, self.b = np.polyfit(np.asarray(px, float), v, 1)
        self.resid = float(np.abs(np.polyval([self.a, self.b], px) - v).max())

    def __call__(self, p):
        v = self.a * np.asarray(p, float) + self.b
        return 10 ** v if self.log else v

    def px(self, val):
        v = np.log10(val) if self.log else val
        return (v - self.b) / self.a


def blobs(mask, min_px=20):
    """Connected components (4-neighbour) of a boolean mask: list of dicts with centroid and bounding box."""
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        ys, xs = np.where(lab[sl] == k)
        if len(xs) < min_px:
            continue
        out.append({"x": float(xs.mean() + sl[1].start), "y": float(ys.mean() + sl[0].start), "n": int(len(xs)),
                    "x0": int(sl[1].start), "x1": int(sl[1].stop - 1), "y0": int(sl[0].start), "y1": int(sl[0].stop - 1)})
    return out


def hextent(mask, y, x_lo, x_hi, band=1):
    """Leftmost and rightmost True column in rows y-band..y+band between x_lo and x_hi (error-bar ends)."""
    rows = mask[int(round(y)) - band:int(round(y)) + band + 1, x_lo:x_hi].any(axis=0)
    idx = np.where(rows)[0]
    return (x_lo + idx.min(), x_lo + idx.max()) if len(idx) else (None, None)


def vextent(mask, x, y_lo, y_hi, band=1):
    cols = mask[y_lo:y_hi, int(round(x)) - band:int(round(x)) + band + 1].any(axis=1)
    idx = np.where(cols)[0]
    return (y_lo + idx.min(), y_lo + idx.max()) if len(idx) else (None, None)


def hbar(mask, y, x_lo, x_hi, half=14, marker_min_w=9, marker_min_h=7, cap_min_h=6):
    """Marker centre and cap centres of a horizontal error bar drawn in one colour around row y.

    Columns are profiled by how many mask pixels they hold within +-half rows of y. The marker is the widest run of
    tall columns; caps are the outermost narrow runs of tall columns; when a bar has no caps the outermost mask column
    is returned instead. Returns (marker_x, lo_x, hi_x) in pixels, or None if nothing is found.
    """
    band = mask[int(round(y)) - half:int(round(y)) + half + 1, x_lo:x_hi]
    prof = band.sum(axis=0)
    tall = runs(prof >= marker_min_h)
    wide = [r for r in tall if r[1] - r[0] + 1 >= marker_min_w]
    if not wide:
        return None
    m = max(wide, key=lambda r: prof[r[0]:r[1] + 1].sum())
    xs = np.arange(m[0], m[1] + 1)
    mx = x_lo + float((xs * prof[xs]).sum() / prof[xs].sum())
    any_ = np.where(prof > 0)[0]
    caps = [r for r in runs(prof >= cap_min_h) if r[1] - r[0] + 1 < marker_min_w]
    left = [r for r in caps if r[1] < m[0]]
    right = [r for r in caps if r[0] > m[1]]
    lo = x_lo + ((left[0][0] + left[0][1]) / 2 if left else any_.min())
    hi = x_lo + ((right[-1][0] + right[-1][1]) / 2 if right else any_.max())
    return mx, lo, hi


def vbar(mask, x, y_lo, y_hi, **kw):
    """hbar for a vertical error bar around column x; returns (marker_y, lo_y, hi_y)."""
    r = hbar(mask.T, x, y_lo, y_hi, **kw)
    return r


def curve(mask, x_lo, x_hi, y_lo, y_hi):
    """Per-column centre of a thin line: returns arrays (x, y) for columns where the mask has pixels."""
    xs, ys = [], []
    for x in range(x_lo, x_hi):
        idx = np.where(mask[y_lo:y_hi, x])[0]
        if len(idx):
            xs.append(x)
            ys.append(y_lo + idx.mean())
    return np.array(xs, float), np.array(ys, float)
