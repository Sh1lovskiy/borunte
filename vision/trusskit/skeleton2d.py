from __future__ import annotations

import numpy as np
from utils.logger import Logger

LOG = Logger.get_logger("tk.skeleton")


def _binarize(img: np.ndarray) -> np.ndarray:
    """Return binary uint8 image in {0,1} with shape (H,W)."""
    if img.ndim != 2:
        raise ValueError("skeletonize expects 2D array")
    a = img.astype(np.float32)
    a = (a > 0).astype(np.uint8)
    return a


def _neighbors(pad: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return ordered neighbors p2..p9 for Zhang–Suen on inner region."""
    c = pad[1:-1, 1:-1]  # center (not used here, but clarifies shape)
    p2 = pad[:-2, 1:-1]
    p3 = pad[:-2, 2:]
    p4 = pad[1:-1, 2:]
    p5 = pad[2:, 2:]
    p6 = pad[2:, 1:-1]
    p7 = pad[2:, :-2]
    p8 = pad[1:-1, :-2]
    p9 = pad[:-2, :-2]
    return p2, p3, p4, p5, p6, p7, p8, p9


def _transitions(p2, p3, p4, p5, p6, p7, p8, p9) -> np.ndarray:
    """Number of 0→1 transitions in circular sequence."""
    seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
    t = 0
    for a, b in zip(seq[:-1], seq[1:]):
        t += ((a == 0) & (b == 1)).astype(np.uint8)
    return t


def _neighbor_count(p2, p3, p4, p5, p6, p7, p8, p9) -> np.ndarray:
    """8-neighborhood sum."""
    return (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9).astype(np.uint8)


def _zs_iter(pad: np.ndarray, step: int) -> int:
    """One Zhang–Suen sub-iteration. Returns number of deletions."""
    p2, p3, p4, p5, p6, p7, p8, p9 = _neighbors(pad)
    c = pad[1:-1, 1:-1]

    A = _transitions(p2, p3, p4, p5, p6, p7, p8, p9)
    B = _neighbor_count(p2, p3, p4, p5, p6, p7, p8, p9)

    m1 = (p2 * p4 * p6) if step == 0 else (p2 * p4 * p8)
    m2 = (p4 * p6 * p8) if step == 0 else (p2 * p6 * p8)

    cond = (c == 1) & (B >= 2) & (B <= 6) & (A == 1) & (m1 == 0) & (m2 == 0)
    removed = int(cond.sum())
    c[cond] = 0
    return removed


def _zhang_suen(bin_img: np.ndarray, max_iters: int) -> np.ndarray:
    """Full Zhang–Suen thinning on binary image in {0,1}."""
    pad = np.pad(bin_img, 1, mode="constant")
    total = 0
    for i in range(max_iters):
        r1 = _zs_iter(pad, step=0)
        r2 = _zs_iter(pad, step=1)
        total += r1 + r2
        if r1 + r2 == 0:
            LOG.info(f"ZS converged in {i+1} iters, removed={total}")
            break
    return pad[1:-1, 1:-1]


def _prune_endpoints(skel: np.ndarray, iters: int) -> np.ndarray:
    """Iteratively remove endpoints (degree == 1) to prune spurs."""
    if iters <= 0:
        return skel
    img = skel.copy()
    for _ in range(iters):
        pad = np.pad(img, 1, "constant")
        p2, p3, p4, p5, p6, p7, p8, p9 = _neighbors(pad)
        deg = _neighbor_count(p2, p3, p4, p5, p6, p7, p8, p9)
        ep = (img == 1) & (deg == 1)
        if not ep.any():
            break
        img[ep] = 0
    return img


def skeletonize(
    img: np.ndarray, max_iters: int = 500, prune_iters: int = 0
) -> np.ndarray:
    """
    Skeletonize a binary mask using Zhang–Suen thinning.

    Args:
        img: 2D array (uint8/bool/float). Non-zero treated as foreground.
        max_iters: safety cap for ZS iterations.
        prune_iters: how many endpoint-pruning passes to run after ZS.

    Returns:
        2D uint8 mask in {0,1} with thin 1-pixel skeleton.
    """
    bin_img = _binarize(img)
    sk = _zhang_suen(bin_img, max_iters=max_iters)
    sk = _prune_endpoints(sk, iters=prune_iters)
    return sk.astype(np.uint8)
