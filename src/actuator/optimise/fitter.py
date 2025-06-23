"""
fitter.py
=========

Global optimiser that fits an N-segment arc-spline
(C¹-continuous) to an arbitrary 3-D poly-line.

Public API
----------
fit(points_3d, n_segments=6, *, bounds=None,
    popsize=15, maxiter=400, seed=None, verbose=False)
    → dict {'arcs', 'points', 'result'}
"""
import numpy as np
from scipy.optimize import differential_evolution
from .. import arcSpline             # one level up inside 'actuator'

# ------------------------------------------------------------------ helpers
def _unpack(x):
    """1-D vector → list[(s, θ, φ)]"""
    return [tuple(x[i:i+3]) for i in range(0, len(x), 3)]

def _default_bounds(n):
    """Physical limits for each segment variable."""
    b = []
    for _ in range(n):
        b += [
            (0.02, 0.50),           # s : 2 – 50 cm (change if metres)
            (-np.pi/2, np.pi/2),    # θ : −90° … +90°
            (-np.pi,   np.pi)       # φ : full rotation
        ]
    return b

# -------------------------------------------------------------------- main
def fit(target, n_segments=6, *, bounds=None,
        popsize=15, maxiter=400, seed=None, verbose=False):
    """
    Parameters
    ----------
    target      : (N,3) ndarray – poly-line to approximate
    n_segments  : int           – arc segments (⇒ 3·N design vars)
    bounds      : list[(low,hi)] – per-var DE bounds (optional)
    """
    target = np.asarray(target, dtype=float)
    if bounds is None:
        bounds = _default_bounds(n_segments)

    def cost(x):
        arcs = _unpack(x)
        pts  = arcSpline.generateSpline(arcs, pointDensity=600)
        M = min(len(pts), len(target))
        return np.mean(np.linalg.norm(pts[:M] - target[:M], axis=1))

    res = differential_evolution(cost, bounds=bounds, popsize=popsize,
                                 maxiter=maxiter, polish=True,
                                 seed=seed, disp=verbose)

    best_arcs = _unpack(res.x)
    best_pts  = arcSpline.generateSpline(best_arcs, pointDensity=600)
    return {'arcs': best_arcs, 'points': best_pts, 'result': res}
