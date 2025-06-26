import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax as opt

from scipy.interpolate import BSpline
from functools import partial

lMin = 0.12
lMax = 2

betaMin = 0.0
betaMax = 2 * np.pi / 3

alphaMin = 0.0
alphaMax = 2 * np.pi

minValue = jnp.array([lMin, betaMin, alphaMin])[None, :]
maxValue = jnp.array([lMax, betaMax, alphaMax])[None, :]


def normalize(v):
    return v / jnp.linalg.norm(v, axis=-1, keepdims=True)

def CurveGenerator(nCtrl=6, order=3, nPoints=400, seed=None):
    np.random.seed(seed)
    ctrl = 2 * np.random.rand(nCtrl, 3) - 1
    ctrl[0] = [0, 0, 0]
    ctrl[1] = [1, 0, 0]
    interior = np.arange(1, nCtrl - order) / (nCtrl - order)
    knots    = np.concatenate((
        np.zeros(order+1),
        interior,
        np.ones(order+1)
    ))
    bSpline = BSpline(knots, ctrl, order)
    u = np.linspace(0, 1, nPoints)
    return jnp.asarray(bSpline(u))

def resampleUniform(points, nSamples=None):
    # 1) compute segment lengths
    diffs = points[1:] - points[:-1]               # [N-1, 3]
    dists = jnp.linalg.norm(diffs, axis=1)         # [N-1]

    # 2) cumulative arc-length
    s = jnp.concatenate([jnp.zeros(1), jnp.cumsum(dists)])  # [N]

    # 3) uniform samples along [0, total_length]
    if nSamples is None:
        nSamples = len(points)
    s_uniform = jnp.linspace(0.0, s[-1], nSamples)         # [n_samples]

    # 4) linear-interpolate each coord at those s’s
    x_u = jnp.interp(s_uniform, s, points[:,0])
    y_u = jnp.interp(s_uniform, s, points[:,1])
    z_u = jnp.interp(s_uniform, s, points[:,2])

    return jnp.stack([x_u, y_u, z_u], axis=-1)

def generateArcSegment(l, beta, alpha, p0, t0, n0, b0, nPoints=100):
    k = beta / l
    # plane normal
    e1 = jnp.cos(alpha) * n0 + jnp.sin(alpha) * b0
    e2 = normalize(t0)

    s = jnp.linspace(0, l, nPoints + 1)[:, None]
    T = e1 * jnp.sin(k * s) + e2 * jnp.cos(k * s)
    P = p0 + e1 * (1 - jnp.cos(k * s)) / k + e2 * jnp.sin(k * s) / k
    # end point & tangent
    p1 = P[-1]
    t1 = T[-1]

    # new normal & binormal for next segment
    n1 = e1 * jnp.cos(beta) - e2 * jnp.sin(beta)
    b1 = normalize(jnp.cross(t1, n1))
    return P, p1, t1, n1, b1

def generateArcSpline(params_, res):
    params = (maxValue - minValue) * jnp.tanh(params_) / 2 + (maxValue + minValue) / 2
    # init frame at origin, tangent +x, normal +y, binormal +z
    p0 = jnp.array([0.0, 0.0, 0.0])
    t0 = jnp.array([1.0, 0.0, 0.0])
    n0 = jnp.array([0.0, 1.0, 0.0])
    b0 = jnp.cross(t0, n0)

    p = [p0.reshape(1,3)]
    for l, beta, alpha in params:
        pts, p1, t1, n1, b1 = generateArcSegment(l, beta, alpha, p0, t0, n0, b0, res)
        # drop the first point to avoid duplicates
        p.append(pts[1:])
        # update for next
        p0, t0, n0, b0 = p1, t1, n1, b1
    return jnp.vstack(p)

def l2Distance(params, target):
    n = len(target)
    arcSpline = resampleUniform(generateArcSpline(params, n), n)
    return ((arcSpline - target)**2).sum(-1).mean()

def optimizationStep(params, target, loss, optimizer, optState):
        L, dLdp = loss(params, target)
        if jnp.isnan(dLdp).any():
            raise ValueError("Gradient is NaN, check your loss function or parameters.")
        dp, optState_ = optimizer.update(dLdp, optState)
        params_ = opt.apply_updates(params, dp)
        return L, params_, optState_