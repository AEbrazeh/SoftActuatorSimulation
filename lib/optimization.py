import numpy as np
import jax
import jax.numpy as jnp
import optax as opt

from functools import partial

from scipy.interpolate import BSpline

def normalize(v):
    return v / jnp.linalg.norm(v, axis=-1, keepdims=True)

@partial(jax.jit, static_argnums=(1,))
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

@partial(jax.jit, static_argnums=(7,))
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

@partial(jax.jit, static_argnums=(1,))
def generateArcSpline(params, res):
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

@jax.jit
def l2Distance(params_, target, lowerBounds, upperBounds):
    n = len(target)
    params = (upperBounds - lowerBounds) * jnp.tanh(params_) / 2 + (upperBounds + lowerBounds) / 2
    arcSpline = resampleUniform(generateArcSpline(params, n), n)
    return ((arcSpline - target)**2).sum(-1).mean()



#-----------------------------------------------------------------

class arcSplineOptimizer:
    def __init__(self, actuator, lr = 1e-2, gamma = 0.9):
        self.actuator = actuator
        self.params = jnp.zeros((self.actuator.M, 3))
        self.lr = lr
        self.gamma = gamma
        self.sched = opt.exponential_decay(lr, 1, gamma, staircase=True, end_value=1e-3 * lr)
        self.optimizer = opt.adam(learning_rate=self.sched)
        self.optState = self.optimizer.init(self.params)
        self.loss = jax.jit(jax.value_and_grad(l2Distance, 0))

    def optimize(self, target, numIterations = 1e3, printFreq = 100):
        loss = []
        for ii in range(int(numIterations)):
            L, dLdp = self.loss(self.params, target, self.actuator.lowerBound, self.actuator.upperBound)
            loss.append(L)
            if jnp.isnan(dLdp).any():
                raise ValueError("Gradient is NaN, check the loss function or parameters.")
            
            dp, optState = self.optimizer.update(dLdp, self.optState)
            self.optState = optState
            params_ = opt.apply_updates(self.params, dp)
            self.params = params_
            
            if ii % printFreq == printFreq - 1:
                print(f"Iteration {ii+1}: Loss = {L:.12f}")

        params_ = (self.actuator.upperBound - self.actuator.lowerBound) * jnp.tanh(self.params) / 2 + (self.actuator.upperBound + self.actuator.lowerBound) / 2
        traj = resampleUniform(generateArcSpline(params_, len(target)), len(target))
        params_ = np.array(params_)
        params_[:, -1] = np.cumsum(params_[:, -1])
        return params_, traj, jnp.array(loss)

    def reset(self):
        self.params = jnp.zeros((self.actuator.M, 3))
        self.optState = self.optimizer.init(self.params)

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
