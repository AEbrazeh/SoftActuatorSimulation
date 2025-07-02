# --- CLAMP PARAMETERS ---
import numpy as np

def clamp_params(l, beta, alpha, lMin, lMax, betaMin, betaMax, alphaMin, alphaMax):
    l = np.clip(l, lMin, lMax)
    beta = np.clip(beta, betaMin, betaMax)
    alpha = np.clip(alpha, alphaMin, alphaMax)
    return l, beta, alpha

def clamp_all_params(params, lMin, lMax, betaMin, betaMax, alphaMin, alphaMax):
    out = np.zeros_like(params)
    for i, (l, beta, alpha) in enumerate(params):
        out[i] = clamp_params(l, beta, alpha, lMin, lMax, betaMin, betaMax, alphaMin, alphaMax)
    return out