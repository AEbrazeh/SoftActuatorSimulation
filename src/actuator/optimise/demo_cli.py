"""
Command-line demo:
  python -m actuator.optimise.demo_cli
"""
import numpy as np, matplotlib.pyplot as plt
from .fitter import fit
from .synthetic import synthetic_s_curve_3d

def main():
    np.random.seed(42)
    target = synthetic_s_curve_3d(120)

    out = fit(target, n_segments=6, seed=42, verbose=True)
    print("Best cost:", out['result'].fun)

    pts = out['points']
    fig = plt.figure(figsize=(7,4))
    ax  = fig.add_subplot(121, projection='3d')
    ax.plot(target[:,0], target[:,1], target[:,2], '-', label='target')
    ax.plot(pts[:,0], pts[:,1], pts[:,2], 'r--', label='fit')
    ax.legend(); ax.set_title("Arc-spline fit")

    ax2 = fig.add_subplot(122)
    M = min(len(pts), len(target))
    ax2.plot(np.linalg.norm(pts[:M] - target[:M], axis=1))
    ax2.set_title("Pointwise error")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
