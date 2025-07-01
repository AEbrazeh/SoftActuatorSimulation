import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import jax.numpy as jnp
# --- PARAMETERS ---
GRID_SIZE = 100
N_EXAMPLES = 10
SEGMENT_OPTIONS = [2, 3]   # Try both for each curve
MIN_SEG = 10               # Minimum points per segment

# ---------------------------------------------------------
# SYNTHETIC 3D S-CURVE GENERATION
# ---------------------------------------------------------
def synthetic_s_curve_3d(grid_size=100):
    s = np.linspace(0, np.pi, grid_size)
    k1 = np.random.uniform(0.3, 0.5)
    k2 = -np.random.uniform(0.3, 0.5)
    frac = np.random.uniform(0.4, 0.6)
    len1 = int(frac * grid_size)
    len2 = grid_size - len1
    theta1 = k1 * np.linspace(0, 1, len1)
    phi1 = np.random.uniform(0, 2*np.pi)
    x1 = np.cumsum(np.cos(theta1) + 0.05*np.random.randn(len1))
    y1 = np.cumsum(np.sin(theta1) + 0.05*np.random.randn(len1))
    z1 = np.cumsum(np.sin(theta1 + phi1) + 0.05*np.random.randn(len1))
    theta2 = theta1[-1] + k2 * np.linspace(0, 1, len2)
    phi2 = phi1 + np.random.uniform(-0.8, 0.8)
    x2 = x1[-1] + np.cumsum(np.cos(theta2) + 0.05*np.random.randn(len2))
    y2 = y1[-1] + np.cumsum(np.sin(theta2) + 0.05*np.random.randn(len2))
    z2 = z1[-1] + np.cumsum(np.sin(theta2 + phi2) + 0.05*np.random.randn(len2))
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    z = np.concatenate([z1, z2])
    return np.column_stack([x, y, z])

# ---------------------------------------------------------
# CURVATURE ESTIMATION FOR 3D CURVE
# ---------------------------------------------------------
def estimate_curvature_3d(points, window=9):
    x, y, z = points[:,0], points[:,1], points[:,2]
    x1 = savgol_filter(x, window, 2, deriv=1)
    y1 = savgol_filter(y, window, 2, deriv=1)
    z1 = savgol_filter(z, window, 2, deriv=1)
    x2 = savgol_filter(x, window, 2, deriv=2)
    y2 = savgol_filter(y, window, 2, deriv=2)
    z2 = savgol_filter(z, window, 2, deriv=2)
    num = np.linalg.norm(np.cross(np.column_stack([x1, y1, z1]),
                                  np.column_stack([x2, y2, z2])), axis=1)
    denom = (x1**2 + y1**2 + z1**2)**1.5 + 1e-8
    curvature = num / denom
    return curvature

# ---------------------------------------------------------
# PARTITIONING THE CURVE BASED ON CURVATURE (KMeans)
# ---------------------------------------------------------
def partition_by_curvature(curvature, n_segments=2, min_seg=10):
    km = KMeans(n_clusters=n_segments, n_init=10, random_state=0)
    labels = km.fit_predict(curvature.reshape(-1, 1))
    idx = np.argsort(km.cluster_centers_.flatten())
    sorted_labels = np.zeros_like(labels)
    for new, old in enumerate(idx):
        sorted_labels[labels == old] = new
    boundaries = [0]
    for i in range(1, len(curvature)):
        if sorted_labels[i] != sorted_labels[i-1]:
            boundaries.append(i)
    boundaries.append(len(curvature)-1)
    # Remove too-small segments
    boundaries = np.array(boundaries)
    clean = [boundaries[0]]
    for i in range(1, len(boundaries)):
        if boundaries[i] - clean[-1] >= min_seg or boundaries[i] == len(curvature)-1:
            clean.append(boundaries[i])
    return np.array(clean)

# ---------------------------------------------------------
# PCA: FIT BEST PLANE FOR 3D SEGMENT
# ---------------------------------------------------------
def fit_plane_pca(points):
    mean = points.mean(axis=0)
    X = points - mean
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    normal = Vt[2]           # Plane normal (smallest variance)
    basis1 = Vt[0]           # Main direction in the plane
    basis2 = Vt[1]           # Second direction in the plane
    return mean, basis1, basis2, normal

# ---------------------------------------------------------
# PROJECT 3D POINTS TO THE LOCAL 2D PLANE
# ---------------------------------------------------------
def project_to_plane(points, mean, basis1, basis2):
    X = points - mean
    x_new = X @ basis1
    y_new = X @ basis2
    return np.column_stack([x_new, y_new])

# ---------------------------------------------------------
# LIFT 2D POINTS BACK TO 3D
# ---------------------------------------------------------
def plane_to_3d(points2d, mean, basis1, basis2):
    return mean + points2d[:,0][:,None]*basis1 + points2d[:,1][:,None]*basis2

# ---------------------------------------------------------
# 2D ARC FITTER USING ENDPOINTS AND INITIAL TANGENT
# ---------------------------------------------------------
def arc_length_2d(points):
    diff = np.diff(points, axis=0)
    ds = np.sqrt((diff ** 2).sum(axis=1))
    return np.concatenate([[0], np.cumsum(ds)])

def simulate_arc_segment_2d(p0, theta0, kappa, length, n_interp=30):
    points = [p0]
    theta = theta0
    p = np.array(p0)
    dl = length / (n_interp-1)
    for i in range(1, n_interp):
        if np.abs(kappa) < 1e-8:
            dp = dl * np.array([np.cos(theta), np.sin(theta)])
        else:
            R = 1.0 / kappa
            dtheta = kappa * dl
            center = p + R * np.array([-np.sin(theta), np.cos(theta)])
            r_vec = p - center
            rot = np.array([[np.cos(dtheta), -np.sin(dtheta)],
                            [np.sin(dtheta),  np.cos(dtheta)]])
            new_p = center + rot @ r_vec
            dp = new_p - p
        p = p + dp
        points.append(p.copy())
        theta += kappa * dl
    return np.array(points)

def fit_2d_arc(points2d):
    s = arc_length_2d(points2d)
    total_len = s[-1]
    theta0_init = np.arctan2(points2d[1,1]-points2d[0,1], points2d[1,0]-points2d[0,0])
    def residual(x):
        theta0, kappa = x
        arc_pts = simulate_arc_segment_2d(points2d[0], theta0, kappa, total_len, n_interp=len(points2d))
        return np.sum((arc_pts - points2d)**2)
    res = minimize(residual, [theta0_init, 0], bounds=[(-np.pi, np.pi), (-5,5)])
    theta0, kappa = res.x
    fit_points = simulate_arc_segment_2d(points2d[0], theta0, kappa, total_len, n_interp=len(points2d))
    return fit_points, theta0, kappa

# ---------------------------------------------------------
# MAIN: PIECEWISE 3D ARC-SPLINE FIT USING 2D LOGIC
# ---------------------------------------------------------
def fit_3d_arcspline(points, n_segments=2):
    curvature = estimate_curvature_3d(points)
    partition = partition_by_curvature(curvature, n_segments=n_segments, min_seg=MIN_SEG)
    fit_points = []
    diagnostics = []
    for i in range(len(partition)-1):
        idx0, idx1 = partition[i], partition[i+1]
        seg = points[idx0:idx1+1]
        mean, b1, b2, nrm = fit_plane_pca(seg)
        seg2d = project_to_plane(seg, mean, b1, b2)
        fit2d, theta0, kappa = fit_2d_arc(seg2d)
        fit3d = plane_to_3d(fit2d, mean, b1, b2)
        fit3d = np.array(fit3d)  # Convert JAX to NumPy
        if i > 0:
            fit3d[0] = fit_points[-1]  # snap start to previous segment
        fit_points.extend(fit3d if i == 0 else fit3d[1:])
        diagnostics.append({
            'seg_idx': i,
            'seg3d': seg,
            'curvature': curvature[idx0:idx1+1],
            'plane': (mean, b1, b2, nrm),
            'proj2d': seg2d,
            'fit2d': fit2d,
            'fit3d': fit3d,
            'theta0': theta0,
            'kappa': kappa,
            'start_idx': idx0,
            'end_idx': idx1
        })
    return np.array(fit_points), partition, diagnostics

# ---------------------------------------------------------
# EXTRACT ARC PARAMETERS FOR SIMULATION
# ---------------------------------------------------------
def extract_arc_parameters(diagnostics):
    """
    Returns a list of dicts, each with:
    - 'p0': (3,) starting point
    - 't0': (3,) initial tangent (unit vector)
    - 'n':  (3,) plane normal (unit vector)
    - 'kappa': scalar curvature
    - 'L': arc length
    - 'p1': (3,) end point (optional)
    """
    arc_segments = []
    for d in diagnostics:
        mean, b1, b2, nrm = d['plane']
        p0 = d['fit3d'][0]
        theta0 = d['theta0']
        t2d = np.array([np.cos(theta0), np.sin(theta0)])
        t0 = t2d[0]*b1 + t2d[1]*b2
        t0 = t0 / np.linalg.norm(t0)
        n = nrm / np.linalg.norm(nrm)
        kappa = d['kappa']
        L = np.sum(np.linalg.norm(np.diff(d['fit3d'], axis=0), axis=1))
        p1 = d['fit3d'][-1]
        arc_segments.append({
            'p0': p0,
            't0': t0,
            'n': n,
            'kappa': kappa,
            'L': L,
            'p1': p1
        })
    return arc_segments

# ---------------------------------------------------------
# RUN AND PLOT 10 EXAMPLES, PRINT ARC PARAMETERS
# ---------------------------------------------------------
np.random.seed(42)
for ex in range(N_EXAMPLES):
    points = synthetic_s_curve_3d(GRID_SIZE)
    best_fit = None
    best_err = np.inf
    for n_segments in SEGMENT_OPTIONS:
        fit_points, partition, diagnostics = fit_3d_arcspline(points, n_segments=n_segments)
        min_len = min(len(fit_points), len(points))
        error = np.mean(np.linalg.norm(fit_points[:min_len] - points[:min_len], axis=1))
        if error < best_err:
            best_err = error
            best_fit = (fit_points, partition, diagnostics, n_segments, error)
    fit_points, partition, diagnostics, n_segments, error = best_fit

    # --- PLOT 3D FIT ---
    fig = plt.figure(figsize=(13,5))
    ax = fig.add_subplot(131, projection='3d')
    ax.plot(points[:,0], points[:,1], points[:,2], '-', label='Target', lw=1)
    ax.plot(fit_points[:,0], fit_points[:,1], fit_points[:,2], 'r--', label='Arc-spline fit', lw=2)
    ax.scatter(points[partition,0], points[partition,1], points[partition,2], c='k', s=50, label='Segment breaks')
    ax.set_title(f"Example {ex+1}: 3D Arc-Spline ({n_segments} segs)\nMean Error={error:.3f}")
    ax.legend()
    ax.view_init(elev=20, azim=45)

    # --- PLOT SEGMENT CURVATURES ---
    ax2 = fig.add_subplot(132)
    curvature = estimate_curvature_3d(points)
    ax2.plot(curvature, label="3D Curvature", lw=1)
    for d in diagnostics:
        ax2.hlines(d['kappa'], d['start_idx'], d['end_idx'], color='r', lw=2)
        ax2.axvline(d['start_idx'], color='k', ls=':', lw=1)
    ax2.axvline(len(points)-1, color='k', ls=':', lw=1)
    ax2.set_title("Curvature Profile\n(Red=fit arc κ per segment)")
    ax2.set_xlabel("Point index")
    ax2.legend()

    # --- PLOT 2D FIT IN PLANE FOR EACH SEGMENT ---
    ax3 = fig.add_subplot(133)
    for d in diagnostics:
        ax3.plot(d['proj2d'][:,0], d['proj2d'][:,1], label=f"Segment {d['seg_idx']+1} target", lw=1)
        ax3.plot(d['fit2d'][:,0], d['fit2d'][:,1], '--', label=f"Seg {d['seg_idx']+1} arc fit", lw=2)
        ax3.scatter(d['proj2d'][0,0], d['proj2d'][0,1], c='g', marker='o', s=60)
        ax3.scatter(d['proj2d'][-1,0], d['proj2d'][-1,1], c='k', marker='s', s=40)
    ax3.set_title("Each Segment: 2D Plane Fit")
    ax3.axis('equal')
    ax3.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    # --- PRINT ARC SEGMENT PARAMETERS ---
    arc_segments = extract_arc_parameters(diagnostics)
    print(f"\n=== Example {ex+1}: Arc Segment Parameters for Simulation ===")
    for i, seg in enumerate(arc_segments):
        print(f"Segment {i+1}:")
        print(f"  Start point p0:  {np.round(seg['p0'], 3)}")
        print(f"  Tangent t0:      {np.round(seg['t0'], 4)}")
        print(f"  Plane normal n:  {np.round(seg['n'], 4)}")
        print(f"  Curvature kappa: {seg['kappa']:.6f}")
        print(f"  Arc length L:    {seg['L']:.4f}")
        print(f"  End point p1:    {np.round(seg['p1'], 3)}")
    print("-" * 55)