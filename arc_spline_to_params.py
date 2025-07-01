import numpy as np

def arc_segments_to_l_beta_alpha(arc_segments):
    """
    Converts extracted arc segment data to (l, beta, alpha) for each segment.
    Returns an (n_segments, 3) array: [arc length, angle, plane orientation]
    """
    params = []
    for seg in arc_segments:
        L = seg['L']                  # Arc length
        kappa = seg['kappa']          # Curvature
        beta = kappa * L              # Total subtended angle (radians)
        n = seg['n']                  # Plane normal vector (3,)
        # The robot's convention: rotation axis is [0, ny, nz]
        # Orientation alpha is arctan2(ny, nz)
        alpha = np.arctan2(n[1], n[2])
        params.append([L, beta, alpha])
    return np.array(params)