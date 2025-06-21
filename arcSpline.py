import numpy as np
import matplotlib.pyplot as plt

def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

def generateArcSegment(s, theta, phi, p0, t0, n0, b0, nPoints=100):
    """
    Generate points along a single circular arc in 3D.
      s     = arc length
      theta = turning angle (radians)
      phi   = plane rotation around tangent (radians)
      p0    = start point (3,)
      t0    = start tangent (3,)
      n0, b0 = current normal/binormal (3,)
    Returns: (points[nPoints, 3], p1, t1, n1, b1)
    """
    R = s / theta
    # plane normal
    e1 = np.cos(phi) * n0 + np.sin(phi) * b0
    e2 = normalize(t0)
    # circle center
    C  = p0 + R * e1

    theta_ = np.linspace(0, theta, nPoints + 1)[:, None]
    pts =  R * (-np.cos(theta_)*e1 + np.sin(theta_)*e2) + C
    pts_ = normalize(R * (np.sin(theta_) * e1 + np.cos(theta_) * e2))
    
    # end point & tangent
    p1 = pts[-1]
    t1 = pts_[-1]

    # new normal & binormal for next segment
    n1 = normalize(p1 - C)
    b1 = normalize(np.cross(t1, n1))
    return pts, p1, t1, n1, b1

def generateSpline(arcs, pointDensity=100):
    """
    arcs: list of (s, theta, phi)
    returns: p[N,3]
    """
    # init frame at origin, tangent +x, normal +y, binormal +z
    p0 = np.array([0.0, 0.0, 0.0])
    t0 = np.array([1.0, 0.0, 0.0])
    n0 = np.array([0.0, 1.0, 0.0])
    b0 = np.cross(t0, n0)

    p = [p0.reshape(1,3)]
    for s, theta, phi in arcs:
        pts, p1, t1, n1, b1 = generateArcSegment(
            s, theta, phi, p0, t0, n0, b0, int(pointDensity * s)
        )
        # drop the first point to avoid duplicates
        p.append(pts[1:])
        # update for next
        p0, t0, n0, b0 = p1, t1, n1, b1
        
    return np.vstack(p)

if __name__ == "__main__":
    np.random.seed(1377)  # for reproducibility
    # Generate random arcs: (arc length, turning angle, plane rotation)
    arcs = np.random.rand(5, 3) * np.array([0.2, np.pi/2, np.pi])
    p = generateSpline(arcs, pointDensity=1000)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p[:,0], p[:,1], p[:,2])
    plt.show()