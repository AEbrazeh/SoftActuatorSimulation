import numpy as np

# identical to your earlier function (shortened comment)
def synthetic_s_curve_3d(grid_size=100):
    s = np.linspace(0, np.pi, grid_size)
    k1 = np.random.uniform(0.3, 0.5)
    k2 = -np.random.uniform(0.3, 0.5)
    frac = np.random.uniform(0.4, 0.6)
    len1 = int(frac * grid_size); len2 = grid_size - len1
    theta1 = k1 * np.linspace(0, 1, len1)
    phi1   = np.random.uniform(0, 2*np.pi)
    x1 = np.cumsum(np.cos(theta1) + 0.05*np.random.randn(len1))
    y1 = np.cumsum(np.sin(theta1) + 0.05*np.random.randn(len1))
    z1 = np.cumsum(np.sin(theta1+phi1)+0.05*np.random.randn(len1))
    theta2 = theta1[-1] + k2*np.linspace(0, 1, len2)
    phi2   = phi1 + np.random.uniform(-0.8, 0.8)
    x2 = x1[-1] + np.cumsum(np.cos(theta2)+0.05*np.random.randn(len2))
    y2 = y1[-1] + np.cumsum(np.sin(theta2)+0.05*np.random.randn(len2))
    z2 = z1[-1] + np.cumsum(np.sin(theta2+phi2)+0.05*np.random.randn(len2))
    return np.column_stack([np.concatenate([x1,x2]),
                            np.concatenate([y1,y2]),
                            np.concatenate([z1,z2])])
