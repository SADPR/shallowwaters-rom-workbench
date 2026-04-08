"""
Scenario builders for shallow-water experiments.
"""

import numpy as np

from .config import BASE_DEPTH, LEFT_CENTER, RIGHT_CENTER, SIGMA


def two_bumps_collision_initial_state(
    X,
    Y,
    mu1,
    mu2,
    base_depth=BASE_DEPTH,
    sigma=SIGMA,
    left_center=LEFT_CENTER,
    right_center=RIGHT_CENTER,
):
    """
    Build initial conservative state U = [h, hu, hv] for two Gaussian bumps.

    Parameters
    ----------
    X, Y : ndarray
        Cell-center meshgrid with shape (nx, ny), indexing='ij'.
    mu1, mu2 : float
        Amplitudes of left and right bumps.
    base_depth : float
        Constant background water depth.
    sigma : float
        Common standard deviation for both Gaussian bumps.
    left_center, right_center : tuple(float, float)
        Centers of the two bumps in physical coordinates.
    """
    mu1 = float(mu1)
    mu2 = float(mu2)
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}.")

    xc1, yc1 = float(left_center[0]), float(left_center[1])
    xc2, yc2 = float(right_center[0]), float(right_center[1])

    bump1 = mu1 * np.exp(-((X - xc1) ** 2 + (Y - yc1) ** 2) / (2.0 * sigma**2))
    bump2 = mu2 * np.exp(-((X - xc2) ** 2 + (Y - yc2) ** 2) / (2.0 * sigma**2))

    h_init = float(base_depth) + bump1 + bump2
    u_init = np.zeros_like(h_init)
    v_init = np.zeros_like(h_init)

    U0 = np.zeros((3,) + h_init.shape, dtype=np.float64)
    U0[0] = h_init
    U0[1] = h_init * u_init
    U0[2] = h_init * v_init
    return U0

