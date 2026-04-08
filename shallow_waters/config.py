"""
Global configuration for the shallow-water workbench.

The project currently focuses on one benchmark:
    SCENARIO = "two_bumps_collision"

The two physical parameters are:
    mu1 = left bump amplitude
    mu2 = right bump amplitude

Time is treated as an explicit third coordinate through saved snapshots at
fixed sampling times.
"""

import os

import numpy as np

SCENARIO = "two_bumps_collision"

# Physics / numerics.
G = 9.81
CFL = 0.18
H_FLOOR = 1e-10

# Time integration:
#   - "explicit_rk2"  : SSP-RK2 (original explicit scheme)
#   - "implicit_be"   : Backward Euler solved by a chosen nonlinear solver
#   - "implicit_bdf2" : BDF2 (BE startup step) solved by a chosen nonlinear solver
TIME_INTEGRATOR = "implicit_bdf2"

# Spatial discretization controls.
#   - LIMITER: "minmod" (robust) or "mc" (less diffusive)
#   - RIEMANN_FLUX: "hll" (robust) or "hllc" (sharper contacts)
LIMITER = "mc"
RIEMANN_FLUX = "hllc"

# Multiply CFL-based dt by this factor. Values > 1 can be useful with implicit BE.
# No adaptive fallback is used; this factor is the only dt scaling knob.
DT_MULTIPLIER = 1.0

# Implicit nonlinear solver controls (used for implicit_be / implicit_bdf2).
IMPLICIT_MAX_ITER = 100
IMPLICIT_TOL = 1e-6
# Used only when IMPLICIT_NONLINEAR_SOLVER == "picard".
IMPLICIT_RELAXATION = 1.0
IMPLICIT_VERBOSE_ITERS = False
# "newton_krylov" (recommended) or "picard" (legacy/prototype baseline)
IMPLICIT_NONLINEAR_SOLVER = "newton_krylov"

# Domain / mesh.
NX = 180
NY = 180
LX = 1.0
LY = 1.0

# Final simulation time and stored snapshot times.
T_FINAL = 0.28
NUM_TIME_SAMPLES = 141

# Two-bump parameterized initial condition.
BASE_DEPTH = 1.0
LEFT_CENTER = (0.30, 0.50)
RIGHT_CENTER = (0.70, 0.50)
SIGMA = 0.045

# Parameter box used for training snapshot generation.
MU1_RANGE = (0.06, 0.14)
MU2_RANGE = (0.06, 0.14)
SAMPLES_PER_MU = 3

# Boundary conditions for this benchmark.
BC_TWO_BUMPS = {
    "x_low": "reflective",
    "x_high": "reflective",
    "y_low": "reflective",
    "y_high": "reflective",
}

# Output locations.
RESULTS_DIR = "Results"
SNAP_FOLDER = os.path.join(RESULTS_DIR, "param_snaps")


def make_grid(nx=NX, ny=NY, lx=LX, ly=LY):
    """
    Build a uniform 2D grid and return centers and steps.
    """
    nx = int(nx)
    ny = int(ny)
    if nx < 1 or ny < 1:
        raise ValueError(f"nx and ny must be >= 1, got nx={nx}, ny={ny}.")

    dx = float(lx) / nx
    dy = float(ly) / ny

    x = np.linspace(dx / 2.0, float(lx) - dx / 2.0, nx)
    y = np.linspace(dy / 2.0, float(ly) - dy / 2.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    return {
        "nx": nx,
        "ny": ny,
        "lx": float(lx),
        "ly": float(ly),
        "dx": dx,
        "dy": dy,
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
    }


def get_sample_times(t_final=T_FINAL, num_time_samples=NUM_TIME_SAMPLES):
    """
    Uniform sampling times used to store states for ROM workflows.
    """
    n = int(num_time_samples)
    if n < 2:
        raise ValueError(f"num_time_samples must be >= 2, got {n}.")
    return np.linspace(0.0, float(t_final), n)
