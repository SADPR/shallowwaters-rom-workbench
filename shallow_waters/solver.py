"""
Numerical kernels for 2D shallow-water equations on a flat bottom.

Discretization:
  - MUSCL reconstruction in primitive variables with selectable limiter
  - HLL or HLLC numerical flux
  - SSP-RK2 / implicit BE / implicit BDF2 time integration
"""

import numpy as np


def primitive(U, h_floor):
    """
    Conservative -> primitive conversion for U = [h, hu, hv].
    """
    h = U[0]
    hu = U[1]
    hv = U[2]

    u = np.zeros_like(h)
    v = np.zeros_like(h)

    wet = h > float(h_floor)
    u[wet] = hu[wet] / h[wet]
    v[wet] = hv[wet] / h[wet]
    return h, u, v


def conservative(h, u, v):
    """
    Primitive -> conservative conversion for [h, u, v].
    """
    U = np.zeros((3,) + h.shape, dtype=np.float64)
    U[0] = h
    U[1] = h * u
    U[2] = h * v
    return U


def flatten_state(U):
    """
    Convert U[3, nx, ny] to state vector w[(3*nx*ny),].
    """
    return np.asarray(U, dtype=np.float64).reshape(-1)


def unflatten_state(w, nx, ny):
    """
    Convert state vector to U[3, nx, ny].
    """
    return np.asarray(w, dtype=np.float64).reshape(3, int(nx), int(ny))


def minmod(a, b):
    out = np.zeros_like(a)
    same_sign = (a * b) > 0.0
    out[same_sign] = np.sign(a[same_sign]) * np.minimum(
        np.abs(a[same_sign]), np.abs(b[same_sign])
    )
    return out


def minmod3(a, b, c):
    out = np.zeros_like(a)
    same_sign = (a * b > 0.0) & (a * c > 0.0)
    out[same_sign] = np.sign(a[same_sign]) * np.minimum(
        np.minimum(np.abs(a[same_sign]), np.abs(b[same_sign])),
        np.abs(c[same_sign]),
    )
    return out


def limited_slope(dL, dR, limiter="minmod"):
    """
    Return limited slope from one-sided differences dL and dR.

    limiter:
      - "minmod" : robust, more diffusive
      - "mc"     : Monotonized Central, less diffusive
    """
    limiter = str(limiter).strip().lower()
    if limiter == "minmod":
        return minmod(dL, dR)
    if limiter in {"mc", "monotonized_central", "monotonized-central"}:
        return minmod3(0.5 * (dL + dR), 2.0 * dL, 2.0 * dR)
    raise ValueError(f"Unknown limiter='{limiter}'. Use 'minmod' or 'mc'.")


def fill_x_boundary(Ug, U, side, bc_type):
    """
    Fill x-side ghost cells.
    """
    if side == "low":
        Ug[:, 0, 1:-1] = U[:, 0, :]
        if bc_type == "reflective":
            Ug[1, 0, 1:-1] *= -1.0
        elif bc_type == "transmissive":
            pass
        else:
            raise ValueError(f"Unknown x boundary type: {bc_type}")
        return

    if side == "high":
        Ug[:, -1, 1:-1] = U[:, -1, :]
        if bc_type == "reflective":
            Ug[1, -1, 1:-1] *= -1.0
        elif bc_type == "transmissive":
            pass
        else:
            raise ValueError(f"Unknown x boundary type: {bc_type}")
        return

    raise ValueError(f"Unknown side: {side}")


def fill_y_boundary(Ug, U, side, bc_type):
    """
    Fill y-side ghost cells.
    """
    if side == "low":
        Ug[:, 1:-1, 0] = U[:, :, 0]
        if bc_type == "reflective":
            Ug[2, 1:-1, 0] *= -1.0
        elif bc_type == "transmissive":
            pass
        else:
            raise ValueError(f"Unknown y boundary type: {bc_type}")
        return

    if side == "high":
        Ug[:, 1:-1, -1] = U[:, :, -1]
        if bc_type == "reflective":
            Ug[2, 1:-1, -1] *= -1.0
        elif bc_type == "transmissive":
            pass
        else:
            raise ValueError(f"Unknown y boundary type: {bc_type}")
        return

    raise ValueError(f"Unknown side: {side}")


def fill_corner(Ug, U, i, j, flip_hu, flip_hv):
    """
    Fill one corner ghost cell from the corresponding physical corner.
    """
    src_i = 0 if i == 0 else -1
    src_j = 0 if j == 0 else -1
    Ug[:, i, j] = U[:, src_i, src_j]
    if flip_hu:
        Ug[1, i, j] *= -1.0
    if flip_hv:
        Ug[2, i, j] *= -1.0


def add_ghost_cells(U, bc):
    """
    Add one ghost-cell layer around U.
    """
    nx, ny = U.shape[1], U.shape[2]
    Ug = np.zeros((3, nx + 2, ny + 2), dtype=np.float64)
    Ug[:, 1:-1, 1:-1] = U

    fill_x_boundary(Ug, U, "low", bc["x_low"])
    fill_x_boundary(Ug, U, "high", bc["x_high"])
    fill_y_boundary(Ug, U, "low", bc["y_low"])
    fill_y_boundary(Ug, U, "high", bc["y_high"])

    flip_hu_low = bc["x_low"] == "reflective"
    flip_hu_high = bc["x_high"] == "reflective"
    flip_hv_low = bc["y_low"] == "reflective"
    flip_hv_high = bc["y_high"] == "reflective"

    fill_corner(Ug, U, 0, 0, flip_hu_low, flip_hv_low)
    fill_corner(Ug, U, 0, -1, flip_hu_low, flip_hv_high)
    fill_corner(Ug, U, -1, 0, flip_hu_high, flip_hv_low)
    fill_corner(Ug, U, -1, -1, flip_hu_high, flip_hv_high)
    return Ug


def reconstruct_x(Ug, h_floor, limiter="minmod"):
    """
    MUSCL reconstruction at x-interfaces.
    """
    h, u, v = primitive(Ug, h_floor=h_floor)
    nx = Ug.shape[1] - 2
    ny = Ug.shape[2] - 2

    dhL = h[1:-1, 1:-1] - h[:-2, 1:-1]
    dhR = h[2:, 1:-1] - h[1:-1, 1:-1]
    duL = u[1:-1, 1:-1] - u[:-2, 1:-1]
    duR = u[2:, 1:-1] - u[1:-1, 1:-1]
    dvL = v[1:-1, 1:-1] - v[:-2, 1:-1]
    dvR = v[2:, 1:-1] - v[1:-1, 1:-1]

    sh = limited_slope(dhL, dhR, limiter=limiter)
    su = limited_slope(duL, duR, limiter=limiter)
    sv = limited_slope(dvL, dvR, limiter=limiter)

    hL_cell = np.maximum(h[1:-1, 1:-1] - 0.5 * sh, h_floor)
    hR_cell = np.maximum(h[1:-1, 1:-1] + 0.5 * sh, h_floor)
    uL_cell = u[1:-1, 1:-1] - 0.5 * su
    uR_cell = u[1:-1, 1:-1] + 0.5 * su
    vL_cell = v[1:-1, 1:-1] - 0.5 * sv
    vR_cell = v[1:-1, 1:-1] + 0.5 * sv

    h_left = np.zeros((nx + 1, ny), dtype=np.float64)
    h_right = np.zeros((nx + 1, ny), dtype=np.float64)
    u_left = np.zeros((nx + 1, ny), dtype=np.float64)
    u_right = np.zeros((nx + 1, ny), dtype=np.float64)
    v_left = np.zeros((nx + 1, ny), dtype=np.float64)
    v_right = np.zeros((nx + 1, ny), dtype=np.float64)

    h_left[1:nx, :] = hR_cell[:-1, :]
    h_right[1:nx, :] = hL_cell[1:, :]
    u_left[1:nx, :] = uR_cell[:-1, :]
    u_right[1:nx, :] = uL_cell[1:, :]
    v_left[1:nx, :] = vR_cell[:-1, :]
    v_right[1:nx, :] = vL_cell[1:, :]

    h_left[0, :] = h[0, 1:-1]
    h_right[0, :] = hL_cell[0, :]
    u_left[0, :] = u[0, 1:-1]
    u_right[0, :] = uL_cell[0, :]
    v_left[0, :] = v[0, 1:-1]
    v_right[0, :] = vL_cell[0, :]

    h_left[nx, :] = hR_cell[-1, :]
    h_right[nx, :] = h[-1, 1:-1]
    u_left[nx, :] = uR_cell[-1, :]
    u_right[nx, :] = u[-1, 1:-1]
    v_left[nx, :] = vR_cell[-1, :]
    v_right[nx, :] = v[-1, 1:-1]

    UL = conservative(h_left, u_left, v_left)
    UR = conservative(h_right, u_right, v_right)
    return UL, UR


def reconstruct_y(Ug, h_floor, limiter="minmod"):
    """
    MUSCL reconstruction at y-interfaces.
    """
    h, u, v = primitive(Ug, h_floor=h_floor)
    nx = Ug.shape[1] - 2
    ny = Ug.shape[2] - 2

    dhL = h[1:-1, 1:-1] - h[1:-1, :-2]
    dhR = h[1:-1, 2:] - h[1:-1, 1:-1]
    duL = u[1:-1, 1:-1] - u[1:-1, :-2]
    duR = u[1:-1, 2:] - u[1:-1, 1:-1]
    dvL = v[1:-1, 1:-1] - v[1:-1, :-2]
    dvR = v[1:-1, 2:] - v[1:-1, 1:-1]

    sh = limited_slope(dhL, dhR, limiter=limiter)
    su = limited_slope(duL, duR, limiter=limiter)
    sv = limited_slope(dvL, dvR, limiter=limiter)

    hL_cell = np.maximum(h[1:-1, 1:-1] - 0.5 * sh, h_floor)
    hR_cell = np.maximum(h[1:-1, 1:-1] + 0.5 * sh, h_floor)
    uL_cell = u[1:-1, 1:-1] - 0.5 * su
    uR_cell = u[1:-1, 1:-1] + 0.5 * su
    vL_cell = v[1:-1, 1:-1] - 0.5 * sv
    vR_cell = v[1:-1, 1:-1] + 0.5 * sv

    h_left = np.zeros((nx, ny + 1), dtype=np.float64)
    h_right = np.zeros((nx, ny + 1), dtype=np.float64)
    u_left = np.zeros((nx, ny + 1), dtype=np.float64)
    u_right = np.zeros((nx, ny + 1), dtype=np.float64)
    v_left = np.zeros((nx, ny + 1), dtype=np.float64)
    v_right = np.zeros((nx, ny + 1), dtype=np.float64)

    h_left[:, 1:ny] = hR_cell[:, :-1]
    h_right[:, 1:ny] = hL_cell[:, 1:]
    u_left[:, 1:ny] = uR_cell[:, :-1]
    u_right[:, 1:ny] = uL_cell[:, 1:]
    v_left[:, 1:ny] = vR_cell[:, :-1]
    v_right[:, 1:ny] = vL_cell[:, 1:]

    h_left[:, 0] = h[1:-1, 0]
    h_right[:, 0] = hL_cell[:, 0]
    u_left[:, 0] = u[1:-1, 0]
    u_right[:, 0] = uL_cell[:, 0]
    v_left[:, 0] = v[1:-1, 0]
    v_right[:, 0] = vL_cell[:, 0]

    h_left[:, ny] = hR_cell[:, -1]
    h_right[:, ny] = h[1:-1, -1]
    u_left[:, ny] = uR_cell[:, -1]
    u_right[:, ny] = u[1:-1, -1]
    v_left[:, ny] = vR_cell[:, -1]
    v_right[:, ny] = v[1:-1, -1]

    UL = conservative(h_left, u_left, v_left)
    UR = conservative(h_right, u_right, v_right)
    return UL, UR


def flux_x(U, g, h_floor):
    h, u, v = primitive(U, h_floor=h_floor)
    F = np.zeros_like(U)
    F[0] = U[1]
    F[1] = U[1] * u + 0.5 * float(g) * h**2
    F[2] = U[1] * v
    return F


def flux_y(U, g, h_floor):
    h, u, v = primitive(U, h_floor=h_floor)
    G = np.zeros_like(U)
    G[0] = U[2]
    G[1] = U[2] * u
    G[2] = U[2] * v + 0.5 * float(g) * h**2
    return G


def _safe_denominator(den):
    eps = 1e-14
    out = den.copy()
    small = np.abs(out) < eps
    out[small] = np.where(out[small] >= 0.0, eps, -eps)
    return out


def hll_flux_x(UL, UR, g, h_floor):
    hL, uL, _ = primitive(UL, h_floor=h_floor)
    hR, uR, _ = primitive(UR, h_floor=h_floor)

    cL = np.sqrt(float(g) * np.maximum(hL, 0.0))
    cR = np.sqrt(float(g) * np.maximum(hR, 0.0))

    sL = np.minimum(uL - cL, uR - cR)
    sR = np.maximum(uL + cL, uR + cR)

    FL = flux_x(UL, g=g, h_floor=h_floor)
    FR = flux_x(UR, g=g, h_floor=h_floor)

    F = np.zeros_like(FL)

    maskL = sL >= 0.0
    maskR = sR <= 0.0
    maskM = (~maskL) & (~maskR)

    F[:, maskL] = FL[:, maskL]
    F[:, maskR] = FR[:, maskR]

    if np.any(maskM):
        sLm = sL[maskM]
        sRm = sR[maskM]
        den = _safe_denominator(sRm - sLm)
        F[:, maskM] = (
            sRm * FL[:, maskM]
            - sLm * FR[:, maskM]
            + sLm * sRm * (UR[:, maskM] - UL[:, maskM])
        ) / den

    return F


def hll_flux_y(UL, UR, g, h_floor):
    hL, _, vL = primitive(UL, h_floor=h_floor)
    hR, _, vR = primitive(UR, h_floor=h_floor)

    cL = np.sqrt(float(g) * np.maximum(hL, 0.0))
    cR = np.sqrt(float(g) * np.maximum(hR, 0.0))

    sL = np.minimum(vL - cL, vR - cR)
    sR = np.maximum(vL + cL, vR + cR)

    GL = flux_y(UL, g=g, h_floor=h_floor)
    GR = flux_y(UR, g=g, h_floor=h_floor)

    Gf = np.zeros_like(GL)

    maskL = sL >= 0.0
    maskR = sR <= 0.0
    maskM = (~maskL) & (~maskR)

    Gf[:, maskL] = GL[:, maskL]
    Gf[:, maskR] = GR[:, maskR]

    if np.any(maskM):
        sLm = sL[maskM]
        sRm = sR[maskM]
        den = _safe_denominator(sRm - sLm)
        Gf[:, maskM] = (
            sRm * GL[:, maskM]
            - sLm * GR[:, maskM]
            + sLm * sRm * (UR[:, maskM] - UL[:, maskM])
        ) / den

    return Gf


def hllc_flux_x(UL, UR, g, h_floor):
    """
    HLLC flux in x-direction for shallow water.

    Falls back to HLL when star-state values become non-finite.
    """
    hL, uL, vL = primitive(UL, h_floor=h_floor)
    hR, uR, vR = primitive(UR, h_floor=h_floor)

    cL = np.sqrt(float(g) * np.maximum(hL, 0.0))
    cR = np.sqrt(float(g) * np.maximum(hR, 0.0))
    sL = np.minimum(uL - cL, uR - cR)
    sR = np.maximum(uL + cL, uR + cR)

    FL = flux_x(UL, g=g, h_floor=h_floor)
    FR = flux_x(UR, g=g, h_floor=h_floor)
    F = np.zeros_like(FL)

    pL = 0.5 * float(g) * hL**2
    pR = 0.5 * float(g) * hR**2
    den_star = _safe_denominator(
        hR * (sR - uR) - hL * (sL - uL)
    )
    s_star = (
        pL
        - pR
        + hR * uR * (sR - uR)
        - hL * uL * (sL - uL)
    ) / den_star

    den_L = _safe_denominator(sL - s_star)
    den_R = _safe_denominator(sR - s_star)
    h_star_L = np.maximum(hL * (sL - uL) / den_L, h_floor)
    h_star_R = np.maximum(hR * (sR - uR) / den_R, h_floor)

    U_star_L = np.zeros_like(UL)
    U_star_R = np.zeros_like(UR)
    U_star_L[0] = h_star_L
    U_star_L[1] = h_star_L * s_star
    U_star_L[2] = h_star_L * vL
    U_star_R[0] = h_star_R
    U_star_R[1] = h_star_R * s_star
    U_star_R[2] = h_star_R * vR

    maskL = sL >= 0.0
    maskR = sR <= 0.0
    maskSL = (~maskL) & (~maskR) & (s_star >= 0.0)
    maskSR = (~maskL) & (~maskR) & (~maskSL)

    F[:, maskL] = FL[:, maskL]
    F[:, maskR] = FR[:, maskR]
    F[:, maskSL] = FL[:, maskSL] + sL[maskSL] * (
        U_star_L[:, maskSL] - UL[:, maskSL]
    )
    F[:, maskSR] = FR[:, maskSR] + sR[maskSR] * (
        U_star_R[:, maskSR] - UR[:, maskSR]
    )
    bad = (~np.isfinite(s_star)) | (~np.isfinite(h_star_L)) | (~np.isfinite(h_star_R))
    if np.any(bad):
        F_hll = hll_flux_x(UL, UR, g=g, h_floor=h_floor)
        F[:, bad] = F_hll[:, bad]
    return F


def hllc_flux_y(UL, UR, g, h_floor):
    """
    HLLC flux in y-direction for shallow water.

    Falls back to HLL when star-state values become non-finite.
    """
    hL, uL, vL = primitive(UL, h_floor=h_floor)
    hR, uR, vR = primitive(UR, h_floor=h_floor)

    cL = np.sqrt(float(g) * np.maximum(hL, 0.0))
    cR = np.sqrt(float(g) * np.maximum(hR, 0.0))
    sL = np.minimum(vL - cL, vR - cR)
    sR = np.maximum(vL + cL, vR + cR)

    GL = flux_y(UL, g=g, h_floor=h_floor)
    GR = flux_y(UR, g=g, h_floor=h_floor)
    Gf = np.zeros_like(GL)

    pL = 0.5 * float(g) * hL**2
    pR = 0.5 * float(g) * hR**2
    den_star = _safe_denominator(
        hR * (sR - vR) - hL * (sL - vL)
    )
    s_star = (
        pL
        - pR
        + hR * vR * (sR - vR)
        - hL * vL * (sL - vL)
    ) / den_star

    den_L = _safe_denominator(sL - s_star)
    den_R = _safe_denominator(sR - s_star)
    h_star_L = np.maximum(hL * (sL - vL) / den_L, h_floor)
    h_star_R = np.maximum(hR * (sR - vR) / den_R, h_floor)

    U_star_L = np.zeros_like(UL)
    U_star_R = np.zeros_like(UR)
    U_star_L[0] = h_star_L
    U_star_L[1] = h_star_L * uL
    U_star_L[2] = h_star_L * s_star
    U_star_R[0] = h_star_R
    U_star_R[1] = h_star_R * uR
    U_star_R[2] = h_star_R * s_star

    maskL = sL >= 0.0
    maskR = sR <= 0.0
    maskSL = (~maskL) & (~maskR) & (s_star >= 0.0)
    maskSR = (~maskL) & (~maskR) & (~maskSL)

    Gf[:, maskL] = GL[:, maskL]
    Gf[:, maskR] = GR[:, maskR]
    Gf[:, maskSL] = GL[:, maskSL] + sL[maskSL] * (
        U_star_L[:, maskSL] - UL[:, maskSL]
    )
    Gf[:, maskSR] = GR[:, maskSR] + sR[maskSR] * (
        U_star_R[:, maskSR] - UR[:, maskSR]
    )
    bad = (~np.isfinite(s_star)) | (~np.isfinite(h_star_L)) | (~np.isfinite(h_star_R))
    if np.any(bad):
        G_hll = hll_flux_y(UL, UR, g=g, h_floor=h_floor)
        Gf[:, bad] = G_hll[:, bad]
    return Gf


def interface_flux_x(UL, UR, g, h_floor, riemann_flux="hll"):
    riemann_flux = str(riemann_flux).strip().lower()
    if riemann_flux == "hll":
        return hll_flux_x(UL, UR, g=g, h_floor=h_floor)
    if riemann_flux == "hllc":
        return hllc_flux_x(UL, UR, g=g, h_floor=h_floor)
    raise ValueError(f"Unknown riemann_flux='{riemann_flux}'. Use 'hll' or 'hllc'.")


def interface_flux_y(UL, UR, g, h_floor, riemann_flux="hll"):
    riemann_flux = str(riemann_flux).strip().lower()
    if riemann_flux == "hll":
        return hll_flux_y(UL, UR, g=g, h_floor=h_floor)
    if riemann_flux == "hllc":
        return hllc_flux_y(UL, UR, g=g, h_floor=h_floor)
    raise ValueError(f"Unknown riemann_flux='{riemann_flux}'. Use 'hll' or 'hllc'.")


def shallow_water_rhs2D(
    U,
    dx,
    dy,
    g,
    h_floor,
    bc,
    limiter="minmod",
    riemann_flux="hll",
):
    """
    Semi-discrete ODE RHS: dU/dt = R(U).
    """
    Ug = add_ghost_cells(U, bc=bc)
    ULx, URx = reconstruct_x(Ug, h_floor=h_floor, limiter=limiter)
    ULy, URy = reconstruct_y(Ug, h_floor=h_floor, limiter=limiter)

    Fx = interface_flux_x(
        ULx, URx, g=g, h_floor=h_floor, riemann_flux=riemann_flux
    )
    Gy = interface_flux_y(
        ULy, URy, g=g, h_floor=h_floor, riemann_flux=riemann_flux
    )

    R = np.zeros_like(U)
    R -= (Fx[:, 1:, :] - Fx[:, :-1, :]) / float(dx)
    R -= (Gy[:, :, 1:] - Gy[:, :, :-1]) / float(dy)
    return R


def shallow_water_rhs2D_flat(
    w,
    nx,
    ny,
    dx,
    dy,
    g,
    h_floor,
    bc,
    limiter="minmod",
    riemann_flux="hll",
):
    """
    Flattened-state wrapper of shallow_water_rhs2D for ROM code reuse.
    """
    U = unflatten_state(w, nx=nx, ny=ny)
    R = shallow_water_rhs2D(
        U,
        dx=dx,
        dy=dy,
        g=g,
        h_floor=h_floor,
        bc=bc,
        limiter=limiter,
        riemann_flux=riemann_flux,
    )
    return flatten_state(R)


def shallow_water_res2D(
    w,
    wp,
    dt,
    nx,
    ny,
    dx,
    dy,
    g,
    h_floor,
    bc,
    limiter="minmod",
    riemann_flux="hll",
):
    """
    Trapezoidal residual:
        res = w - wp - 0.5*dt*(rhs(w) + rhs(wp))

    This mirrors the residual-style interface used in the burgers workbench.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    wp = np.asarray(wp, dtype=np.float64).reshape(-1)
    rhs_w = shallow_water_rhs2D_flat(
        w,
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        g=g,
        h_floor=h_floor,
        bc=bc,
        limiter=limiter,
        riemann_flux=riemann_flux,
    )
    rhs_wp = shallow_water_rhs2D_flat(
        wp,
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        g=g,
        h_floor=h_floor,
        bc=bc,
        limiter=limiter,
        riemann_flux=riemann_flux,
    )
    return w - wp - 0.5 * float(dt) * (rhs_w + rhs_wp)


def positivity_fix(U, h_floor):
    Unew = U.copy()
    Unew[0] = np.maximum(Unew[0], float(h_floor))
    small = Unew[0] < 10.0 * float(h_floor)
    Unew[1][small] = 0.0
    Unew[2][small] = 0.0
    return Unew


def rk2_step(
    U,
    dt,
    dx,
    dy,
    g,
    cfl,
    h_floor,
    bc,
    limiter="minmod",
    riemann_flux="hll",
):
    """
    SSP-RK2 step for U' = rhs(U).
    """
    _ = cfl  # Kept in signature for compatibility with other integrators.
    R0 = shallow_water_rhs2D(
        U,
        dx=dx,
        dy=dy,
        g=g,
        h_floor=h_floor,
        bc=bc,
        limiter=limiter,
        riemann_flux=riemann_flux,
    )
    U1 = positivity_fix(U + float(dt) * R0, h_floor=h_floor)
    R1 = shallow_water_rhs2D(
        U1,
        dx=dx,
        dy=dy,
        g=g,
        h_floor=h_floor,
        bc=bc,
        limiter=limiter,
        riemann_flux=riemann_flux,
    )
    U2 = positivity_fix(0.5 * U + 0.5 * (U1 + float(dt) * R1), h_floor=h_floor)
    return U2


def implicit_residual(
    U_new,
    U_old,
    U_older,
    rhs_scale,
    coeff_old,
    coeff_older,
    dx,
    dy,
    g,
    h_floor,
    bc,
    limiter="minmod",
    riemann_flux="hll",
):
    """
    Generic one-step implicit residual:
        G(U_new) = U_new - coeff_old*U_old - coeff_older*U_older
                   - rhs_scale*rhs(U_new)
    """
    rhs_new = shallow_water_rhs2D(
        U_new,
        dx=dx,
        dy=dy,
        g=g,
        h_floor=h_floor,
        bc=bc,
        limiter=limiter,
        riemann_flux=riemann_flux,
    )
    return (
        U_new
        - float(coeff_old) * U_old
        - float(coeff_older) * U_older
        - float(rhs_scale) * rhs_new
    )


def backward_euler_residual(
    U_new,
    U_old,
    dt,
    dx,
    dy,
    g,
    h_floor,
    bc,
    limiter="minmod",
    riemann_flux="hll",
):
    """
    Backward-Euler residual:
        G(U^{n+1}) = U^{n+1} - U^n - dt * rhs(U^{n+1})
    """
    return implicit_residual(
        U_new=U_new,
        U_old=U_old,
        U_older=U_old,
        rhs_scale=float(dt),
        coeff_old=1.0,
        coeff_older=0.0,
        dx=dx,
        dy=dy,
        g=g,
        h_floor=h_floor,
        bc=bc,
        limiter=limiter,
        riemann_flux=riemann_flux,
    )


def bdf2_coefficients(dt, dt_prev):
    """
    Variable-step BDF2 coefficients in normalized residual form:
        U_{n+1} - a1*U_n - a2*U_{n-1} - beta*rhs(U_{n+1}) = 0
    """
    dt = float(dt)
    dt_prev = float(dt_prev)
    if dt <= 0.0 or dt_prev <= 0.0:
        raise ValueError(f"dt and dt_prev must be > 0, got dt={dt}, dt_prev={dt_prev}.")

    r = dt / dt_prev
    c0 = (1.0 + 2.0 * r) / (1.0 + r)
    c1 = -(1.0 + r)
    c2 = r * r / (1.0 + r)

    coeff_old = -c1 / c0
    coeff_older = -c2 / c0
    rhs_scale = dt / c0
    return coeff_old, coeff_older, rhs_scale


def _implicit_step_picard_generic(
    U_old,
    U_older,
    coeff_old,
    coeff_older,
    rhs_scale,
    dx,
    dy,
    g,
    h_floor,
    bc,
    max_iter=25,
    tol=1e-8,
    relaxation=1.0,
    limiter="minmod",
    riemann_flux="hll",
    residual_name="imp_res",
    verbose=False,
):
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}.")

    tol = float(tol)
    relaxation = float(relaxation)
    if not (0.0 < relaxation <= 1.0):
        raise ValueError(f"relaxation must be in (0, 1], got {relaxation}.")

    U_old = np.asarray(U_old, dtype=np.float64)
    U_older = np.asarray(U_older, dtype=np.float64)
    U_base = float(coeff_old) * U_old + float(coeff_older) * U_older

    U_guess = positivity_fix(
        U_base
        + float(rhs_scale)
        * shallow_water_rhs2D(
            U_old,
            dx=dx,
            dy=dy,
            g=g,
            h_floor=h_floor,
            bc=bc,
            limiter=limiter,
            riemann_flux=riemann_flux,
        ),
        h_floor=h_floor,
    )

    converged = False
    residual = np.inf
    iterations = 0

    for k in range(1, max_iter + 1):
        G = implicit_residual(
            U_new=U_guess,
            U_old=U_old,
            U_older=U_older,
            rhs_scale=rhs_scale,
            coeff_old=coeff_old,
            coeff_older=coeff_older,
            dx=dx,
            dy=dy,
            g=g,
            h_floor=h_floor,
            bc=bc,
            limiter=limiter,
            riemann_flux=riemann_flux,
        )
        residual = float(np.linalg.norm(G.reshape(-1)))
        scale = max(float(np.linalg.norm(U_guess.reshape(-1))), 1e-14)
        residual /= scale
        iterations = k

        if verbose:
            print(f"    [NL] iter={k:3d} | {residual_name}={residual:.3e}")

        if residual <= tol:
            converged = True
            break

        rhs_guess = shallow_water_rhs2D(
            U_guess,
            dx=dx,
            dy=dy,
            g=g,
            h_floor=h_floor,
            bc=bc,
            limiter=limiter,
            riemann_flux=riemann_flux,
        )
        U_target = positivity_fix(U_base + float(rhs_scale) * rhs_guess, h_floor=h_floor)
        U_guess = positivity_fix(
            (1.0 - relaxation) * U_guess + relaxation * U_target,
            h_floor=h_floor,
        )

    return U_guess, {
        "converged": bool(converged),
        "iterations": int(iterations),
        "residual": float(residual),
        "linear_iterations": 0,
        "line_search_reductions": 0,
    }


def backward_euler_step_picard(
    U_old,
    dt,
    dx,
    dy,
    g,
    h_floor,
    bc,
    max_iter=25,
    tol=1e-8,
    relaxation=1.0,
    limiter="minmod",
    riemann_flux="hll",
    residual_name="be_res",
    verbose=False,
):
    """
    Implicit Backward-Euler step solved by Picard fixed-point iterations.
    """
    return _implicit_step_picard_generic(
        U_old=U_old,
        U_older=U_old,
        coeff_old=1.0,
        coeff_older=0.0,
        rhs_scale=float(dt),
        dx=dx,
        dy=dy,
        g=g,
        h_floor=h_floor,
        bc=bc,
        max_iter=max_iter,
        tol=tol,
        relaxation=relaxation,
        limiter=limiter,
        riemann_flux=riemann_flux,
        residual_name=residual_name,
        verbose=verbose,
    )


def bdf2_step_picard(
    U_old,
    U_older,
    dt,
    dt_prev,
    dx,
    dy,
    g,
    h_floor,
    bc,
    max_iter=25,
    tol=1e-8,
    relaxation=1.0,
    limiter="minmod",
    riemann_flux="hll",
    residual_name="bdf2_res",
    verbose=False,
):
    """
    Implicit BDF2 step solved by Picard fixed-point iterations.
    """
    coeff_old, coeff_older, rhs_scale = bdf2_coefficients(dt=dt, dt_prev=dt_prev)
    return _implicit_step_picard_generic(
        U_old=U_old,
        U_older=U_older,
        coeff_old=coeff_old,
        coeff_older=coeff_older,
        rhs_scale=rhs_scale,
        dx=dx,
        dy=dy,
        g=g,
        h_floor=h_floor,
        bc=bc,
        max_iter=max_iter,
        tol=tol,
        relaxation=relaxation,
        limiter=limiter,
        riemann_flux=riemann_flux,
        residual_name=residual_name,
        verbose=verbose,
    )


def _relative_nl_residual(G, U):
    num = float(np.linalg.norm(np.asarray(G, dtype=np.float64).reshape(-1)))
    den = float(np.linalg.norm(np.asarray(U, dtype=np.float64).reshape(-1)))
    den = max(den, 1e-14)
    return num / den


def _gmres_matrix_free(jvp, b, restart=12, max_restarts=3, tol=1e-2):
    """
    Basic restarted GMRES for matrix-free Jacobian-vector products.

    Solves approximately: J * x = b.
    """
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = b.size
    if n == 0:
        return np.asarray([], dtype=np.float64), True, 0, 0.0

    restart = int(max(restart, 1))
    max_restarts = int(max(max_restarts, 0))
    tol = float(max(tol, 1e-14))

    x = np.zeros(n, dtype=np.float64)
    r = b.copy()
    bnorm = max(float(np.linalg.norm(b)), 1e-14)
    rnorm = float(np.linalg.norm(r))
    total_iters = 0

    for _ in range(max_restarts + 1):
        if rnorm <= tol * bnorm:
            return x, True, total_iters, rnorm / bnorm

        m = restart
        V = np.zeros((n, m + 1), dtype=np.float64)
        H = np.zeros((m + 1, m), dtype=np.float64)
        V[:, 0] = r / max(rnorm, 1e-14)

        g = np.zeros(m + 1, dtype=np.float64)
        g[0] = rnorm

        converged_inner = False
        k_last = 0
        for k in range(m):
            w = np.asarray(jvp(V[:, k]), dtype=np.float64).reshape(-1)

            for j in range(k + 1):
                H[j, k] = float(np.dot(V[:, j], w))
                w -= H[j, k] * V[:, j]

            H[k + 1, k] = float(np.linalg.norm(w))
            if H[k + 1, k] > 1e-14 and (k + 1) < (m + 1):
                V[:, k + 1] = w / H[k + 1, k]

            A = H[: k + 2, : k + 1]
            y, *_ = np.linalg.lstsq(A, g[: k + 2], rcond=None)
            res_est = float(np.linalg.norm(g[: k + 2] - A @ y))
            total_iters += 1
            k_last = k

            if res_est <= tol * bnorm:
                x += V[:, : k + 1] @ y
                r = b - np.asarray(jvp(x), dtype=np.float64).reshape(-1)
                rnorm = float(np.linalg.norm(r))
                converged_inner = True
                break

        if not converged_inner:
            A = H[: k_last + 2, : k_last + 1]
            y, *_ = np.linalg.lstsq(A, g[: k_last + 2], rcond=None)
            x += V[:, : k_last + 1] @ y
            r = b - np.asarray(jvp(x), dtype=np.float64).reshape(-1)
            rnorm = float(np.linalg.norm(r))

    return x, bool(rnorm <= tol * bnorm), total_iters, rnorm / bnorm


def _implicit_step_newton_krylov_generic(
    U_old,
    U_older,
    coeff_old,
    coeff_older,
    rhs_scale,
    dx,
    dy,
    g,
    h_floor,
    bc,
    max_iter=25,
    tol=1e-8,
    gmres_restart=12,
    gmres_max_restarts=3,
    jac_eps=1e-7,
    line_search_max_steps=8,
    line_search_c1=1e-4,
    limiter="minmod",
    riemann_flux="hll",
    residual_name="imp_res",
    verbose=False,
):
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}.")

    tol = float(tol)
    gmres_restart = int(max(gmres_restart, 1))
    gmres_max_restarts = int(max(gmres_max_restarts, 0))
    jac_eps = float(max(jac_eps, 1e-14))
    line_search_max_steps = int(max(line_search_max_steps, 0))
    line_search_c1 = float(line_search_c1)

    U_old = np.asarray(U_old, dtype=np.float64)
    U_older = np.asarray(U_older, dtype=np.float64)
    nx, ny = U_old.shape[1], U_old.shape[2]
    U_base = float(coeff_old) * U_old + float(coeff_older) * U_older

    U = positivity_fix(
        U_base
        + float(rhs_scale)
        * shallow_water_rhs2D(
            U_old,
            dx=dx,
            dy=dy,
            g=g,
            h_floor=h_floor,
            bc=bc,
            limiter=limiter,
            riemann_flux=riemann_flux,
        ),
        h_floor=h_floor,
    )

    converged = False
    residual = np.inf
    iterations = 0
    total_linear_iterations = 0
    total_line_search_reductions = 0

    for k in range(1, max_iter + 1):
        G = implicit_residual(
            U_new=U,
            U_old=U_old,
            U_older=U_older,
            rhs_scale=rhs_scale,
            coeff_old=coeff_old,
            coeff_older=coeff_older,
            dx=dx,
            dy=dy,
            g=g,
            h_floor=h_floor,
            bc=bc,
            limiter=limiter,
            riemann_flux=riemann_flux,
        )
        G_flat = flatten_state(G)
        residual = _relative_nl_residual(G, U)
        iterations = k

        if verbose:
            print(f"    [NL] iter={k:3d} | {residual_name}={residual:.3e}")

        if residual <= tol:
            converged = True
            break

        U_flat = flatten_state(U)
        norm_u = max(float(np.linalg.norm(U_flat)), 1.0)

        def jvp(v_flat):
            v_flat = np.asarray(v_flat, dtype=np.float64).reshape(-1)
            v_norm = float(np.linalg.norm(v_flat))
            if v_norm <= 1e-14:
                return np.zeros_like(v_flat)
            eps = jac_eps * norm_u / v_norm
            U_pert = unflatten_state(U_flat + eps * v_flat, nx=nx, ny=ny)
            U_pert = positivity_fix(U_pert, h_floor=h_floor)
            G_pert = implicit_residual(
                U_new=U_pert,
                U_old=U_old,
                U_older=U_older,
                rhs_scale=rhs_scale,
                coeff_old=coeff_old,
                coeff_older=coeff_older,
                dx=dx,
                dy=dy,
                g=g,
                h_floor=h_floor,
                bc=bc,
                limiter=limiter,
                riemann_flux=riemann_flux,
            )
            return (flatten_state(G_pert) - G_flat) / eps

        linear_tol = min(0.5, max(1e-3, 0.5 * np.sqrt(max(residual, 1e-16))))
        delta_flat, _, lin_its, _ = _gmres_matrix_free(
            jvp=jvp,
            b=-G_flat,
            restart=gmres_restart,
            max_restarts=gmres_max_restarts,
            tol=linear_tol,
        )
        total_linear_iterations += int(lin_its)
        delta = unflatten_state(delta_flat, nx=nx, ny=ny)

        alpha = 1.0
        accepted = False
        best_U = U
        best_res = residual

        for _ in range(line_search_max_steps + 1):
            U_trial = positivity_fix(U + alpha * delta, h_floor=h_floor)
            G_trial = implicit_residual(
                U_new=U_trial,
                U_old=U_old,
                U_older=U_older,
                rhs_scale=rhs_scale,
                coeff_old=coeff_old,
                coeff_older=coeff_older,
                dx=dx,
                dy=dy,
                g=g,
                h_floor=h_floor,
                bc=bc,
                limiter=limiter,
                riemann_flux=riemann_flux,
            )
            res_trial = _relative_nl_residual(G_trial, U_trial)

            if res_trial < best_res:
                best_res = res_trial
                best_U = U_trial

            if res_trial <= (1.0 - line_search_c1 * alpha) * residual:
                U = U_trial
                accepted = True
                break

            alpha *= 0.5
            total_line_search_reductions += 1

        if not accepted:
            U = best_U
            if best_res >= residual:
                break

    return U, {
        "converged": bool(converged),
        "iterations": int(iterations),
        "residual": float(residual),
        "linear_iterations": int(total_linear_iterations),
        "line_search_reductions": int(total_line_search_reductions),
    }


def backward_euler_step_newton_krylov(
    U_old,
    dt,
    dx,
    dy,
    g,
    h_floor,
    bc,
    max_iter=25,
    tol=1e-8,
    gmres_restart=12,
    gmres_max_restarts=3,
    jac_eps=1e-7,
    line_search_max_steps=8,
    line_search_c1=1e-4,
    limiter="minmod",
    riemann_flux="hll",
    residual_name="be_res",
    verbose=False,
):
    """
    Implicit Backward-Euler step solved with matrix-free Newton-Krylov.
    """
    return _implicit_step_newton_krylov_generic(
        U_old=U_old,
        U_older=U_old,
        coeff_old=1.0,
        coeff_older=0.0,
        rhs_scale=float(dt),
        dx=dx,
        dy=dy,
        g=g,
        h_floor=h_floor,
        bc=bc,
        max_iter=max_iter,
        tol=tol,
        gmres_restart=gmres_restart,
        gmres_max_restarts=gmres_max_restarts,
        jac_eps=jac_eps,
        line_search_max_steps=line_search_max_steps,
        line_search_c1=line_search_c1,
        limiter=limiter,
        riemann_flux=riemann_flux,
        residual_name=residual_name,
        verbose=verbose,
    )


def bdf2_step_newton_krylov(
    U_old,
    U_older,
    dt,
    dt_prev,
    dx,
    dy,
    g,
    h_floor,
    bc,
    max_iter=25,
    tol=1e-8,
    gmres_restart=12,
    gmres_max_restarts=3,
    jac_eps=1e-7,
    line_search_max_steps=8,
    line_search_c1=1e-4,
    limiter="minmod",
    riemann_flux="hll",
    residual_name="bdf2_res",
    verbose=False,
):
    """
    Implicit BDF2 step solved with matrix-free Newton-Krylov.
    """
    coeff_old, coeff_older, rhs_scale = bdf2_coefficients(dt=dt, dt_prev=dt_prev)
    return _implicit_step_newton_krylov_generic(
        U_old=U_old,
        U_older=U_older,
        coeff_old=coeff_old,
        coeff_older=coeff_older,
        rhs_scale=rhs_scale,
        dx=dx,
        dy=dy,
        g=g,
        h_floor=h_floor,
        bc=bc,
        max_iter=max_iter,
        tol=tol,
        gmres_restart=gmres_restart,
        gmres_max_restarts=gmres_max_restarts,
        jac_eps=jac_eps,
        line_search_max_steps=line_search_max_steps,
        line_search_c1=line_search_c1,
        limiter=limiter,
        riemann_flux=riemann_flux,
        residual_name=residual_name,
        verbose=verbose,
    )


def total_mass(U, dx, dy):
    return float(np.sum(U[0]) * float(dx) * float(dy))


def relative_step_residual(U_new, U_old):
    """
    Relative step-update norm used as an explicit-solver progress monitor.

    This is NOT a Newton residual. It is:
        ||U^{n+1} - U^n|| / max(||U^{n+1}||, eps)
    """
    du = np.asarray(U_new, dtype=np.float64) - np.asarray(U_old, dtype=np.float64)
    num = float(np.linalg.norm(du.reshape(-1)))
    den = float(np.linalg.norm(np.asarray(U_new, dtype=np.float64).reshape(-1)))
    den = max(den, 1e-14)
    return num / den


def simulate_with_sampling(
    U0,
    sample_times,
    t_final,
    dx,
    dy,
    g,
    cfl,
    h_floor,
    bc,
    time_integrator="explicit_rk2",
    fixed_dt=None,
    implicit_nonlinear_solver="newton_krylov",
    implicit_max_iter=25,
    implicit_tol=1e-8,
    implicit_relaxation=1.0,
    implicit_verbose_iters=False,
    limiter="minmod",
    riemann_flux="hll",
    verbose=False,
    print_every=10,
):
    """
    Integrate the HDM and store states at fixed sample_times.

    Sampling uses linear interpolation between accepted RK2 steps so all
    parameters share the exact same snapshot time grid.
    """
    sample_times = np.asarray(sample_times, dtype=np.float64).reshape(-1)
    if sample_times.size == 0:
        raise ValueError("sample_times must contain at least one value.")
    if np.any(np.diff(sample_times) < 0.0):
        raise ValueError("sample_times must be sorted in ascending order.")

    t_final = float(t_final)
    if sample_times[0] < -1e-14:
        raise ValueError("sample_times cannot contain negative values.")
    if sample_times[-1] > t_final + 1e-12:
        raise ValueError(
            f"Last sample time ({sample_times[-1]}) exceeds t_final ({t_final})."
        )

    U = np.asarray(U0, dtype=np.float64).copy()
    nt = sample_times.size
    sampled_states = np.zeros((nt,) + U.shape, dtype=np.float64)

    t = 0.0
    step = 0
    sample_idx = 0

    solver_times = [0.0]
    solver_mass = [total_mass(U, dx=dx, dy=dy)]
    step_residuals = []
    nonlinear_iterations = []
    linear_iterations = []
    line_search_reductions = []
    nonlinear_converged = []
    initial_mass = solver_mass[0]

    time_integrator = str(time_integrator).strip().lower()
    if time_integrator not in {"explicit_rk2", "implicit_be", "implicit_bdf2"}:
        raise ValueError(
            f"Unknown time_integrator='{time_integrator}'. "
            "Use 'explicit_rk2', 'implicit_be', or 'implicit_bdf2'."
        )
    implicit_nonlinear_solver = str(implicit_nonlinear_solver).strip().lower()
    if implicit_nonlinear_solver not in {"picard", "newton_krylov"}:
        raise ValueError(
            f"Unknown implicit_nonlinear_solver='{implicit_nonlinear_solver}'. "
            "Use 'picard' or 'newton_krylov'."
        )
    limiter = str(limiter).strip().lower()
    if limiter not in {"minmod", "mc", "monotonized_central", "monotonized-central"}:
        raise ValueError(
            f"Unknown limiter='{limiter}'. Use 'minmod' or 'mc'."
        )
    if limiter in {"monotonized_central", "monotonized-central"}:
        limiter = "mc"
    riemann_flux = str(riemann_flux).strip().lower()
    if riemann_flux not in {"hll", "hllc"}:
        raise ValueError(
            f"Unknown riemann_flux='{riemann_flux}'. Use 'hll' or 'hllc'."
        )

    if fixed_dt is None:
        raise ValueError(
            "fixed_dt must be provided. Variable dt mode has been removed."
        )
    fixed_dt = float(fixed_dt)
    if fixed_dt <= 0.0:
        raise ValueError(f"fixed_dt must be > 0, got {fixed_dt}.")
    n_fixed_steps = int(np.rint(t_final / fixed_dt))
    if n_fixed_steps < 1:
        raise ValueError(
            f"fixed_dt={fixed_dt} is too large for t_final={t_final}."
        )
    tf_check = n_fixed_steps * fixed_dt
    tol_tf = 1e-12 * max(abs(t_final), 1.0)
    if abs(tf_check - t_final) > tol_tf:
        raise ValueError(
            f"fixed_dt={fixed_dt:.12e} does not divide t_final={t_final:.12e}. "
            f"Closest integer step count is {n_fixed_steps}, giving "
            f"n*fixed_dt={tf_check:.12e}. "
            "Choose fixed_dt so t_final/fixed_dt is an integer."
        )

    if verbose:
        print(
            f"[HDM] Starting integration | method={time_integrator} | "
            f"t_final={t_final:.6f} | target_samples={nt} | "
            f"fixed_dt={fixed_dt:.3e} | "
            f"implicit_solver={implicit_nonlinear_solver} | "
            f"limiter={limiter} | flux={riemann_flux}"
        )

    while sample_idx < nt and sample_times[sample_idx] <= 1e-14:
        sampled_states[sample_idx] = U
        sample_idx += 1

    # Previous accepted state/time-step needed for BDF2.
    U_older = None
    dt_prev = None

    while t < t_final - 1e-14:
        if step >= int(n_fixed_steps):
            break
        dt = fixed_dt

        U_prev = U
        t_prev = t

        if time_integrator == "explicit_rk2":
            U = rk2_step(
                U_prev,
                dt=dt,
                dx=dx,
                dy=dy,
                g=g,
                cfl=cfl,
                h_floor=h_floor,
                bc=bc,
                limiter=limiter,
                riemann_flux=riemann_flux,
            )
            step_res = relative_step_residual(U, U_prev)
            residual_name = "step_res"
            nl_iters = 1
            lin_iters = 0
            ls_reductions = 0
            nl_conv = True
        else:
            use_bdf2 = time_integrator == "implicit_bdf2" and step > 0
            residual_name = "bdf2_res" if use_bdf2 else "be_res"

            if implicit_nonlinear_solver == "newton_krylov":
                if use_bdf2:
                    if U_older is None or dt_prev is None:
                        raise RuntimeError(
                            "BDF2 requested but previous state/time-step is unavailable."
                        )
                    U, nl_info = bdf2_step_newton_krylov(
                        U_old=U_prev,
                        U_older=U_older,
                        dt=dt,
                        dt_prev=dt_prev,
                        dx=dx,
                        dy=dy,
                        g=g,
                        h_floor=h_floor,
                        bc=bc,
                        max_iter=implicit_max_iter,
                        tol=implicit_tol,
                        limiter=limiter,
                        riemann_flux=riemann_flux,
                        residual_name=residual_name,
                        verbose=(verbose and implicit_verbose_iters),
                    )
                else:
                    U, nl_info = backward_euler_step_newton_krylov(
                        U_old=U_prev,
                        dt=dt,
                        dx=dx,
                        dy=dy,
                        g=g,
                        h_floor=h_floor,
                        bc=bc,
                        max_iter=implicit_max_iter,
                        tol=implicit_tol,
                        limiter=limiter,
                        riemann_flux=riemann_flux,
                        residual_name=residual_name,
                        verbose=(verbose and implicit_verbose_iters),
                    )
            else:
                if use_bdf2:
                    if U_older is None or dt_prev is None:
                        raise RuntimeError(
                            "BDF2 requested but previous state/time-step is unavailable."
                        )
                    U, nl_info = bdf2_step_picard(
                        U_old=U_prev,
                        U_older=U_older,
                        dt=dt,
                        dt_prev=dt_prev,
                        dx=dx,
                        dy=dy,
                        g=g,
                        h_floor=h_floor,
                        bc=bc,
                        max_iter=implicit_max_iter,
                        tol=implicit_tol,
                        relaxation=implicit_relaxation,
                        limiter=limiter,
                        riemann_flux=riemann_flux,
                        residual_name=residual_name,
                        verbose=(verbose and implicit_verbose_iters),
                    )
                else:
                    U, nl_info = backward_euler_step_picard(
                        U_old=U_prev,
                        dt=dt,
                        dx=dx,
                        dy=dy,
                        g=g,
                        h_floor=h_floor,
                        bc=bc,
                        max_iter=implicit_max_iter,
                        tol=implicit_tol,
                        relaxation=implicit_relaxation,
                        limiter=limiter,
                        riemann_flux=riemann_flux,
                        residual_name=residual_name,
                        verbose=(verbose and implicit_verbose_iters),
                    )
            step_res = float(nl_info["residual"])
            nl_iters = int(nl_info["iterations"])
            lin_iters = int(nl_info.get("linear_iterations", 0))
            ls_reductions = int(nl_info.get("line_search_reductions", 0))
            nl_conv = bool(nl_info["converged"])
            if not nl_conv:
                raise RuntimeError(
                    f"Implicit nonlinear solve did not converge ({time_integrator}) at "
                    f"step={step + 1}, t={t_prev:.6f}, dt={dt:.3e}, "
                    f"{residual_name}={step_res:.3e}, solver={implicit_nonlinear_solver}. "
                    "Try increasing implicit_max_iter, relaxing implicit_tol, "
                    "reducing fixed_dt, or switching implicit_nonlinear_solver."
                )

        t = t_prev + dt
        step += 1
        step_residuals.append(step_res)
        nonlinear_iterations.append(int(nl_iters))
        linear_iterations.append(int(lin_iters))
        line_search_reductions.append(int(ls_reductions))
        nonlinear_converged.append(bool(nl_conv))

        if time_integrator == "implicit_bdf2":
            U_older = np.asarray(U_prev, dtype=np.float64).copy()
            dt_prev = float(dt)

        solver_times.append(t)
        current_mass = total_mass(U, dx=dx, dy=dy)
        solver_mass.append(current_mass)

        if verbose and (step == 1 or step % int(max(print_every, 1)) == 0 or t >= t_final - 1e-14):
            rel_mass_drift = (current_mass - initial_mass) / max(abs(initial_mass), 1e-14)
            if time_integrator == "explicit_rk2":
                print(
                    f"[HDM] step={step:5d} | t={t:.6f}/{t_final:.6f} | "
                    f"dt={dt:.3e} | step_res={step_res:.3e} | "
                    f"rel_mass_drift={rel_mass_drift:.3e}"
                )
            else:
                print(
                    f"[HDM] step={step:5d} | t={t:.6f}/{t_final:.6f} | "
                    f"dt={dt:.3e} | {residual_name}={step_res:.3e} | "
                    f"nl_its={nl_iters:2d} | lin_its={lin_iters:3d} | "
                    f"ls_red={ls_reductions:2d} | nl_conv={nl_conv} | "
                    f"rel_mass_drift={rel_mass_drift:.3e}"
                )

        while sample_idx < nt and sample_times[sample_idx] <= t + 1e-12:
            if dt <= 0.0:
                sampled_states[sample_idx] = U
            else:
                alpha = (sample_times[sample_idx] - t_prev) / dt
                alpha = float(np.clip(alpha, 0.0, 1.0))
                sampled_states[sample_idx] = (1.0 - alpha) * U_prev + alpha * U
            sample_idx += 1

    while sample_idx < nt:
        sampled_states[sample_idx] = U
        sample_idx += 1

    return {
        "sampled_states": sampled_states,
        "sample_times": sample_times,
        "solver_times": np.asarray(solver_times, dtype=np.float64),
        "solver_mass": np.asarray(solver_mass, dtype=np.float64),
        "step_residuals": np.asarray(step_residuals, dtype=np.float64),
        "nonlinear_iterations": np.asarray(nonlinear_iterations, dtype=np.int64),
        "linear_iterations": np.asarray(linear_iterations, dtype=np.int64),
        "line_search_reductions": np.asarray(
            line_search_reductions, dtype=np.int64
        ),
        "nonlinear_converged": np.asarray(nonlinear_converged, dtype=np.int64),
        "num_solver_steps": int(step),
        "time_integrator": time_integrator,
        "implicit_nonlinear_solver": implicit_nonlinear_solver,
        "limiter": limiter,
        "riemann_flux": riemann_flux,
        "fixed_dt": float(fixed_dt),
        "implicit_max_iter": int(implicit_max_iter),
        "implicit_tol": float(implicit_tol),
        "implicit_relaxation": float(implicit_relaxation),
        "implicit_nonconverged_steps": int(
            np.sum(np.asarray(nonlinear_converged, dtype=np.int64) == 0)
        ),
    }
