"""
High-level shallow-water workflow helpers.

This module intentionally mirrors the organization of the burgers workbench:
parameter grid helpers, snapshot caching, residual/RHS wrappers, and plotting.
"""

import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from .config import (
    BC_TWO_BUMPS,
    CFL,
    FIXED_DT,
    G,
    H_FLOOR,
    IMPLICIT_MAX_ITER,
    IMPLICIT_NONLINEAR_SOLVER,
    IMPLICIT_RELAXATION,
    IMPLICIT_TOL,
    LIMITER,
    LX,
    LY,
    MU1_RANGE,
    MU2_RANGE,
    NUM_TIME_SAMPLES,
    NX,
    NY,
    RIEMANN_FLUX,
    SAMPLES_PER_MU,
    SIGMA,
    SNAP_FOLDER,
    TIME_INTEGRATOR,
    T_FINAL,
    get_fixed_step_sample_times,
    make_grid,
)
from .scenarios import two_bumps_collision_initial_state
from .solver import (
    flatten_state,
    shallow_water_res2D,
    shallow_water_rhs2D,
    shallow_water_rhs2D_flat,
    simulate_with_sampling,
)


def get_snapshot_params(
    mu1_range=MU1_RANGE,
    mu2_range=MU2_RANGE,
    samples_per_mu=SAMPLES_PER_MU,
):
    """
    Cartesian training grid for (mu1, mu2) snapshot generation.
    """
    n = int(samples_per_mu)
    if n < 1:
        raise ValueError("samples_per_mu must be >= 1.")

    mu1_low, mu1_high = float(mu1_range[0]), float(mu1_range[1])
    mu2_low, mu2_high = float(mu2_range[0]), float(mu2_range[1])
    if mu1_low > mu1_high:
        raise ValueError(f"mu1_range must be increasing, got {mu1_range}.")
    if mu2_low > mu2_high:
        raise ValueError(f"mu2_range must be increasing, got {mu2_range}.")

    mu1_samples = np.linspace(mu1_low, mu1_high, n)
    mu2_samples = np.linspace(mu2_low, mu2_high, n)

    mu_samples = []
    for mu1 in mu1_samples:
        for mu2 in mu2_samples:
            mu_samples.append([float(mu1), float(mu2)])
    return mu_samples


def solver_cache_tag(
    time_integrator=TIME_INTEGRATOR,
    fixed_dt=FIXED_DT,
    implicit_nonlinear_solver=IMPLICIT_NONLINEAR_SOLVER,
    implicit_max_iter=IMPLICIT_MAX_ITER,
    implicit_tol=IMPLICIT_TOL,
    implicit_relaxation=IMPLICIT_RELAXATION,
    limiter=LIMITER,
    riemann_flux=RIEMANN_FLUX,
):
    method = str(time_integrator).strip().lower()
    lim = str(limiter).strip().lower()
    flux = str(riemann_flux).strip().lower()
    if fixed_dt is None:
        raise ValueError(
            "fixed_dt must be provided. Variable dt mode has been removed."
        )
    fixed_dt = float(fixed_dt)
    if fixed_dt <= 0.0:
        raise ValueError(f"fixed_dt must be > 0, got {fixed_dt}.")
    parts = [
        f"int_{method}",
        f"fixeddt_{fixed_dt:.8e}",
        f"lim_{lim}",
        f"flux_{flux}",
    ]
    if method in {"implicit_be", "implicit_bdf2"}:
        nls = str(implicit_nonlinear_solver).strip().lower()
        parts.append(f"nls_{nls}")
        parts.append(f"it_{int(implicit_max_iter)}")
        parts.append(f"tol_{float(implicit_tol):.1e}")
        if nls == "picard":
            parts.append(f"relax_{float(implicit_relaxation):.3f}")
    return "+".join(parts)


def param_to_snap_fn(
    mu,
    snap_folder=SNAP_FOLDER,
    suffix=".npy",
    solver_tag=None,
):
    """
    Snapshot filename associated with parameter vector mu=[mu1, mu2].
    """
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    if mu.size != 2:
        raise ValueError(f"Expected 2 parameters [mu1, mu2], got {mu.size}.")

    filename = f"mu1_{mu[0]:.3f}_mu2_{mu[1]:.3f}{suffix}"
    return os.path.join(snap_folder, filename)


def get_saved_params(snap_folder=SNAP_FOLDER):
    """
    Set of existing cached snapshot files in snap_folder.
    """
    return set(glob.glob(os.path.join(snap_folder, "**", "*.npy"), recursive=True))


def _build_initial_state_for_mu(mu, grid, sigma=SIGMA):
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    if mu.size != 2:
        raise ValueError(f"Expected mu=[mu1, mu2], got shape {mu.shape}.")

    return two_bumps_collision_initial_state(
        X=grid["X"],
        Y=grid["Y"],
        mu1=float(mu[0]),
        mu2=float(mu[1]),
        sigma=float(sigma),
    )


def state_matrix_from_sampled_states(sampled_states):
    """
    Convert sampled_states[nt, 3, nx, ny] to snapshot matrix [N, nt].
    """
    sampled_states = np.asarray(sampled_states, dtype=np.float64)
    nt = sampled_states.shape[0]
    return sampled_states.reshape(nt, -1).T


def extract_h_snapshots(state_snapshots, nx, ny):
    """
    Recover h snapshots from state matrix [N, nt] as [nt, nx, ny].
    """
    snaps = np.asarray(state_snapshots, dtype=np.float64)
    nxy = int(nx) * int(ny)
    if snaps.shape[0] < nxy:
        raise ValueError(
            f"state_snapshots has {snaps.shape[0]} rows, expected at least {nxy}."
        )
    h = snaps[:nxy, :].T.reshape(snaps.shape[1], int(nx), int(ny))
    return h


def mass_history_from_state_snapshots(state_snapshots, nx, ny, dx, dy):
    """
    Total mass history from state matrix [N, nt].
    """
    h = extract_h_snapshots(state_snapshots, nx=nx, ny=ny)
    return np.sum(h, axis=(1, 2)) * float(dx) * float(dy)


def compute_error(rom_snaps, hdm_snaps):
    """
    Relative error per time sample.
    """
    rom_snaps = np.asarray(rom_snaps, dtype=np.float64)
    hdm_snaps = np.asarray(hdm_snaps, dtype=np.float64)
    if rom_snaps.shape != hdm_snaps.shape:
        raise ValueError(
            f"Shape mismatch: rom_snaps={rom_snaps.shape}, hdm_snaps={hdm_snaps.shape}."
        )

    hdm_norm = np.sqrt(np.sum(hdm_snaps**2, axis=0))
    err_norm = np.sqrt(np.sum((rom_snaps - hdm_snaps) ** 2, axis=0))
    hdm_norm = np.where(hdm_norm > 0.0, hdm_norm, 1.0)
    rel_err = err_norm / hdm_norm
    return rel_err, float(np.mean(rel_err))


def run_two_bumps_case(
    mu,
    nx=NX,
    ny=NY,
    lx=LX,
    ly=LY,
    g=G,
    cfl=CFL,
    h_floor=H_FLOOR,
    t_final=T_FINAL,
    num_time_samples=NUM_TIME_SAMPLES,
    bc=None,
    sigma=SIGMA,
    time_integrator=TIME_INTEGRATOR,
    fixed_dt=FIXED_DT,
    implicit_nonlinear_solver=IMPLICIT_NONLINEAR_SOLVER,
    implicit_max_iter=IMPLICIT_MAX_ITER,
    implicit_tol=IMPLICIT_TOL,
    implicit_relaxation=IMPLICIT_RELAXATION,
    implicit_verbose_iters=False,
    limiter=LIMITER,
    riemann_flux=RIEMANN_FLUX,
    verbose=False,
    print_every=10,
):
    """
    Run one HDM case for parameter mu=[mu1, mu2].

    Snapshots are stored at every fixed time step (including t=0).
    """
    if bc is None:
        bc = dict(BC_TWO_BUMPS)
    else:
        bc = dict(bc)

    grid = make_grid(nx=nx, ny=ny, lx=lx, ly=ly)
    sample_times = get_fixed_step_sample_times(t_final=t_final, fixed_dt=fixed_dt)
    n_all_samples = int(sample_times.size)
    if num_time_samples is not None and int(num_time_samples) != n_all_samples and verbose:
        print(
            f"[HDM] num_time_samples={int(num_time_samples)} ignored. "
            f"Using all fixed-dt snapshots: {n_all_samples}."
        )
    U0 = _build_initial_state_for_mu(mu=mu, grid=grid, sigma=sigma)

    sim = simulate_with_sampling(
        U0=U0,
        sample_times=sample_times,
        t_final=t_final,
        dx=grid["dx"],
        dy=grid["dy"],
        g=g,
        cfl=cfl,
        h_floor=h_floor,
        bc=bc,
        time_integrator=time_integrator,
        fixed_dt=fixed_dt,
        implicit_nonlinear_solver=implicit_nonlinear_solver,
        implicit_max_iter=implicit_max_iter,
        implicit_tol=implicit_tol,
        implicit_relaxation=implicit_relaxation,
        implicit_verbose_iters=implicit_verbose_iters,
        limiter=limiter,
        riemann_flux=riemann_flux,
        verbose=verbose,
        print_every=print_every,
    )

    state_snapshots = state_matrix_from_sampled_states(sim["sampled_states"])
    return {
        "mu": np.asarray(mu, dtype=np.float64).reshape(-1),
        "state_snapshots": state_snapshots,
        "times": sim["sample_times"],
        "solver_times": sim["solver_times"],
        "solver_mass": sim["solver_mass"],
        "step_residuals": sim["step_residuals"],
        "nonlinear_iterations": sim["nonlinear_iterations"],
        "linear_iterations": sim.get("linear_iterations", np.asarray([], dtype=np.int64)),
        "line_search_reductions": sim.get(
            "line_search_reductions", np.asarray([], dtype=np.int64)
        ),
        "nonlinear_converged": sim["nonlinear_converged"],
        "num_solver_steps": sim["num_solver_steps"],
        "time_integrator": str(sim.get("time_integrator", time_integrator)),
        "implicit_nonlinear_solver": str(
            sim.get("implicit_nonlinear_solver", implicit_nonlinear_solver)
        ),
        "limiter": str(sim.get("limiter", limiter)),
        "riemann_flux": str(sim.get("riemann_flux", riemann_flux)),
        "fixed_dt": float(sim.get("fixed_dt", np.nan)),
        "implicit_max_iter": int(sim.get("implicit_max_iter", implicit_max_iter)),
        "implicit_tol": float(sim.get("implicit_tol", implicit_tol)),
        "implicit_relaxation": float(
            sim.get("implicit_relaxation", implicit_relaxation)
        ),
        "implicit_nonconverged_steps": int(sim.get("implicit_nonconverged_steps", 0)),
        "nx": int(grid["nx"]),
        "ny": int(grid["ny"]),
        "lx": float(grid["lx"]),
        "ly": float(grid["ly"]),
        "dx": float(grid["dx"]),
        "dy": float(grid["dy"]),
        "x": np.asarray(grid["x"], dtype=np.float64),
        "y": np.asarray(grid["y"], dtype=np.float64),
        "g": float(g),
        "cfl": float(cfl),
        "h_floor": float(h_floor),
        "t_final": float(t_final),
        "num_time_samples_requested": (
            int(num_time_samples) if num_time_samples is not None else None
        ),
        "num_time_samples_stored": int(sample_times.size),
        "sigma": float(sigma),
        "bc_x_low": str(bc["x_low"]),
        "bc_x_high": str(bc["x_high"]),
        "bc_y_low": str(bc["y_low"]),
        "bc_y_high": str(bc["y_high"]),
    }


def save_snapshot_bundle(path, case_data):
    """
    Persist one case dictionary to disk as .npy (dict payload).
    """
    out = {
        "mu": np.asarray(case_data["mu"], dtype=np.float64),
        "state_snapshots": np.asarray(case_data["state_snapshots"], dtype=np.float64),
        "times": np.asarray(case_data["times"], dtype=np.float64),
        "solver_times": np.asarray(case_data["solver_times"], dtype=np.float64),
        "solver_mass": np.asarray(case_data["solver_mass"], dtype=np.float64),
        "step_residuals": np.asarray(case_data["step_residuals"], dtype=np.float64),
        "nonlinear_iterations": np.asarray(
            case_data.get("nonlinear_iterations", []), dtype=np.int64
        ),
        "linear_iterations": np.asarray(
            case_data.get("linear_iterations", []), dtype=np.int64
        ),
        "line_search_reductions": np.asarray(
            case_data.get("line_search_reductions", []), dtype=np.int64
        ),
        "nonlinear_converged": np.asarray(
            case_data.get("nonlinear_converged", []), dtype=np.int64
        ),
        "num_solver_steps": int(case_data["num_solver_steps"]),
        "time_integrator": str(case_data.get("time_integrator", TIME_INTEGRATOR)),
        "implicit_nonlinear_solver": str(
            case_data.get("implicit_nonlinear_solver", IMPLICIT_NONLINEAR_SOLVER)
        ),
        "limiter": str(case_data.get("limiter", LIMITER)),
        "riemann_flux": str(case_data.get("riemann_flux", RIEMANN_FLUX)),
        "fixed_dt": float(case_data.get("fixed_dt", np.nan)),
        "implicit_max_iter": int(case_data.get("implicit_max_iter", IMPLICIT_MAX_ITER)),
        "implicit_tol": float(case_data.get("implicit_tol", IMPLICIT_TOL)),
        "implicit_relaxation": float(
            case_data.get("implicit_relaxation", IMPLICIT_RELAXATION)
        ),
        "implicit_nonconverged_steps": int(
            case_data.get("implicit_nonconverged_steps", 0)
        ),
        "nx": int(case_data["nx"]),
        "ny": int(case_data["ny"]),
        "lx": float(case_data["lx"]),
        "ly": float(case_data["ly"]),
        "dx": float(case_data["dx"]),
        "dy": float(case_data["dy"]),
        "x": np.asarray(case_data["x"], dtype=np.float64),
        "y": np.asarray(case_data["y"], dtype=np.float64),
        "g": float(case_data["g"]),
        "cfl": float(case_data["cfl"]),
        "h_floor": float(case_data["h_floor"]),
        "t_final": float(case_data["t_final"]),
        "num_time_samples_requested": (
            int(case_data["num_time_samples_requested"])
            if case_data.get("num_time_samples_requested") is not None
            else -1
        ),
        "num_time_samples_stored": int(
            case_data.get(
                "num_time_samples_stored",
                np.asarray(case_data["state_snapshots"], dtype=np.float64).shape[1],
            )
        ),
        "sigma": float(case_data["sigma"]),
        "bc_x_low": str(case_data["bc_x_low"]),
        "bc_x_high": str(case_data["bc_x_high"]),
        "bc_y_low": str(case_data["bc_y_low"]),
        "bc_y_high": str(case_data["bc_y_high"]),
        "simulation_elapsed_seconds": float(
            case_data.get("simulation_elapsed_seconds", np.nan)
        ),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, out, allow_pickle=True)


def load_snapshot_bundle(path):
    """
    Load one cached snapshot file (.npy dict payload or legacy .npz) and normalize types.
    """
    if path.endswith(".npy"):
        data = np.load(path, allow_pickle=True)
        data = data.item() if hasattr(data, "item") else data
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected dict payload in {path}, got type={type(data).__name__}."
            )
        nonlinear_converged = np.asarray(
            data.get("nonlinear_converged", []), dtype=np.int64
        )
        implicit_nonconverged_steps = int(
            data.get("implicit_nonconverged_steps", np.sum(nonlinear_converged == 0))
        )
        return {
            "mu": np.asarray(data["mu"], dtype=np.float64),
            "state_snapshots": np.asarray(data["state_snapshots"], dtype=np.float64),
            "times": np.asarray(data["times"], dtype=np.float64),
            "solver_times": np.asarray(data["solver_times"], dtype=np.float64),
            "solver_mass": np.asarray(data["solver_mass"], dtype=np.float64),
            "step_residuals": np.asarray(data.get("step_residuals", []), dtype=np.float64),
            "nonlinear_iterations": np.asarray(
                data.get("nonlinear_iterations", []), dtype=np.int64
            ),
            "linear_iterations": np.asarray(
                data.get("linear_iterations", []), dtype=np.int64
            ),
            "line_search_reductions": np.asarray(
                data.get("line_search_reductions", []), dtype=np.int64
            ),
            "nonlinear_converged": nonlinear_converged,
            "num_solver_steps": int(data["num_solver_steps"]),
            "time_integrator": str(data.get("time_integrator", TIME_INTEGRATOR)),
            "implicit_nonlinear_solver": str(
                data.get("implicit_nonlinear_solver", IMPLICIT_NONLINEAR_SOLVER)
            ),
            "limiter": str(data.get("limiter", LIMITER)),
            "riemann_flux": str(data.get("riemann_flux", RIEMANN_FLUX)),
            "fixed_dt": float(data.get("fixed_dt", np.nan)),
            "implicit_max_iter": int(data.get("implicit_max_iter", IMPLICIT_MAX_ITER)),
            "implicit_tol": float(data.get("implicit_tol", IMPLICIT_TOL)),
            "implicit_relaxation": float(
                data.get("implicit_relaxation", IMPLICIT_RELAXATION)
            ),
            "implicit_nonconverged_steps": implicit_nonconverged_steps,
            "nx": int(data["nx"]),
            "ny": int(data["ny"]),
            "lx": float(data["lx"]),
            "ly": float(data["ly"]),
            "dx": float(data["dx"]),
            "dy": float(data["dy"]),
            "x": np.asarray(data["x"], dtype=np.float64),
            "y": np.asarray(data["y"], dtype=np.float64),
            "g": float(data["g"]),
            "cfl": float(data["cfl"]),
            "h_floor": float(data["h_floor"]),
            "t_final": float(data["t_final"]),
            "num_time_samples_requested": (
                None
                if int(data.get("num_time_samples_requested", -1)) < 0
                else int(data.get("num_time_samples_requested", -1))
            ),
            "num_time_samples_stored": int(
                data.get(
                    "num_time_samples_stored",
                    np.asarray(data["state_snapshots"], dtype=np.float64).shape[1],
                )
            ),
            "sigma": float(data["sigma"]),
            "bc_x_low": str(data["bc_x_low"]),
            "bc_x_high": str(data["bc_x_high"]),
            "bc_y_low": str(data["bc_y_low"]),
            "bc_y_high": str(data["bc_y_high"]),
            "simulation_elapsed_seconds": float(
                data.get("simulation_elapsed_seconds", np.nan)
            ),
        }

    # Legacy .npz support.
    with np.load(path, allow_pickle=False) as data:
        nonlinear_converged = (
            np.asarray(data["nonlinear_converged"], dtype=np.int64)
            if "nonlinear_converged" in data.files
            else np.asarray([], dtype=np.int64)
        )
        implicit_nonconverged_steps = (
            int(np.asarray(data["implicit_nonconverged_steps"]).item())
            if "implicit_nonconverged_steps" in data.files
            else int(np.sum(nonlinear_converged == 0))
        )

        return {
            "mu": np.asarray(data["mu"], dtype=np.float64),
            "state_snapshots": np.asarray(data["state_snapshots"], dtype=np.float64),
            "times": np.asarray(data["times"], dtype=np.float64),
            "solver_times": np.asarray(data["solver_times"], dtype=np.float64),
            "solver_mass": np.asarray(data["solver_mass"], dtype=np.float64),
            "step_residuals": (
                np.asarray(data["step_residuals"], dtype=np.float64)
                if "step_residuals" in data.files
                else np.asarray([], dtype=np.float64)
            ),
            "nonlinear_iterations": (
                np.asarray(data["nonlinear_iterations"], dtype=np.int64)
                if "nonlinear_iterations" in data.files
                else np.asarray([], dtype=np.int64)
            ),
            "linear_iterations": (
                np.asarray(data["linear_iterations"], dtype=np.int64)
                if "linear_iterations" in data.files
                else np.asarray([], dtype=np.int64)
            ),
            "line_search_reductions": (
                np.asarray(data["line_search_reductions"], dtype=np.int64)
                if "line_search_reductions" in data.files
                else np.asarray([], dtype=np.int64)
            ),
            "nonlinear_converged": nonlinear_converged,
            "num_solver_steps": int(np.asarray(data["num_solver_steps"]).item()),
            "time_integrator": (
                str(np.asarray(data["time_integrator"]).item())
                if "time_integrator" in data.files
                else TIME_INTEGRATOR
            ),
            "implicit_nonlinear_solver": (
                str(np.asarray(data["implicit_nonlinear_solver"]).item())
                if "implicit_nonlinear_solver" in data.files
                else IMPLICIT_NONLINEAR_SOLVER
            ),
            "limiter": (
                str(np.asarray(data["limiter"]).item())
                if "limiter" in data.files
                else LIMITER
            ),
            "riemann_flux": (
                str(np.asarray(data["riemann_flux"]).item())
                if "riemann_flux" in data.files
                else RIEMANN_FLUX
            ),
            "fixed_dt": (
                float(np.asarray(data["fixed_dt"]).item())
                if "fixed_dt" in data.files
                else np.nan
            ),
            "implicit_max_iter": (
                int(np.asarray(data["implicit_max_iter"]).item())
                if "implicit_max_iter" in data.files
                else IMPLICIT_MAX_ITER
            ),
            "implicit_tol": (
                float(np.asarray(data["implicit_tol"]).item())
                if "implicit_tol" in data.files
                else IMPLICIT_TOL
            ),
            "implicit_relaxation": (
                float(np.asarray(data["implicit_relaxation"]).item())
                if "implicit_relaxation" in data.files
                else IMPLICIT_RELAXATION
            ),
            "implicit_nonconverged_steps": implicit_nonconverged_steps,
            "nx": int(np.asarray(data["nx"]).item()),
            "ny": int(np.asarray(data["ny"]).item()),
            "lx": float(np.asarray(data["lx"]).item()),
            "ly": float(np.asarray(data["ly"]).item()),
            "dx": float(np.asarray(data["dx"]).item()),
            "dy": float(np.asarray(data["dy"]).item()),
            "x": np.asarray(data["x"], dtype=np.float64),
            "y": np.asarray(data["y"], dtype=np.float64),
            "g": float(np.asarray(data["g"]).item()),
            "cfl": float(np.asarray(data["cfl"]).item()),
            "h_floor": float(np.asarray(data["h_floor"]).item()),
            "t_final": float(np.asarray(data["t_final"]).item()),
            "num_time_samples_requested": (
                int(np.asarray(data["num_time_samples_requested"]).item())
                if "num_time_samples_requested" in data.files
                else None
            ),
            "num_time_samples_stored": (
                int(np.asarray(data["num_time_samples_stored"]).item())
                if "num_time_samples_stored" in data.files
                else int(np.asarray(data["state_snapshots"]).shape[1])
            ),
            "sigma": float(np.asarray(data["sigma"]).item()),
            "bc_x_low": str(np.asarray(data["bc_x_low"]).item()),
            "bc_x_high": str(np.asarray(data["bc_x_high"]).item()),
            "bc_y_low": str(np.asarray(data["bc_y_low"]).item()),
            "bc_y_high": str(np.asarray(data["bc_y_high"]).item()),
            "simulation_elapsed_seconds": (
                float(np.asarray(data["simulation_elapsed_seconds"]).item())
                if "simulation_elapsed_seconds" in data.files
                else np.nan
            ),
        }


def load_or_compute_snaps(
    mu,
    nx=NX,
    ny=NY,
    lx=LX,
    ly=LY,
    g=G,
    cfl=CFL,
    h_floor=H_FLOOR,
    t_final=T_FINAL,
    num_time_samples=NUM_TIME_SAMPLES,
    snap_folder=SNAP_FOLDER,
    bc=None,
    sigma=SIGMA,
    force_recompute=False,
    time_integrator=TIME_INTEGRATOR,
    fixed_dt=FIXED_DT,
    implicit_nonlinear_solver=IMPLICIT_NONLINEAR_SOLVER,
    implicit_max_iter=IMPLICIT_MAX_ITER,
    implicit_tol=IMPLICIT_TOL,
    implicit_relaxation=IMPLICIT_RELAXATION,
    implicit_verbose_iters=False,
    limiter=LIMITER,
    riemann_flux=RIEMANN_FLUX,
    verbose=False,
    print_every=10,
):
    """
    Load snapshots for mu if cached; otherwise compute and save them.
    """
    if not os.path.exists(snap_folder):
        os.makedirs(snap_folder, exist_ok=True)

    snap_fn = param_to_snap_fn(mu, snap_folder=snap_folder)
    os.makedirs(os.path.dirname(snap_fn), exist_ok=True)

    if os.path.exists(snap_fn) and not force_recompute:
        print(f"Loading saved snaps for mu1={mu[0]}, mu2={mu[1]}")
        cached = load_snapshot_bundle(snap_fn)
        expected_n = int(
            get_fixed_step_sample_times(t_final=t_final, fixed_dt=fixed_dt).size
        )
        cached_n = int(np.asarray(cached["state_snapshots"], dtype=np.float64).shape[1])
        if cached_n != expected_n:
            print(
                f"Cached snapshot count ({cached_n}) does not match fixed-dt "
                f"requirement ({expected_n}). Recomputing."
            )
        else:
            nonconv = int(cached.get("implicit_nonconverged_steps", 0))
            if (
                str(cached.get("time_integrator", time_integrator)).strip().lower().startswith("implicit_")
                and nonconv > 0
            ):
                print(
                    f"[WARN] Cached implicit run has {nonconv} non-converged steps. "
                    "Using cached data (set force_recompute=True to recompute)."
                )
            cached["from_cache"] = True
            cached["snapshot_path"] = snap_fn
            return cached

    print(f"Computing new snaps for mu1={mu[0]}, mu2={mu[1]}")
    t0 = time.time()
    case = run_two_bumps_case(
        mu=mu,
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        g=g,
        cfl=cfl,
        h_floor=h_floor,
        t_final=t_final,
        num_time_samples=num_time_samples,
        bc=bc,
        sigma=sigma,
        time_integrator=time_integrator,
        fixed_dt=fixed_dt,
        implicit_nonlinear_solver=implicit_nonlinear_solver,
        implicit_max_iter=implicit_max_iter,
        implicit_tol=implicit_tol,
        implicit_relaxation=implicit_relaxation,
        implicit_verbose_iters=implicit_verbose_iters,
        limiter=limiter,
        riemann_flux=riemann_flux,
        verbose=verbose,
        print_every=print_every,
    )
    sim_elapsed = float(time.time() - t0)
    case["simulation_elapsed_seconds"] = sim_elapsed
    case["from_cache"] = False
    case["snapshot_path"] = snap_fn
    print(f"Elapsed time: {sim_elapsed:3.3e}")
    save_snapshot_bundle(snap_fn, case)
    return case


def plot_snaps(
    x,
    y,
    h_snapshots,
    snaps_to_plot,
    linewidth=2.0,
    color="black",
    linestyle="solid",
    label=None,
    fig_ax=None,
    h_limits=None,
):
    """
    Plot h midline slices (x-mid and y-mid), mirroring burgers.plot_snaps.
    """
    h_snapshots = np.asarray(h_snapshots, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if fig_ax is None:
        fig, (ax1, ax2) = plt.subplots(2, 1)
    else:
        fig, ax1, ax2 = fig_ax

    mid_x = x.size // 2
    mid_y = y.size // 2
    first_line = True

    for ind in snaps_to_plot:
        line_label = label if first_line else None
        first_line = False

        h = h_snapshots[int(ind)]

        ax1.plot(
            x,
            h[:, mid_y],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=line_label,
        )
        ax1.set_xlabel("x")
        ax1.set_ylabel(f"h(x, y={y[mid_y]:0.3f})")
        ax1.grid(True, alpha=0.3)
        if h_limits is not None:
            ax1.set_ylim(float(h_limits[0]), float(h_limits[1]))

        ax2.plot(
            y,
            h[mid_x, :],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=line_label,
        )
        ax2.set_xlabel("y")
        ax2.set_ylabel(f"h(x={x[mid_x]:0.3f}, y)")
        ax2.grid(True, alpha=0.3)
        if h_limits is not None:
            ax2.set_ylim(float(h_limits[0]), float(h_limits[1]))

    return fig, ax1, ax2


__all__ = [
    "compute_error",
    "extract_h_snapshots",
    "flatten_state",
    "get_saved_params",
    "get_snapshot_params",
    "load_or_compute_snaps",
    "mass_history_from_state_snapshots",
    "param_to_snap_fn",
    "plot_snaps",
    "run_two_bumps_case",
    "solver_cache_tag",
    "shallow_water_res2D",
    "shallow_water_rhs2D",
    "shallow_water_rhs2D_flat",
]
