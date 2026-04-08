#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Precompute (or load) HDM snapshots for shallow-water training parameters.

Outputs:
  - Results/param_snaps/*.npz
  - Results/fom_training_metadata.npz
  - Results/fom_training_summary.txt
"""

import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from shallow_waters.config import (
    CFL,
    DT_MULTIPLIER,
    G,
    H_FLOOR,
    IMPLICIT_MAX_ITER,
    IMPLICIT_NONLINEAR_SOLVER,
    IMPLICIT_RELAXATION,
    IMPLICIT_TOL,
    IMPLICIT_VERBOSE_ITERS,
    LIMITER,
    LX,
    LY,
    MU1_RANGE,
    MU2_RANGE,
    NUM_TIME_SAMPLES,
    NX,
    NY,
    RIEMANN_FLUX,
    RESULTS_DIR,
    SAMPLES_PER_MU,
    SNAP_FOLDER,
    TIME_INTEGRATOR,
    T_FINAL,
)
from shallow_waters.core import (
    get_snapshot_params,
    load_or_compute_snaps,
    param_to_snap_fn,
    solver_cache_tag,
)
from shallow_waters.reporting import write_txt_report


def save_training_param_space_plot(
    mu_list,
    mu1_range,
    mu2_range,
    out_path,
):
    """
    Save a figure of sampled training parameters in (mu1, mu2) space.
    """
    pts = np.asarray(mu_list, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0:
        raise ValueError("mu_list is empty; cannot plot parameter space.")

    mu1_low = float(mu1_range[0])
    mu1_high = float(mu1_range[1])
    mu2_low = float(mu2_range[0])
    mu2_high = float(mu2_range[1])

    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)

    # Draw the training box.
    box_x = [mu1_low, mu1_high, mu1_high, mu1_low, mu1_low]
    box_y = [mu2_low, mu2_low, mu2_high, mu2_high, mu2_low]
    ax.plot(box_x, box_y, color="tab:blue", linewidth=1.8, label="training box")

    # Plot sampled points.
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        color="black",
        s=45,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.5,
        label=f"samples ({pts.shape[0]})",
    )

    # Keep view tight with a small margin.
    xpad = 0.04 * max(mu1_high - mu1_low, 1e-12)
    ypad = 0.04 * max(mu2_high - mu2_low, 1e-12)
    ax.set_xlim(mu1_low - xpad, mu1_high + xpad)
    ax.set_ylim(mu2_low - ypad, mu2_high + ypad)

    ax.set_xlabel("mu1 (left bump amplitude)")
    ax.set_ylabel("mu2 (right bump amplitude)")
    ax.set_title("FOM training parameter space")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", frameon=True)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(
    mu1_range=MU1_RANGE,
    mu2_range=MU2_RANGE,
    samples_per_mu=SAMPLES_PER_MU,
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
    report_file=os.path.join(RESULTS_DIR, "fom_training_summary.txt"),
    metadata_file=os.path.join(RESULTS_DIR, "fom_training_metadata.npz"),
    param_space_plot_file=os.path.join(RESULTS_DIR, "fom_training_param_space.png"),
    save_param_space_plot=True,
    time_integrator=TIME_INTEGRATOR,
    dt_multiplier=DT_MULTIPLIER,
    implicit_nonlinear_solver=IMPLICIT_NONLINEAR_SOLVER,
    implicit_max_iter=IMPLICIT_MAX_ITER,
    implicit_tol=IMPLICIT_TOL,
    implicit_relaxation=IMPLICIT_RELAXATION,
    implicit_verbose_iters=IMPLICIT_VERBOSE_ITERS,
    limiter=LIMITER,
    riemann_flux=RIEMANN_FLUX,
    force_recompute=False,
    solver_verbose=False,
    solver_print_every=25,
):
    os.makedirs(snap_folder, exist_ok=True)
    os.makedirs(os.path.dirname(report_file), exist_ok=True)

    mu_list = get_snapshot_params(
        mu1_range=mu1_range,
        mu2_range=mu2_range,
        samples_per_mu=samples_per_mu,
    )
    if len(mu_list) == 0:
        raise RuntimeError("get_snapshot_params() returned an empty parameter set.")

    param_space_plot_path = None
    if save_param_space_plot:
        os.makedirs(os.path.dirname(param_space_plot_file), exist_ok=True)
        save_training_param_space_plot(
            mu_list=mu_list,
            mu1_range=mu1_range,
            mu2_range=mu2_range,
            out_path=param_space_plot_file,
        )
        param_space_plot_path = param_space_plot_file

    print("\n====================================================")
    print("      SHALLOW-WATER FOM TRAINING SNAPSHOTS")
    print("====================================================")
    print(f"[FOM-TRAIN] Number of training parameters: {len(mu_list)}")
    print(
        f"[FOM-TRAIN] method={time_integrator} | "
        f"implicit_solver={implicit_nonlinear_solver} | "
        f"limiter={limiter} | flux={riemann_flux}"
    )

    params = []
    elapsed_list = []
    cached_flags = []
    shape_list = []
    step_list = []
    final_mass_list = []

    t_total0 = time.time()
    tag = solver_cache_tag(
        time_integrator=time_integrator,
        dt_multiplier=dt_multiplier,
        implicit_nonlinear_solver=implicit_nonlinear_solver,
        implicit_max_iter=implicit_max_iter,
        implicit_tol=implicit_tol,
        implicit_relaxation=implicit_relaxation,
        limiter=limiter,
        riemann_flux=riemann_flux,
    )
    for mu in mu_list:
        mu = [float(mu[0]), float(mu[1])]
        snap_fn = param_to_snap_fn(mu, snap_folder=snap_folder, solver_tag=tag)
        was_cached = os.path.exists(snap_fn) and (not force_recompute)

        t0 = time.time()
        case = load_or_compute_snaps(
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
            snap_folder=snap_folder,
            force_recompute=force_recompute,
            time_integrator=time_integrator,
            dt_multiplier=dt_multiplier,
            implicit_nonlinear_solver=implicit_nonlinear_solver,
            implicit_max_iter=implicit_max_iter,
            implicit_tol=implicit_tol,
            implicit_relaxation=implicit_relaxation,
            implicit_verbose_iters=implicit_verbose_iters,
            limiter=limiter,
            riemann_flux=riemann_flux,
            verbose=solver_verbose,
            print_every=solver_print_every,
        )
        elapsed = time.time() - t0

        snaps = np.asarray(case["state_snapshots"], dtype=np.float64)

        params.append(mu)
        elapsed_list.append(float(elapsed))
        cached_flags.append(bool(was_cached))
        shape_list.append(list(snaps.shape))
        step_list.append(int(case["num_solver_steps"]))
        final_mass_list.append(float(case["solver_mass"][-1]))

        status = "cache" if was_cached else "computed"
        print(
            f"[FOM-TRAIN] mu=({mu[0]:.4f}, {mu[1]:.4f}) | {status} | "
            f"shape={snaps.shape} | steps={case['num_solver_steps']} | "
            f"time={elapsed:.3e}s"
        )

    elapsed_total = time.time() - t_total0

    params_arr = np.asarray(params, dtype=np.float64)
    elapsed_arr = np.asarray(elapsed_list, dtype=np.float64)
    cached_arr = np.asarray(cached_flags, dtype=np.int64)
    shape_arr = np.asarray(shape_list, dtype=np.int64)
    step_arr = np.asarray(step_list, dtype=np.int64)
    mass_arr = np.asarray(final_mass_list, dtype=np.float64)

    n_cached = int(np.sum(cached_arr))
    n_computed = int(cached_arr.size - n_cached)

    np.savez(
        metadata_file,
        params=params_arr,
        elapsed_seconds=elapsed_arr,
        was_cached=cached_arr,
        snapshot_shapes=shape_arr,
        num_solver_steps=step_arr,
        final_mass=mass_arr,
        mu1_range=np.asarray(mu1_range, dtype=np.float64),
        mu2_range=np.asarray(mu2_range, dtype=np.float64),
        samples_per_mu=np.asarray(int(samples_per_mu), dtype=np.int64),
        nx=np.asarray(int(nx), dtype=np.int64),
        ny=np.asarray(int(ny), dtype=np.int64),
        lx=np.asarray(float(lx), dtype=np.float64),
        ly=np.asarray(float(ly), dtype=np.float64),
        g=np.asarray(float(g), dtype=np.float64),
        cfl=np.asarray(float(cfl), dtype=np.float64),
        h_floor=np.asarray(float(h_floor), dtype=np.float64),
        t_final=np.asarray(float(t_final), dtype=np.float64),
        num_time_samples=np.asarray(int(num_time_samples), dtype=np.int64),
        time_integrator=np.asarray(str(time_integrator)),
        implicit_nonlinear_solver=np.asarray(str(implicit_nonlinear_solver)),
        limiter=np.asarray(str(limiter)),
        riemann_flux=np.asarray(str(riemann_flux)),
        dt_multiplier=np.asarray(float(dt_multiplier), dtype=np.float64),
        implicit_max_iter=np.asarray(int(implicit_max_iter), dtype=np.int64),
        implicit_tol=np.asarray(float(implicit_tol), dtype=np.float64),
        implicit_relaxation=np.asarray(float(implicit_relaxation), dtype=np.float64),
    )

    write_txt_report(
        report_file,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "run_fom_training.py"),
                ],
            ),
            (
                "configuration",
                [
                    ("snap_folder", snap_folder),
                    ("mu1_range", mu1_range),
                    ("mu2_range", mu2_range),
                    ("samples_per_mu", samples_per_mu),
                    ("nx", nx),
                    ("ny", ny),
                    ("t_final", t_final),
                    ("num_time_samples", num_time_samples),
                    ("time_integrator", time_integrator),
                    ("implicit_nonlinear_solver", implicit_nonlinear_solver),
                    ("limiter", limiter),
                    ("riemann_flux", riemann_flux),
                    ("dt_multiplier", dt_multiplier),
                    ("implicit_max_iter", implicit_max_iter),
                    ("implicit_tol", implicit_tol),
                    ("implicit_relaxation", implicit_relaxation),
                    ("implicit_verbose_iters", implicit_verbose_iters),
                    ("solver_verbose", solver_verbose),
                    ("solver_print_every", solver_print_every),
                    ("save_param_space_plot", save_param_space_plot),
                    ("param_space_plot_file", param_space_plot_file),
                ],
            ),
            (
                "results",
                [
                    ("num_loaded_from_cache", n_cached),
                    ("num_computed_new", n_computed),
                    ("total_time_seconds", elapsed_total),
                    ("mean_time_per_parameter_seconds", float(np.mean(elapsed_arr))),
                    ("max_time_per_parameter_seconds", float(np.max(elapsed_arr))),
                    ("min_time_per_parameter_seconds", float(np.min(elapsed_arr))),
                    ("snapshot_shape_example", shape_arr[0].tolist()),
                    ("mean_solver_steps", float(np.mean(step_arr))),
                ],
            ),
            (
                "outputs",
                [
                    ("metadata_npz", metadata_file),
                    ("summary_txt", report_file),
                    ("param_space_plot_png", param_space_plot_path),
                ],
            ),
        ],
    )

    if param_space_plot_path is not None:
        print(f"[FOM-TRAIN] Parameter-space plot saved: {param_space_plot_path}")
    print(f"[FOM-TRAIN] Metadata saved: {metadata_file}")
    print(f"[FOM-TRAIN] Summary saved: {report_file}")
    return elapsed_total, n_computed


if __name__ == "__main__":
    main()
