#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Precompute (or load) HDM snapshots for shallow-water training parameters.

Outputs:
  - Results/param_snaps/mu1_..._mu2_....npy
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
    FIXED_DT,
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
    PLOT_H_MAX,
    PLOT_H_MIN,
    RIEMANN_FLUX,
    RESULTS_DIR,
    SAMPLES_PER_MU,
    SNAP_FOLDER,
    TIME_INTEGRATOR,
    T_FINAL,
)
from shallow_waters.core import (
    extract_h_snapshots,
    get_snapshot_params,
    load_or_compute_snaps,
    param_to_snap_fn,
)
from shallow_waters.plotting import (
    plot_depth_maps_grid,
    plot_midline_spacetime,
    save_depth_movie3d_mp4,
    save_depth_movie_mp4,
    save_midline_slice_movie_mp4,
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


def save_training_case_slices_plot(
    h_snapshots,
    x,
    y,
    times,
    out_path,
    title_prefix="",
    h_limits=None,
):
    """
    Save 4x2 midline slices for one training case.
    """
    h_snapshots = np.asarray(h_snapshots, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    mid_x = x.size // 2
    mid_y = y.size // 2
    inds = np.unique(np.linspace(0, times.size - 1, 4, dtype=int))

    fig, axes = plt.subplots(4, 2, figsize=(12.0, 12.0), constrained_layout=True, sharex="col")
    for row, k in enumerate(inds):
        ax_x = axes[row, 0]
        ax_y = axes[row, 1]

        ax_x.plot(x, h_snapshots[k, :, mid_y], color="black", linewidth=2.0)
        ax_y.plot(y, h_snapshots[k, mid_x, :], color="black", linewidth=2.0)

        ax_x.set_ylabel(f"t={times[k]:.3f}\nh")
        ax_y.set_ylabel(f"t={times[k]:.3f}\nh")
        ax_x.grid(True, alpha=0.35)
        ax_y.grid(True, alpha=0.35)
        if h_limits is not None:
            ax_x.set_ylim(float(h_limits[0]), float(h_limits[1]))
            ax_y.set_ylim(float(h_limits[0]), float(h_limits[1]))

        if row == 0:
            ax_x.set_title(f"x-slice: h(x, y={y[mid_y]:.3f})")
            ax_y.set_title(f"y-slice: h(x={x[mid_x]:.3f}, y)")

    axes[-1, 0].set_xlabel("x")
    axes[-1, 1].set_xlabel("y")
    if title_prefix:
        fig.suptitle(title_prefix, y=1.01)
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
    training_visuals_dir=os.path.join(RESULTS_DIR, "fom_training_visuals"),
    save_case_maps=True,
    save_case_spacetime=True,
    save_case_slices=True,
    save_case_movie_mp4=True,
    save_case_slice_movie_mp4=True,
    save_case_movie3d_mp4=True,
    case_movie_fps=20,
    case_movie_frame_stride=None,
    case_movie3d_elev=28.0,
    case_movie3d_azim=-130.0,
    time_integrator=TIME_INTEGRATOR,
    fixed_dt=FIXED_DT,
    implicit_nonlinear_solver=IMPLICIT_NONLINEAR_SOLVER,
    implicit_max_iter=IMPLICIT_MAX_ITER,
    implicit_tol=IMPLICIT_TOL,
    implicit_relaxation=IMPLICIT_RELAXATION,
    implicit_verbose_iters=IMPLICIT_VERBOSE_ITERS,
    limiter=LIMITER,
    riemann_flux=RIEMANN_FLUX,
    force_recompute=False,
    solver_verbose=True,
    solver_print_every=1,
    plot_h_min=PLOT_H_MIN,
    plot_h_max=PLOT_H_MAX,
):
    os.makedirs(snap_folder, exist_ok=True)
    os.makedirs(os.path.dirname(report_file), exist_ok=True)

    save_case_visuals = any(
        [
            bool(save_case_maps),
            bool(save_case_spacetime),
            bool(save_case_slices),
            bool(save_case_movie_mp4),
            bool(save_case_slice_movie_mp4),
            bool(save_case_movie3d_mp4),
        ]
    )

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
    load_elapsed_list = []
    cached_flags = []
    shape_list = []
    step_list = []
    final_mass_list = []

    t_total0 = time.time()
    case_viz_root = training_visuals_dir
    if save_case_visuals:
        os.makedirs(case_viz_root, exist_ok=True)

    for mu in mu_list:
        mu = [float(mu[0]), float(mu[1])]
        snap_fn = param_to_snap_fn(mu, snap_folder=snap_folder)
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
            fixed_dt=fixed_dt,
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
        simulation_elapsed = float(case.get("simulation_elapsed_seconds", np.nan))
        if not np.isfinite(simulation_elapsed):
            simulation_elapsed = float(elapsed)
        from_cache = bool(case.get("from_cache", False))

        snaps = np.asarray(case["state_snapshots"], dtype=np.float64)

        params.append(mu)
        elapsed_list.append(float(simulation_elapsed))
        load_elapsed_list.append(float(elapsed))
        cached_flags.append(bool(was_cached))
        shape_list.append(list(snaps.shape))
        step_list.append(int(case["num_solver_steps"]))
        final_mass_list.append(float(case["solver_mass"][-1]))

        status = "cache" if was_cached else "computed"
        print(
            f"[FOM-TRAIN] mu=({mu[0]:.4f}, {mu[1]:.4f}) | {status} | "
            f"shape={snaps.shape} | steps={case['num_solver_steps']} | "
            f"sim_time={simulation_elapsed:.3e}s | "
            f"load_time={elapsed:.3e}s | "
            f"source={'cache' if from_cache else 'new_compute'}"
        )

        if save_case_visuals:
            times_case = np.asarray(case["times"], dtype=np.float64)
            x_case = np.asarray(case["x"], dtype=np.float64)
            y_case = np.asarray(case["y"], dtype=np.float64)
            h_snaps = extract_h_snapshots(
                snaps, nx=int(case["nx"]), ny=int(case["ny"])
            )
            mu_tag = f"mu1_{mu[0]:.3f}_mu2_{mu[1]:.3f}"

            if save_case_maps:
                p = os.path.join(case_viz_root, f"train_maps_{mu_tag}.png")
                plot_depth_maps_grid(
                    h_snapshots=h_snaps,
                    times=times_case,
                    lx=case["lx"],
                    ly=case["ly"],
                    out_path=p,
                    title_prefix=f"Training maps ({mu_tag})",
                    n_panels=6,
                    h_limits=(plot_h_min, plot_h_max),
                )

            if save_case_spacetime:
                p = os.path.join(case_viz_root, f"train_spacetime_{mu_tag}.png")
                plot_midline_spacetime(
                    h_snapshots=h_snaps,
                    x=x_case,
                    y=y_case,
                    times=times_case,
                    out_path=p,
                    title_prefix=f"Training space-time ({mu_tag})",
                    h_limits=(plot_h_min, plot_h_max),
                )

            if save_case_slices:
                p = os.path.join(case_viz_root, f"train_slices_{mu_tag}.png")
                save_training_case_slices_plot(
                    h_snapshots=h_snaps,
                    x=x_case,
                    y=y_case,
                    times=times_case,
                    out_path=p,
                    title_prefix=f"Training slices ({mu_tag})",
                    h_limits=(plot_h_min, plot_h_max),
                )

            if save_case_movie_mp4:
                p = os.path.join(case_viz_root, f"train_movie2d_{mu_tag}.mp4")
                try:
                    save_depth_movie_mp4(
                        h_snapshots=h_snaps,
                        times=times_case,
                        lx=case["lx"],
                        ly=case["ly"],
                        out_path=p,
                        fps=case_movie_fps,
                        frame_stride=case_movie_frame_stride,
                        title_prefix=f"Training 2D {mu_tag}",
                        h_limits=(plot_h_min, plot_h_max),
                    )
                except RuntimeError as exc:
                    print(f"[WARN] Training 2D MP4 export skipped ({mu_tag}): {exc}")

            if save_case_slice_movie_mp4:
                p = os.path.join(case_viz_root, f"train_movie_slices2d_{mu_tag}.mp4")
                try:
                    save_midline_slice_movie_mp4(
                        h_snapshots=h_snaps,
                        x=x_case,
                        y=y_case,
                        times=times_case,
                        out_path=p,
                        fps=case_movie_fps,
                        frame_stride=case_movie_frame_stride,
                        title_prefix=f"Training slices {mu_tag}",
                        line_color="black",
                        line_label="HDM",
                        h_limits=(plot_h_min, plot_h_max),
                    )
                except RuntimeError as exc:
                    print(f"[WARN] Training 2D slice MP4 export skipped ({mu_tag}): {exc}")

            if save_case_movie3d_mp4:
                p = os.path.join(case_viz_root, f"train_movie3d_{mu_tag}.mp4")
                try:
                    save_depth_movie3d_mp4(
                        h_snapshots=h_snaps,
                        x=x_case,
                        y=y_case,
                        times=times_case,
                        out_path=p,
                        fps=case_movie_fps,
                        frame_stride=case_movie_frame_stride,
                        elev=case_movie3d_elev,
                        azim=case_movie3d_azim,
                        title_prefix=f"Training 3D {mu_tag}",
                        h_limits=(plot_h_min, plot_h_max),
                    )
                except RuntimeError as exc:
                    print(f"[WARN] Training 3D MP4 export skipped ({mu_tag}): {exc}")

    elapsed_total = time.time() - t_total0

    params_arr = np.asarray(params, dtype=np.float64)
    elapsed_arr = np.asarray(elapsed_list, dtype=np.float64)
    load_elapsed_arr = np.asarray(load_elapsed_list, dtype=np.float64)
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
        load_elapsed_seconds=load_elapsed_arr,
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
        fixed_dt=np.asarray(float(fixed_dt), dtype=np.float64),
        implicit_max_iter=np.asarray(int(implicit_max_iter), dtype=np.int64),
        implicit_tol=np.asarray(float(implicit_tol), dtype=np.float64),
        implicit_relaxation=np.asarray(float(implicit_relaxation), dtype=np.float64),
        save_case_maps=np.asarray(int(save_case_maps), dtype=np.int64),
        save_case_spacetime=np.asarray(int(save_case_spacetime), dtype=np.int64),
        save_case_slices=np.asarray(int(save_case_slices), dtype=np.int64),
        save_case_movie_mp4=np.asarray(int(save_case_movie_mp4), dtype=np.int64),
        save_case_slice_movie_mp4=np.asarray(
            int(save_case_slice_movie_mp4), dtype=np.int64
        ),
        save_case_movie3d_mp4=np.asarray(int(save_case_movie3d_mp4), dtype=np.int64),
        case_movie_fps=np.asarray(int(case_movie_fps), dtype=np.int64),
        case_movie_frame_stride=np.asarray(
            -1 if case_movie_frame_stride is None else int(case_movie_frame_stride),
            dtype=np.int64,
        ),
        case_movie3d_elev=np.asarray(float(case_movie3d_elev), dtype=np.float64),
        case_movie3d_azim=np.asarray(float(case_movie3d_azim), dtype=np.float64),
        plot_h_min=np.asarray(float(plot_h_min), dtype=np.float64),
        plot_h_max=np.asarray(float(plot_h_max), dtype=np.float64),
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
                    ("fixed_dt", fixed_dt),
                    ("implicit_max_iter", implicit_max_iter),
                    ("implicit_tol", implicit_tol),
                    ("implicit_relaxation", implicit_relaxation),
                    ("implicit_verbose_iters", implicit_verbose_iters),
                    ("solver_verbose", solver_verbose),
                    ("solver_print_every", solver_print_every),
                    ("save_param_space_plot", save_param_space_plot),
                    ("param_space_plot_file", param_space_plot_file),
                    ("training_visuals_dir", case_viz_root if save_case_visuals else None),
                    ("save_case_maps", save_case_maps),
                    ("save_case_spacetime", save_case_spacetime),
                    ("save_case_slices", save_case_slices),
                    ("save_case_movie_mp4", save_case_movie_mp4),
                    ("save_case_slice_movie_mp4", save_case_slice_movie_mp4),
                    ("save_case_movie3d_mp4", save_case_movie3d_mp4),
                    ("case_movie_fps", case_movie_fps),
                    ("case_movie_frame_stride", case_movie_frame_stride),
                    ("case_movie3d_elev", case_movie3d_elev),
                    ("case_movie3d_azim", case_movie3d_azim),
                    ("plot_h_min", plot_h_min),
                    ("plot_h_max", plot_h_max),
                ],
            ),
            (
                "results",
                [
                    ("num_loaded_from_cache", n_cached),
                    ("num_computed_new", n_computed),
                    ("total_time_seconds", float(np.sum(elapsed_arr))),
                    ("total_load_or_compute_seconds", elapsed_total),
                    ("mean_time_per_parameter_seconds", float(np.mean(elapsed_arr))),
                    ("max_time_per_parameter_seconds", float(np.max(elapsed_arr))),
                    ("min_time_per_parameter_seconds", float(np.min(elapsed_arr))),
                    ("mean_load_or_compute_seconds", float(np.mean(load_elapsed_arr))),
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
                    ("training_visuals_dir", case_viz_root if save_case_visuals else None),
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
