#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run one shallow-water HDM case, save snapshots, and generate baseline plots.
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
    NUM_TIME_SAMPLES,
    NX,
    NY,
    PLOT_H_MAX,
    PLOT_H_MIN,
    RIEMANN_FLUX,
    RESULTS_DIR,
    SNAP_FOLDER,
    TIME_INTEGRATOR,
    T_FINAL,
)
from shallow_waters.core import (
    extract_h_snapshots,
    load_or_compute_snaps,
    mass_history_from_state_snapshots,
)
from shallow_waters.plotting import (
    plot_depth_maps_grid,
    plot_midline_spacetime,
    save_depth_movie_mp4,
    save_depth_movie3d_mp4,
    save_midline_slice_movie_mp4,
)
from shallow_waters.reporting import write_txt_report


def main(
    mu1=0.10,
    mu2=0.10,
    nx=NX,
    ny=NY,
    lx=LX,
    ly=LY,
    g=G,
    cfl=CFL,
    h_floor=H_FLOOR,
    t_final=T_FINAL,
    num_time_samples=NUM_TIME_SAMPLES,
    results_dir=RESULTS_DIR,
    snap_folder=SNAP_FOLDER,
    save_snaps=True,
    save_plot=True,
    save_slices=True,
    save_movie_mp4=True,
    movie_fps=24,
    movie_frame_stride=None,
    save_movie3d_mp4=True,
    movie3d_elev=28.0,
    movie3d_azim=-130.0,
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
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    mu = [float(mu1), float(mu2)]

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

    state_snaps = np.asarray(case["state_snapshots"], dtype=np.float64)
    times = np.asarray(case["times"], dtype=np.float64)
    x = np.asarray(case["x"], dtype=np.float64)
    y = np.asarray(case["y"], dtype=np.float64)
    nx_case = int(case["nx"])
    ny_case = int(case["ny"])

    h_snaps = extract_h_snapshots(state_snaps, nx=nx_case, ny=ny_case)
    mass_history = mass_history_from_state_snapshots(
        state_snaps, nx=nx_case, ny=ny_case, dx=case["dx"], dy=case["dy"]
    )
    rel_mass_drift = (mass_history - mass_history[0]) / mass_history[0]
    step_residuals = np.asarray(case.get("step_residuals", []), dtype=np.float64)
    nonlinear_iterations = np.asarray(
        case.get("nonlinear_iterations", []), dtype=np.int64
    )
    linear_iterations = np.asarray(case.get("linear_iterations", []), dtype=np.int64)
    line_search_reductions = np.asarray(
        case.get("line_search_reductions", []), dtype=np.int64
    )
    nonlinear_converged = np.asarray(
        case.get("nonlinear_converged", []), dtype=np.int64
    )
    solver_method = str(case.get("time_integrator", time_integrator))

    print(
        f"Elapsed HDM time: {simulation_elapsed:.3e} seconds "
        f"(source={'cache' if from_cache else 'new_compute'})"
    )
    print(f"Snapshot matrix shape: {state_snaps.shape}")
    print(f"Stored time samples: {times.size}")
    print(f"Time integrator: {solver_method}")
    print(
        f"Spatial numerics: limiter={case.get('limiter', limiter)}, "
        f"flux={case.get('riemann_flux', riemann_flux)}"
    )
    if step_residuals.size > 0:
        metric_name = "step_res" if solver_method == "explicit_rk2" else "imp_res"
        print(
            f"{metric_name} monitor: "
            f"last={step_residuals[-1]:.3e}, "
            f"max={np.max(step_residuals):.3e}, "
            f"mean={np.mean(step_residuals):.3e}"
        )
    if solver_method.startswith("implicit_") and nonlinear_iterations.size > 0:
        conv_rate = float(np.mean(nonlinear_converged > 0))
        nonconv_count = int(np.sum(nonlinear_converged == 0))
        implicit_solver = str(
            case.get("implicit_nonlinear_solver", implicit_nonlinear_solver)
        ).strip().lower()
        print(
            f"Nonlinear iterations ({solver_method}): "
            f"solver={implicit_solver} | "
            f"mean={np.mean(nonlinear_iterations):.2f}, "
            f"max={np.max(nonlinear_iterations)}, "
            f"convergence_rate={conv_rate:.3f}, "
            f"nonconverged_steps={nonconv_count}"
        )
        if implicit_solver == "newton_krylov" and linear_iterations.size > 0:
            print(
                f"Linear iterations (Newton-Krylov): "
                f"mean={np.mean(linear_iterations):.2f}, "
                f"max={np.max(linear_iterations)} | "
                f"line-search reductions total={int(np.sum(line_search_reductions))}"
            )

    snaps_path = None
    if save_snaps:
        snaps_path = os.path.join(
            results_dir,
            f"hdm_snaps_mu1_{mu[0]:.3f}_mu2_{mu[1]:.3f}.npy",
        )
        np.save(snaps_path, state_snaps)
        print(f"HDM snapshots saved to: {snaps_path}")

    fig_maps_path = None
    fig_slices_path = None
    fig_spacetime_path = None
    fig_mass_path = None
    movie_2d_mp4_path = None
    movie_slices_2d_mp4_path = None
    movie_3d_mp4_path = None

    if save_plot:
        fig_maps_path = os.path.join(
            results_dir,
            f"hdm_maps_mu1_{mu[0]:.3f}_mu2_{mu[1]:.3f}.png",
        )
        plot_depth_maps_grid(
            h_snapshots=h_snaps,
            times=times,
            lx=case["lx"],
            ly=case["ly"],
            out_path=fig_maps_path,
            title_prefix=(
                f"Shallow-water HDM maps (mu1={mu[0]:.3f}, mu2={mu[1]:.3f})"
            ),
            n_panels=6,
            h_limits=(plot_h_min, plot_h_max),
        )
        print(f"HDM depth maps saved to: {fig_maps_path}")

        fig_spacetime_path = os.path.join(
            results_dir,
            f"hdm_spacetime_mu1_{mu[0]:.3f}_mu2_{mu[1]:.3f}.png",
        )
        plot_midline_spacetime(
            h_snapshots=h_snaps,
            x=x,
            y=y,
            times=times,
            out_path=fig_spacetime_path,
            title_prefix=(
                f"Space-time midlines (mu1={mu[0]:.3f}, mu2={mu[1]:.3f})"
            ),
            h_limits=(plot_h_min, plot_h_max),
        )
        print(f"HDM space-time plot saved to: {fig_spacetime_path}")

        if save_slices:
            mid_x = x.size // 2
            mid_y = y.size // 2
            slice_inds = np.unique(np.linspace(0, times.size - 1, 4, dtype=int))
            fig2, axes = plt.subplots(
                4, 2, figsize=(12.0, 12.0), constrained_layout=True, sharex="col"
            )
            for row, idx in enumerate(slice_inds):
                ax_x = axes[row, 0]
                ax_y = axes[row, 1]

                ax_x.plot(x, h_snaps[idx, :, mid_y], linewidth=2.0, color="black")
                ax_y.plot(y, h_snaps[idx, mid_x, :], linewidth=2.0, color="black")

                ax_x.set_ylabel(f"t={times[idx]:.3f}\nh")
                ax_y.set_ylabel(f"t={times[idx]:.3f}\nh")
                ax_x.grid(True, alpha=0.35)
                ax_y.grid(True, alpha=0.35)
                ax_x.set_ylim(float(plot_h_min), float(plot_h_max))
                ax_y.set_ylim(float(plot_h_min), float(plot_h_max))

                if row == 0:
                    ax_x.set_title(f"x-slice: h(x, y={y[mid_y]:.3f})")
                    ax_y.set_title(f"y-slice: h(x={x[mid_x]:.3f}, y)")

            axes[-1, 0].set_xlabel("x")
            axes[-1, 1].set_xlabel("y")
            fig2.suptitle(
                f"Midline slices at 4 times (mu1={mu[0]:.3f}, mu2={mu[1]:.3f})",
                y=1.01,
            )
            fig_slices_path = os.path.join(
                results_dir,
                f"hdm_slices_mu1_{mu[0]:.3f}_mu2_{mu[1]:.3f}.png",
            )
            fig2.savefig(fig_slices_path, dpi=220, bbox_inches="tight")
            plt.close(fig2)
            print(f"HDM slice plot saved to: {fig_slices_path}")

        if save_movie_mp4:
            movie_2d_mp4_path = os.path.join(
                results_dir,
                f"hdm_movie2d_mu1_{mu[0]:.3f}_mu2_{mu[1]:.3f}.mp4",
            )
            try:
                save_depth_movie_mp4(
                    h_snapshots=h_snaps,
                    times=times,
                    lx=case["lx"],
                    ly=case["ly"],
                    out_path=movie_2d_mp4_path,
                    fps=movie_fps,
                    frame_stride=movie_frame_stride,
                    title_prefix=f"HDM mu1={mu[0]:.3f}, mu2={mu[1]:.3f}",
                    h_limits=(plot_h_min, plot_h_max),
                )
                print(f"HDM 2D MP4 movie saved to: {movie_2d_mp4_path}")
            except RuntimeError as exc:
                movie_2d_mp4_path = None
                print(f"[WARN] MP4 export skipped: {exc}")

            movie_slices_2d_mp4_path = os.path.join(
                results_dir,
                f"hdm_movie_slices2d_mu1_{mu[0]:.3f}_mu2_{mu[1]:.3f}.mp4",
            )
            try:
                save_midline_slice_movie_mp4(
                    h_snapshots=h_snaps,
                    x=x,
                    y=y,
                    times=times,
                    out_path=movie_slices_2d_mp4_path,
                    fps=movie_fps,
                    frame_stride=movie_frame_stride,
                    title_prefix=f"HDM slices mu1={mu[0]:.3f}, mu2={mu[1]:.3f}",
                    line_color="black",
                    line_label="HDM",
                    h_limits=(plot_h_min, plot_h_max),
                )
                print(f"HDM 2D slice MP4 movie saved to: {movie_slices_2d_mp4_path}")
            except RuntimeError as exc:
                movie_slices_2d_mp4_path = None
                print(f"[WARN] 2D slice MP4 export skipped: {exc}")

        if save_movie3d_mp4:
            movie_3d_mp4_path = os.path.join(
                results_dir,
                f"hdm_movie3d_mu1_{mu[0]:.3f}_mu2_{mu[1]:.3f}.mp4",
            )
            try:
                save_depth_movie3d_mp4(
                    h_snapshots=h_snaps,
                    x=x,
                    y=y,
                    times=times,
                    out_path=movie_3d_mp4_path,
                    fps=movie_fps,
                    frame_stride=movie_frame_stride,
                    elev=movie3d_elev,
                    azim=movie3d_azim,
                    title_prefix=f"HDM mu1={mu[0]:.3f}, mu2={mu[1]:.3f}",
                    h_limits=(plot_h_min, plot_h_max),
                )
                print(f"HDM 3D MP4 movie saved to: {movie_3d_mp4_path}")
            except RuntimeError as exc:
                movie_3d_mp4_path = None
                print(f"[WARN] 3D MP4 export skipped: {exc}")

        fig3, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(times, rel_mass_drift, color="black", linewidth=2.0)
        ax.set_xlabel("time")
        ax.set_ylabel("relative mass drift")
        ax.set_title("Mass conservation diagnostic")
        ax.grid(True, alpha=0.35)
        fig_mass_path = os.path.join(
            results_dir,
            f"hdm_mass_mu1_{mu[0]:.3f}_mu2_{mu[1]:.3f}.png",
        )
        fig3.savefig(fig_mass_path, dpi=220, bbox_inches="tight")
        plt.close(fig3)
        print(f"Mass-drift plot saved to: {fig_mass_path}")

    report_path = os.path.join(
        results_dir,
        f"fom_summary_mu1_{mu[0]:.3f}_mu2_{mu[1]:.3f}.txt",
    )
    write_txt_report(
        report_path,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("mu1", mu[0]),
                    ("mu2", mu[1]),
                ],
            ),
            (
                "configuration",
                [
                    ("save_snaps", save_snaps),
                    ("save_plot", save_plot),
                    ("save_slices", save_slices),
                    ("save_movie_mp4", save_movie_mp4),
                    ("movie_fps", movie_fps),
                    ("movie_frame_stride", movie_frame_stride),
                    ("save_movie3d_mp4", save_movie3d_mp4),
                    ("movie3d_elev", movie3d_elev),
                    ("movie3d_azim", movie3d_azim),
                    ("time_integrator", case.get("time_integrator", time_integrator)),
                    (
                        "implicit_nonlinear_solver",
                        case.get("implicit_nonlinear_solver", implicit_nonlinear_solver),
                    ),
                    ("limiter", case.get("limiter", limiter)),
                    ("riemann_flux", case.get("riemann_flux", riemann_flux)),
                    ("fixed_dt", case.get("fixed_dt", fixed_dt)),
                    ("implicit_max_iter", case.get("implicit_max_iter", implicit_max_iter)),
                    ("implicit_tol", case.get("implicit_tol", implicit_tol)),
                    ("implicit_relaxation", case.get("implicit_relaxation", implicit_relaxation)),
                    ("implicit_verbose_iters", implicit_verbose_iters),
                    ("force_recompute", force_recompute),
                    ("solver_verbose", solver_verbose),
                    ("solver_print_every", solver_print_every),
                    ("plot_h_min", plot_h_min),
                    ("plot_h_max", plot_h_max),
                    ("snap_folder", snap_folder),
                    ("snapshot_path", case.get("snapshot_path")),
                    ("loaded_from_cache", from_cache),
                ],
            ),
            (
                "discretization",
                [
                    ("nx", nx_case),
                    ("ny", ny_case),
                    ("lx", case["lx"]),
                    ("ly", case["ly"]),
                    ("dx", case["dx"]),
                    ("dy", case["dy"]),
                    ("g", case["g"]),
                    ("cfl", case["cfl"]),
                    ("h_floor", case["h_floor"]),
                    ("t_final", case["t_final"]),
                    ("num_time_samples", times.size),
                    ("num_solver_steps", case["num_solver_steps"]),
                    ("full_state_size", state_snaps.shape[0]),
                ],
            ),
            (
                "fom_timing",
                [
                    ("total_hdm_time_seconds", simulation_elapsed),
                    ("simulation_elapsed_seconds", simulation_elapsed),
                    ("load_or_compute_elapsed_seconds", elapsed),
                    (
                        "avg_hdm_time_per_solver_step_seconds",
                        simulation_elapsed / max(int(case["num_solver_steps"]), 1),
                    ),
                ],
            ),
            (
                "step_monitor",
                [
                    (
                        "step_residual_last",
                        step_residuals[-1] if step_residuals.size > 0 else None,
                    ),
                    (
                        "step_residual_max",
                        np.max(step_residuals) if step_residuals.size > 0 else None,
                    ),
                    (
                        "step_residual_mean",
                        np.mean(step_residuals) if step_residuals.size > 0 else None,
                    ),
                    (
                        "nonlinear_iterations_mean",
                        np.mean(nonlinear_iterations)
                        if nonlinear_iterations.size > 0
                        else None,
                    ),
                    (
                        "nonlinear_iterations_max",
                        np.max(nonlinear_iterations)
                        if nonlinear_iterations.size > 0
                        else None,
                    ),
                    (
                        "linear_iterations_mean",
                        np.mean(linear_iterations)
                        if linear_iterations.size > 0
                        else None,
                    ),
                    (
                        "linear_iterations_max",
                        np.max(linear_iterations)
                        if linear_iterations.size > 0
                        else None,
                    ),
                    (
                        "line_search_reductions_total",
                        int(np.sum(line_search_reductions))
                        if line_search_reductions.size > 0
                        else 0,
                    ),
                    (
                        "nonlinear_convergence_rate",
                        np.mean(nonlinear_converged > 0)
                        if nonlinear_converged.size > 0
                        else None,
                    ),
                    (
                        "nonlinear_nonconverged_steps",
                        int(np.sum(nonlinear_converged == 0))
                        if nonlinear_converged.size > 0
                        else 0,
                    ),
                ],
            ),
            (
                "mass_diagnostics",
                [
                    ("initial_mass", mass_history[0]),
                    ("final_mass", mass_history[-1]),
                    ("relative_mass_drift_final", rel_mass_drift[-1]),
                    ("relative_mass_drift_max_abs", np.max(np.abs(rel_mass_drift))),
                ],
            ),
            (
                "outputs",
                [
                    ("hdm_snapshots_npy", snaps_path),
                    ("hdm_maps_png", fig_maps_path),
                    ("hdm_spacetime_png", fig_spacetime_path),
                    ("hdm_slices_png", fig_slices_path),
                    ("hdm_mass_png", fig_mass_path),
                    ("hdm_movie2d_mp4", movie_2d_mp4_path),
                    ("hdm_movie_slices2d_mp4", movie_slices_2d_mp4_path),
                    ("hdm_movie3d_mp4", movie_3d_mp4_path),
                ],
            ),
        ],
    )
    print(f"FOM text summary saved to: {report_path}")
    return simulation_elapsed, case


if __name__ == "__main__":
    main()
