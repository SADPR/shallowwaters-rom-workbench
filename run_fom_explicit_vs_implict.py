#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run and compare shallow-water HDM with:
  - explicit SSP-RK2
  - implicit Backward Euler (configurable nonlinear solver)

Outputs are written to Results/explicit_vs_implict by default.
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
    plot_comparison_time_errors,
    save_depth_movie3d_mp4,
    save_depth_movie_mp4,
    save_midline_slice_comparison_movie_mp4,
)
from shallow_waters.reporting import write_txt_report


def _format_mu(mu1, mu2):
    return f"mu1_{mu1:.3f}_mu2_{mu2:.3f}"


def _rel_l2_over_time(ref, model):
    ref = np.asarray(ref, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)
    ref_norm = np.linalg.norm(ref.reshape(ref.shape[0], -1), axis=1)
    err = np.linalg.norm((model - ref).reshape(ref.shape[0], -1), axis=1)
    ref_norm = np.where(ref_norm > 0.0, ref_norm, 1.0)
    return err / ref_norm


def _save_mass_comparison(times, mass_exp, mass_imp, out_path, implicit_label):
    t = np.asarray(times, dtype=np.float64)
    m0e = max(abs(float(mass_exp[0])), 1e-14)
    m0i = max(abs(float(mass_imp[0])), 1e-14)
    drift_e = (np.asarray(mass_exp, dtype=np.float64) - mass_exp[0]) / m0e
    drift_i = (np.asarray(mass_imp, dtype=np.float64) - mass_imp[0]) / m0i

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(t, drift_e, color="black", linewidth=2.0, label="explicit_rk2")
    ax.plot(
        t,
        drift_i,
        color="tab:red",
        linewidth=2.0,
        linestyle="--",
        label=str(implicit_label),
    )
    ax.set_xlabel("time")
    ax.set_ylabel("relative mass drift")
    ax.set_title("Mass drift comparison")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_final_state_comparison(h_exp, h_imp, lx, ly, out_path, implicit_label):
    h_exp = np.asarray(h_exp, dtype=np.float64)
    h_imp = np.asarray(h_imp, dtype=np.float64)
    diff = h_imp - h_exp

    vmin = float(min(np.min(h_exp), np.min(h_imp)))
    vmax = float(max(np.max(h_exp), np.max(h_imp)))
    dmax = float(np.max(np.abs(diff)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    im0 = axes[0].imshow(
        h_exp.T, origin="lower", extent=[0.0, float(lx), 0.0, float(ly)], vmin=vmin, vmax=vmax
    )
    axes[0].set_title("explicit_rk2")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    im1 = axes[1].imshow(
        h_imp.T, origin="lower", extent=[0.0, float(lx), 0.0, float(ly)], vmin=vmin, vmax=vmax
    )
    axes[1].set_title(str(implicit_label))
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    im2 = axes[2].imshow(
        diff.T,
        origin="lower",
        extent=[0.0, float(lx), 0.0, float(ly)],
        vmin=-dmax,
        vmax=dmax,
        cmap="coolwarm",
    )
    axes[2].set_title("implicit - explicit")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    fig.colorbar(im1, ax=axes[:2], shrink=0.9, label="h")
    fig.colorbar(im2, ax=axes[2], shrink=0.9, label="delta h")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_slice_comparison(h_exp, h_imp, x, y, times, out_path, implicit_label):
    """
    Save 4x2 slice comparison figure:
      - column 1: x-midline slices
      - column 2: y-midline slices
      - each row: one representative time
    """
    h_exp = np.asarray(h_exp, dtype=np.float64)
    h_imp = np.asarray(h_imp, dtype=np.float64)
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

        ax_x.plot(x, h_exp[k, :, mid_y], color="black", linewidth=2.0, label="explicit_rk2")
        ax_x.plot(
            x,
            h_imp[k, :, mid_y],
            color="tab:red",
            linestyle="--",
            linewidth=2.0,
            label=str(implicit_label),
        )

        ax_y.plot(y, h_exp[k, mid_x, :], color="black", linewidth=2.0, label="explicit_rk2")
        ax_y.plot(
            y,
            h_imp[k, mid_x, :],
            color="tab:red",
            linestyle="--",
            linewidth=2.0,
            label=str(implicit_label),
        )

        ax_x.set_ylabel(f"t={times[k]:.3f}\nh")
        ax_y.set_ylabel(f"t={times[k]:.3f}\nh")
        ax_x.grid(True, alpha=0.35)
        ax_y.grid(True, alpha=0.35)

        if row == 0:
            ax_x.set_title(f"x-slice: h(x, y={y[mid_y]:.3f})")
            ax_y.set_title(f"y-slice: h(x={x[mid_x]:.3f}, y)")
            ax_x.legend(loc="best", frameon=True, fontsize=9)
            ax_y.legend(loc="best", frameon=True, fontsize=9)

    axes[-1, 0].set_xlabel("x")
    axes[-1, 1].set_xlabel("y")
    fig.suptitle("Explicit vs implicit slice comparison", y=1.01)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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
    explicit_fixed_dt=FIXED_DT,
    implicit_time_integrator=TIME_INTEGRATOR,
    implicit_fixed_dt=FIXED_DT,
    implicit_nonlinear_solver=IMPLICIT_NONLINEAR_SOLVER,
    implicit_max_iter=IMPLICIT_MAX_ITER,
    implicit_tol=IMPLICIT_TOL,
    implicit_relaxation=IMPLICIT_RELAXATION,
    implicit_verbose_iters=IMPLICIT_VERBOSE_ITERS,
    limiter=LIMITER,
    riemann_flux=RIEMANN_FLUX,
    results_dir=os.path.join(RESULTS_DIR, "explicit_vs_implict"),
    snap_folder=SNAP_FOLDER,
    save_snapshots=True,
    save_plot=True,
    save_slices=True,
    save_movie_mp4=True,
    save_movie3d_mp4=True,
    movie_fps=20,
    movie_frame_stride=None,
    movie3d_elev=28.0,
    movie3d_azim=-130.0,
    force_recompute=False,
    solver_verbose=True,
    solver_print_every=10,
):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(snap_folder, exist_ok=True)

    mu = [float(mu1), float(mu2)]
    mu_tag = _format_mu(mu1, mu2)
    implicit_time_integrator = str(implicit_time_integrator).strip().lower()
    if implicit_time_integrator == "explicit_rk2":
        raise ValueError("implicit_time_integrator must be an implicit method.")

    print("\n====================================================")
    print(
        f"       FOM COMPARISON: explicit_rk2 vs {implicit_time_integrator}"
    )
    print("====================================================")
    print(f"[COMPARE] Parameters: mu=({mu1:.4f}, {mu2:.4f})")
    print(f"[COMPARE] Spatial numerics: limiter={limiter} | flux={riemann_flux}")

    t0 = time.time()
    case_exp = load_or_compute_snaps(
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
        time_integrator="explicit_rk2",
        fixed_dt=explicit_fixed_dt,
        limiter=limiter,
        riemann_flux=riemann_flux,
        verbose=solver_verbose,
        print_every=solver_print_every,
    )
    elapsed_exp = time.time() - t0

    t0 = time.time()
    case_imp = load_or_compute_snaps(
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
        time_integrator=implicit_time_integrator,
        fixed_dt=implicit_fixed_dt,
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
    elapsed_imp = time.time() - t0

    times = np.asarray(case_exp["times"], dtype=np.float64)
    if not np.allclose(times, np.asarray(case_imp["times"], dtype=np.float64)):
        raise RuntimeError("Explicit and implicit runs returned different sample times.")

    x = np.asarray(case_exp["x"], dtype=np.float64)
    y = np.asarray(case_exp["y"], dtype=np.float64)
    nx_case = int(case_exp["nx"])
    ny_case = int(case_exp["ny"])

    snaps_exp = np.asarray(case_exp["state_snapshots"], dtype=np.float64)
    snaps_imp = np.asarray(case_imp["state_snapshots"], dtype=np.float64)
    h_exp = extract_h_snapshots(snaps_exp, nx=nx_case, ny=ny_case)
    h_imp = extract_h_snapshots(snaps_imp, nx=nx_case, ny=ny_case)
    implicit_label = str(case_imp.get("time_integrator", implicit_time_integrator))

    rel_err_imp_vs_exp = _rel_l2_over_time(h_exp, h_imp)

    mass_exp = mass_history_from_state_snapshots(
        snaps_exp, nx=nx_case, ny=ny_case, dx=case_exp["dx"], dy=case_exp["dy"]
    )
    mass_imp = mass_history_from_state_snapshots(
        snaps_imp, nx=nx_case, ny=ny_case, dx=case_imp["dx"], dy=case_imp["dy"]
    )

    print(
        f"[COMPARE] elapsed explicit={elapsed_exp:.3e}s | implicit={elapsed_imp:.3e}s | "
        f"speed_ratio(imp/exp)={elapsed_imp/max(elapsed_exp,1e-14):.3f}"
    )
    print(
        f"[COMPARE] implicit-vs-explicit relative L2(h): "
        f"mean={np.mean(rel_err_imp_vs_exp):.3e}, max={np.max(rel_err_imp_vs_exp):.3e}"
    )
    imp_nonconv = int(
        np.sum(np.asarray(case_imp.get("nonlinear_converged", []), dtype=np.int64) == 0)
    )
    print(
        f"[COMPARE] implicit nonlinear summary: "
        f"method={implicit_label} | "
        f"solver={case_imp.get('implicit_nonlinear_solver', implicit_nonlinear_solver)} | "
        f"nonconverged_steps={imp_nonconv}"
    )

    snaps_exp_path = None
    snaps_imp_path = None
    if save_snapshots:
        snaps_exp_path = os.path.join(results_dir, f"hdm_explicit_snaps_{mu_tag}.npy")
        snaps_imp_path = os.path.join(results_dir, f"hdm_implicit_snaps_{mu_tag}.npy")
        np.save(snaps_exp_path, snaps_exp)
        np.save(snaps_imp_path, snaps_imp)

    err_plot_path = None
    mass_plot_path = None
    final_cmp_path = None
    slices_cmp_path = None
    movie2d_exp_path = None
    movie2d_imp_path = None
    movie2d_slices_cmp_path = None
    movie3d_exp_path = None
    movie3d_imp_path = None

    if save_plot:
        err_plot_path = os.path.join(results_dir, f"compare_relerr_{mu_tag}.png")
        plot_comparison_time_errors(
            h_ref=h_exp,
            model_dict={implicit_label: h_imp},
            times=times,
            out_path=err_plot_path,
            title_prefix=f"Implicit vs explicit error ({mu_tag})",
        )

        mass_plot_path = os.path.join(results_dir, f"compare_mass_{mu_tag}.png")
        _save_mass_comparison(
            times, mass_exp, mass_imp, out_path=mass_plot_path, implicit_label=implicit_label
        )

        final_cmp_path = os.path.join(results_dir, f"compare_final_maps_{mu_tag}.png")
        _save_final_state_comparison(
            h_exp=h_exp[-1],
            h_imp=h_imp[-1],
            lx=case_exp["lx"],
            ly=case_exp["ly"],
            out_path=final_cmp_path,
            implicit_label=implicit_label,
        )

        if save_slices:
            slices_cmp_path = os.path.join(results_dir, f"compare_slices_{mu_tag}.png")
            _save_slice_comparison(
                h_exp=h_exp,
                h_imp=h_imp,
                x=x,
                y=y,
                times=times,
                out_path=slices_cmp_path,
                implicit_label=implicit_label,
            )

        if save_movie_mp4:
            try:
                movie2d_exp_path = os.path.join(results_dir, f"movie2d_explicit_{mu_tag}.mp4")
                save_depth_movie_mp4(
                    h_snapshots=h_exp,
                    times=times,
                    lx=case_exp["lx"],
                    ly=case_exp["ly"],
                    out_path=movie2d_exp_path,
                    fps=movie_fps,
                    frame_stride=movie_frame_stride,
                    title_prefix=f"explicit_rk2 {mu_tag}",
                )
                movie2d_imp_path = os.path.join(results_dir, f"movie2d_implicit_{mu_tag}.mp4")
                save_depth_movie_mp4(
                    h_snapshots=h_imp,
                    times=times,
                    lx=case_imp["lx"],
                    ly=case_imp["ly"],
                    out_path=movie2d_imp_path,
                    fps=movie_fps,
                    frame_stride=movie_frame_stride,
                    title_prefix=f"{implicit_label} {mu_tag}",
                )
            except RuntimeError as exc:
                movie2d_exp_path = None
                movie2d_imp_path = None
                print(f"[WARN] 2D MP4 comparison export skipped: {exc}")

            movie2d_slices_cmp_path = os.path.join(
                results_dir, f"movie2d_slices_compare_{mu_tag}.mp4"
            )
            try:
                save_midline_slice_comparison_movie_mp4(
                    h_ref=h_exp,
                    h_model=h_imp,
                    x=x,
                    y=y,
                    times=times,
                    out_path=movie2d_slices_cmp_path,
                    model_label=str(implicit_label),
                    ref_label="explicit_rk2",
                    fps=movie_fps,
                    frame_stride=movie_frame_stride,
                    title_prefix=f"Slice comparison {mu_tag}",
                )
                print(
                    f"[COMPARE] 2D slice comparison MP4 movie saved to: {movie2d_slices_cmp_path}"
                )
            except RuntimeError as exc:
                movie2d_slices_cmp_path = None
                print(f"[WARN] 2D slice MP4 comparison export skipped: {exc}")

        if save_movie3d_mp4:
            try:
                movie3d_exp_path = os.path.join(results_dir, f"movie3d_explicit_{mu_tag}.mp4")
                save_depth_movie3d_mp4(
                    h_snapshots=h_exp,
                    x=x,
                    y=y,
                    times=times,
                    out_path=movie3d_exp_path,
                    fps=movie_fps,
                    frame_stride=movie_frame_stride,
                    elev=movie3d_elev,
                    azim=movie3d_azim,
                    title_prefix=f"explicit_rk2 {mu_tag}",
                )
                movie3d_imp_path = os.path.join(results_dir, f"movie3d_implicit_{mu_tag}.mp4")
                save_depth_movie3d_mp4(
                    h_snapshots=h_imp,
                    x=x,
                    y=y,
                    times=times,
                    out_path=movie3d_imp_path,
                    fps=movie_fps,
                    frame_stride=movie_frame_stride,
                    elev=movie3d_elev,
                    azim=movie3d_azim,
                    title_prefix=f"{implicit_label} {mu_tag}",
                )
            except RuntimeError as exc:
                movie3d_exp_path = None
                movie3d_imp_path = None
                print(f"[WARN] 3D MP4 comparison export skipped: {exc}")

    report_path = os.path.join(results_dir, f"compare_summary_{mu_tag}.txt")
    write_txt_report(
        report_path,
        [
            (
                "run",
                [
                    ("timestamp", datetime.now().isoformat(timespec="seconds")),
                    ("script", "run_fom_explicit_vs_implict.py"),
                    ("mu1", mu1),
                    ("mu2", mu2),
                ],
            ),
            (
                "configuration",
                [
                    ("nx", nx_case),
                    ("ny", ny_case),
                    ("t_final", t_final),
                    ("num_time_samples", num_time_samples),
                    ("explicit_fixed_dt", explicit_fixed_dt),
                    ("implicit_time_integrator", implicit_label),
                    ("implicit_fixed_dt", implicit_fixed_dt),
                    ("implicit_nonlinear_solver", implicit_nonlinear_solver),
                    ("implicit_max_iter", implicit_max_iter),
                    ("implicit_tol", implicit_tol),
                    ("implicit_relaxation", implicit_relaxation),
                    ("limiter", limiter),
                    ("riemann_flux", riemann_flux),
                    ("save_slices", save_slices),
                    ("save_movie_mp4", save_movie_mp4),
                    ("save_movie3d_mp4", save_movie3d_mp4),
                ],
            ),
            (
                "timing",
                [
                    ("elapsed_explicit_seconds", elapsed_exp),
                    ("elapsed_implicit_seconds", elapsed_imp),
                    ("speed_ratio_imp_over_exp", elapsed_imp / max(elapsed_exp, 1e-14)),
                ],
            ),
            (
                "comparison_metrics",
                [
                    ("mean_rel_l2_h_implicit_vs_explicit", np.mean(rel_err_imp_vs_exp)),
                    ("max_rel_l2_h_implicit_vs_explicit", np.max(rel_err_imp_vs_exp)),
                    (
                        "explicit_mass_drift_final",
                        (mass_exp[-1] - mass_exp[0]) / max(abs(mass_exp[0]), 1e-14),
                    ),
                    (
                        "implicit_mass_drift_final",
                        (mass_imp[-1] - mass_imp[0]) / max(abs(mass_imp[0]), 1e-14),
                    ),
                    ("explicit_num_steps", case_exp["num_solver_steps"]),
                    ("implicit_num_steps", case_imp["num_solver_steps"]),
                    ("implicit_nonconverged_steps", imp_nonconv),
                ],
            ),
            (
                "outputs",
                [
                    ("explicit_snapshots_npy", snaps_exp_path),
                    ("implicit_snapshots_npy", snaps_imp_path),
                    ("error_plot_png", err_plot_path),
                    ("mass_plot_png", mass_plot_path),
                    ("final_state_comparison_png", final_cmp_path),
                    ("slice_comparison_png", slices_cmp_path),
                    ("movie2d_explicit_mp4", movie2d_exp_path),
                    ("movie2d_implicit_mp4", movie2d_imp_path),
                    ("movie2d_slices_compare_mp4", movie2d_slices_cmp_path),
                    ("movie3d_explicit_mp4", movie3d_exp_path),
                    ("movie3d_implicit_mp4", movie3d_imp_path),
                ],
            ),
        ],
    )
    print(f"[COMPARE] Summary saved to: {report_path}")

    return {
        "explicit_case": case_exp,
        "implicit_case": case_imp,
        "rel_err_implicit_vs_explicit": rel_err_imp_vs_exp,
        "report_path": report_path,
    }


if __name__ == "__main__":
    main()
