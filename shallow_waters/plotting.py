"""
Plotting utilities for shallow-water HDM workflows.

These plots are designed to remain readable when comparing model variants.
"""

import subprocess

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def nearest_indices(times, target_times):
    times = np.asarray(times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    return [int(np.argmin(np.abs(times - t))) for t in target_times]


def _resolve_h_limits(h_data, h_limits=None):
    if h_limits is None:
        return float(np.min(h_data)), float(np.max(h_data))
    hmin, hmax = float(h_limits[0]), float(h_limits[1])
    if hmax <= hmin:
        raise ValueError(f"Invalid h_limits={h_limits}; require hmax > hmin.")
    return hmin, hmax


def plot_depth_maps_grid(
    h_snapshots,
    times,
    lx,
    ly,
    out_path,
    title_prefix="",
    n_panels=6,
    h_limits=None,
):
    """
    Plot 2x3 top-view depth maps at representative times.
    """
    h_snapshots = np.asarray(h_snapshots, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if h_snapshots.ndim != 3:
        raise ValueError(f"Expected h_snapshots[nt, nx, ny], got {h_snapshots.shape}.")

    n_panels = int(n_panels)
    if n_panels < 1:
        raise ValueError("n_panels must be >= 1.")

    vmin_h, vmax_h = _resolve_h_limits(h_snapshots, h_limits=h_limits)

    nrows, ncols = 2, 3
    n_use = min(n_panels, nrows * ncols)
    t_targets = np.linspace(0.0, float(times[-1]), n_use)
    inds = nearest_indices(times, t_targets)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 9), constrained_layout=True)
    axes = axes.ravel()

    im = None
    for i, ax in enumerate(axes):
        if i < n_use:
            k = inds[i]
            im = ax.imshow(
                h_snapshots[k].T,
                origin="lower",
                extent=[0.0, float(lx), 0.0, float(ly)],
                vmin=vmin_h,
                vmax=vmax_h,
                aspect="equal",
            )
            ax.set_title(f"t = {times[k]:.3f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        else:
            ax.axis("off")

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.85, label="water depth h")
    if title_prefix:
        fig.suptitle(title_prefix, y=1.02)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_midline_spacetime(
    h_snapshots,
    x,
    y,
    times,
    out_path,
    title_prefix="",
    h_limits=None,
):
    """
    Space-time maps along the two midlines:
      - h(x, y_mid, t)
      - h(x_mid, y, t)

    These are typically easier to compare than many static slice overlays.
    """
    h_snapshots = np.asarray(h_snapshots, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    mid_x = x.size // 2
    mid_y = y.size // 2

    hx_t = h_snapshots[:, :, mid_y]      # [nt, nx]
    hy_t = h_snapshots[:, mid_x, :]      # [nt, ny]

    vmin, vmax = _resolve_h_limits(h_snapshots, h_limits=h_limits)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    im1 = ax1.imshow(
        hx_t,
        origin="lower",
        aspect="auto",
        extent=[x[0], x[-1], times[0], times[-1]],
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("time")
    ax1.set_title(f"h(x, y_mid={y[mid_y]:.3f}, t)")

    im2 = ax2.imshow(
        hy_t,
        origin="lower",
        aspect="auto",
        extent=[y[0], y[-1], times[0], times[-1]],
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_xlabel("y")
    ax2.set_ylabel("time")
    ax2.set_title(f"h(x_mid={x[mid_x]:.3f}, y, t)")

    cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.92)
    cbar.set_label("water depth h")

    if title_prefix:
        fig.suptitle(title_prefix, y=1.02)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def relative_l2_error_over_time(h_ref, h_model):
    """
    Relative L2 error over time for h-fields [nt, nx, ny].
    """
    h_ref = np.asarray(h_ref, dtype=np.float64)
    h_model = np.asarray(h_model, dtype=np.float64)
    if h_ref.shape != h_model.shape:
        raise ValueError(f"Shape mismatch: ref={h_ref.shape}, model={h_model.shape}.")

    ref_norm = np.linalg.norm(h_ref.reshape(h_ref.shape[0], -1), axis=1)
    err_norm = np.linalg.norm((h_model - h_ref).reshape(h_ref.shape[0], -1), axis=1)
    ref_norm = np.where(ref_norm > 0.0, ref_norm, 1.0)
    return err_norm / ref_norm


def plot_comparison_time_errors(h_ref, model_dict, times, out_path, title_prefix=""):
    """
    Plot relative L2-in-space error vs time for multiple models.

    Parameters
    ----------
    h_ref : ndarray [nt, nx, ny]
        Reference solution (typically HDM).
    model_dict : dict[str, ndarray]
        Mapping model name -> h snapshots with same shape as h_ref.
    """
    times = np.asarray(times, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)

    for name, h_model in model_dict.items():
        rel = relative_l2_error_over_time(h_ref, h_model)
        ax.plot(times, rel, linewidth=2.0, label=name)

    ax.set_xlabel("time")
    ax.set_ylabel("relative L2 error in h")
    ax.set_title("Model error vs time")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True)

    if title_prefix:
        fig.suptitle(title_prefix, y=1.02)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _movie_frame_ids(nt, frame_stride=None):
    nt = int(nt)
    if nt < 1:
        return [0]
    if frame_stride is None:
        stride = max(1, nt // 240)
    else:
        stride = max(1, int(frame_stride))
    frame_ids = list(range(0, nt, stride))
    if frame_ids[-1] != nt - 1:
        frame_ids.append(nt - 1)
    return frame_ids


_FFMPEG_CODEC_CACHE = None


def _select_ffmpeg_codec():
    """
    Select a video codec that is actually available in the current ffmpeg build.

    Priority:
      1) h264
      2) libx264
      3) mpeg4
    """
    global _FFMPEG_CODEC_CACHE
    if _FFMPEG_CODEC_CACHE is not None:
        return _FFMPEG_CODEC_CACHE

    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError(
            "ffmpeg writer is not available in matplotlib. Install ffmpeg to enable MP4 export."
        )

    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
        )
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except Exception:
        text = ""

    available = set()
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) >= 2 and tokens[0].startswith("V"):
            available.add(tokens[1])

    for codec in ("h264", "libx264", "mpeg4"):
        if codec in available:
            _FFMPEG_CODEC_CACHE = codec
            return codec

    # Fallback for minimal ffmpeg builds where encoder list parsing fails.
    _FFMPEG_CODEC_CACHE = "mpeg4"
    return _FFMPEG_CODEC_CACHE


def _ffmpeg_writer(fps, bitrate):
    return animation.FFMpegWriter(
        fps=int(max(1, fps)),
        bitrate=int(bitrate),
        codec=_select_ffmpeg_codec(),
    )


def save_depth_movie_mp4(
    h_snapshots,
    times,
    lx,
    ly,
    out_path,
    fps=24,
    frame_stride=None,
    title_prefix="",
    h_limits=None,
):
    """
    Save a 2D top-view depth animation as MP4 using ffmpeg.

    Raises RuntimeError if ffmpeg is unavailable.
    """
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError(
            "ffmpeg writer is not available in matplotlib. Install ffmpeg to enable MP4 export."
        )

    h_snapshots = np.asarray(h_snapshots, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if h_snapshots.ndim != 3:
        raise ValueError(f"Expected h_snapshots[nt, nx, ny], got {h_snapshots.shape}.")

    nt = h_snapshots.shape[0]
    frame_ids = _movie_frame_ids(nt, frame_stride=frame_stride)

    vmin_h, vmax_h = _resolve_h_limits(h_snapshots, h_limits=h_limits)

    fig, ax = plt.subplots(figsize=(6.8, 5.5), constrained_layout=True)
    im = ax.imshow(
        h_snapshots[0].T,
        origin="lower",
        extent=[0.0, float(lx), 0.0, float(ly)],
        vmin=vmin_h,
        vmax=vmax_h,
        aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("water depth h")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title("")

    def _update(frame):
        k = frame_ids[frame]
        im.set_data(h_snapshots[k].T)
        if title_prefix:
            title.set_text(f"{title_prefix} | t = {times[k]:.3f}")
        else:
            title.set_text(f"t = {times[k]:.3f}")
        return [im, title]

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=len(frame_ids),
        interval=1000.0 / max(float(fps), 1.0),
        blit=False,
    )
    writer = _ffmpeg_writer(fps=fps, bitrate=2200)
    ani.save(out_path, writer=writer)
    plt.close(fig)


def save_depth_movie3d_mp4(
    h_snapshots,
    x,
    y,
    times,
    out_path,
    fps=20,
    frame_stride=None,
    elev=28.0,
    azim=-130.0,
    title_prefix="",
    h_limits=None,
):
    """
    Save a 3D surface depth animation as MP4 using ffmpeg.

    Raises RuntimeError if ffmpeg is unavailable.
    """
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError(
            "ffmpeg writer is not available in matplotlib. Install ffmpeg to enable MP4 export."
        )

    h_snapshots = np.asarray(h_snapshots, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if h_snapshots.ndim != 3:
        raise ValueError(f"Expected h_snapshots[nt, nx, ny], got {h_snapshots.shape}.")

    nt = h_snapshots.shape[0]
    frame_ids = _movie_frame_ids(nt, frame_stride=frame_stride)
    X, Y = np.meshgrid(x, y, indexing="ij")

    zmin, zmax = _resolve_h_limits(h_snapshots, h_limits=h_limits)

    fig = plt.figure(figsize=(8.4, 6.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    # Downsample surface rendering when grid is large for smoother animation.
    rstride = max(1, int(np.ceil(h_snapshots.shape[1] / 120)))
    cstride = max(1, int(np.ceil(h_snapshots.shape[2] / 120)))

    def _update(frame):
        k = frame_ids[frame]
        ax.clear()
        ax.plot_surface(
            X,
            Y,
            h_snapshots[k],
            linewidth=0,
            antialiased=True,
            cmap="viridis",
            rstride=rstride,
            cstride=cstride,
        )
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("h")
        if title_prefix:
            ax.set_title(f"{title_prefix} | t = {times[k]:.3f}")
        else:
            ax.set_title(f"t = {times[k]:.3f}")
        ax.view_init(elev=float(elev), azim=float(azim))
        return []

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=len(frame_ids),
        interval=1000.0 / max(float(fps), 1.0),
        blit=False,
    )
    writer = _ffmpeg_writer(fps=fps, bitrate=2600)
    ani.save(out_path, writer=writer)
    plt.close(fig)


def save_midline_slice_movie_mp4(
    h_snapshots,
    x,
    y,
    times,
    out_path,
    fps=24,
    frame_stride=None,
    title_prefix="",
    line_color="black",
    line_label="HDM",
    h_limits=None,
):
    """
    Save an MP4 movie of 2D midline slices:
      - left panel: h(x, y_mid, t)
      - right panel: h(x_mid, y, t)

    Raises RuntimeError if ffmpeg is unavailable.
    """
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError(
            "ffmpeg writer is not available in matplotlib. Install ffmpeg to enable MP4 export."
        )

    h_snapshots = np.asarray(h_snapshots, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if h_snapshots.ndim != 3:
        raise ValueError(f"Expected h_snapshots[nt, nx, ny], got {h_snapshots.shape}.")

    nt = h_snapshots.shape[0]
    frame_ids = _movie_frame_ids(nt, frame_stride=frame_stride)
    mid_x = x.size // 2
    mid_y = y.size // 2

    ymin, ymax = _resolve_h_limits(h_snapshots, h_limits=h_limits)

    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    line_x = ax_x.plot(
        x,
        h_snapshots[0, :, mid_y],
        color=str(line_color),
        linewidth=2.2,
        label=str(line_label),
    )[0]
    line_y = ax_y.plot(
        y,
        h_snapshots[0, mid_x, :],
        color=str(line_color),
        linewidth=2.2,
        label=str(line_label),
    )[0]

    ax_x.set_xlim(float(x[0]), float(x[-1]))
    ax_y.set_xlim(float(y[0]), float(y[-1]))
    ax_x.set_ylim(ymin, ymax)
    ax_y.set_ylim(ymin, ymax)
    ax_x.set_xlabel("x")
    ax_y.set_xlabel("y")
    ax_x.set_ylabel("h")
    ax_y.set_ylabel("h")
    ax_x.set_title(f"x-slice: h(x, y={y[mid_y]:.3f})")
    ax_y.set_title(f"y-slice: h(x={x[mid_x]:.3f}, y)")
    ax_x.grid(True, alpha=0.35)
    ax_y.grid(True, alpha=0.35)
    ax_x.legend(loc="best", frameon=True)
    ax_y.legend(loc="best", frameon=True)
    suptitle = fig.suptitle("")

    def _update(frame):
        k = frame_ids[frame]
        line_x.set_ydata(h_snapshots[k, :, mid_y])
        line_y.set_ydata(h_snapshots[k, mid_x, :])
        if title_prefix:
            suptitle.set_text(f"{title_prefix} | t = {times[k]:.3f}")
        else:
            suptitle.set_text(f"t = {times[k]:.3f}")
        return [line_x, line_y, suptitle]

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=len(frame_ids),
        interval=1000.0 / max(float(fps), 1.0),
        blit=False,
    )
    writer = _ffmpeg_writer(fps=fps, bitrate=2200)
    ani.save(out_path, writer=writer)
    plt.close(fig)


def save_midline_slice_comparison_movie_mp4(
    h_ref,
    h_model,
    x,
    y,
    times,
    out_path,
    model_label="model",
    ref_label="reference",
    fps=24,
    frame_stride=None,
    title_prefix="",
    h_limits=None,
):
    """
    Save an MP4 movie of 2D midline slice comparisons:
      - left panel: h(x, y_mid, t)
      - right panel: h(x_mid, y, t)
      - each panel overlays ref and model curves.

    Raises RuntimeError if ffmpeg is unavailable.
    """
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError(
            "ffmpeg writer is not available in matplotlib. Install ffmpeg to enable MP4 export."
        )

    h_ref = np.asarray(h_ref, dtype=np.float64)
    h_model = np.asarray(h_model, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if h_ref.ndim != 3 or h_model.ndim != 3:
        raise ValueError("Expected h_ref and h_model with shape [nt, nx, ny].")
    if h_ref.shape != h_model.shape:
        raise ValueError(f"Shape mismatch: ref={h_ref.shape}, model={h_model.shape}.")

    nt = h_ref.shape[0]
    frame_ids = _movie_frame_ids(nt, frame_stride=frame_stride)
    mid_x = x.size // 2
    mid_y = y.size // 2

    if h_limits is None:
        ymin = float(min(np.min(h_ref), np.min(h_model)))
        ymax = float(max(np.max(h_ref), np.max(h_model)))
    else:
        ymin, ymax = _resolve_h_limits(h_ref, h_limits=h_limits)

    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    line_x_ref = ax_x.plot(
        x, h_ref[0, :, mid_y], color="black", linewidth=2.2, label=str(ref_label)
    )[0]
    line_x_model = ax_x.plot(
        x,
        h_model[0, :, mid_y],
        color="tab:red",
        linewidth=2.2,
        linestyle="--",
        label=str(model_label),
    )[0]
    line_y_ref = ax_y.plot(
        y, h_ref[0, mid_x, :], color="black", linewidth=2.2, label=str(ref_label)
    )[0]
    line_y_model = ax_y.plot(
        y,
        h_model[0, mid_x, :],
        color="tab:red",
        linewidth=2.2,
        linestyle="--",
        label=str(model_label),
    )[0]

    ax_x.set_xlim(float(x[0]), float(x[-1]))
    ax_y.set_xlim(float(y[0]), float(y[-1]))
    ax_x.set_ylim(ymin, ymax)
    ax_y.set_ylim(ymin, ymax)
    ax_x.set_xlabel("x")
    ax_y.set_xlabel("y")
    ax_x.set_ylabel("h")
    ax_y.set_ylabel("h")
    ax_x.set_title(f"x-slice: h(x, y={y[mid_y]:.3f})")
    ax_y.set_title(f"y-slice: h(x={x[mid_x]:.3f}, y)")
    ax_x.grid(True, alpha=0.35)
    ax_y.grid(True, alpha=0.35)
    ax_x.legend(loc="best", frameon=True)
    ax_y.legend(loc="best", frameon=True)
    suptitle = fig.suptitle("")

    def _update(frame):
        k = frame_ids[frame]
        line_x_ref.set_ydata(h_ref[k, :, mid_y])
        line_x_model.set_ydata(h_model[k, :, mid_y])
        line_y_ref.set_ydata(h_ref[k, mid_x, :])
        line_y_model.set_ydata(h_model[k, mid_x, :])
        if title_prefix:
            suptitle.set_text(f"{title_prefix} | t = {times[k]:.3f}")
        else:
            suptitle.set_text(f"t = {times[k]:.3f}")
        return [line_x_ref, line_x_model, line_y_ref, line_y_model, suptitle]

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=len(frame_ids),
        interval=1000.0 / max(float(fps), 1.0),
        blit=False,
    )
    writer = _ffmpeg_writer(fps=fps, bitrate=2200)
    ani.save(out_path, writer=writer)
    plt.close(fig)
