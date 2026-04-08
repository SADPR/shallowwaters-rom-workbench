# shallowwaters-rom-workbench

Shallow-water ROM workbench scaffold, aligned with the `burgers2d-rom-workbench` workflow.

## Current scope
- Scenario: `two_bumps_collision`
- Parameters: `mu1` (left bump amplitude), `mu2` (right bump amplitude)
- Time is treated as an explicit coordinate via fixed snapshot times.
- Numerics: MUSCL (selectable limiter) + HLL/HLLC (selectable flux) with three time integrators:
  - `implicit_bdf2` (default): BDF2 with BE startup + fixed-dt nonlinear solve
  - `implicit_be`: Backward Euler + fixed-dt nonlinear solve
  - `explicit_rk2`: SSP-RK2 (kept for comparison)

## Repository logic
- `shallow_waters/config.py`: global constants and parameter ranges.
- `shallow_waters/scenarios.py`: initial-condition builders.
- `shallow_waters/solver.py`: fluxes, reconstruction, RHS, residual wrappers, stepping.
- `shallow_waters/core.py`: parameter grid helpers, snapshot caching, plotting helpers.
- `shallow_waters/plotting.py`: map, space-time, movie, and comparison plots.
- `run_fom_training.py`: precompute/load training snapshots over a `(mu1, mu2)` grid and save `fom_training_param_space.png`.
- `run_fom.py`: single-case HDM run with saved outputs.
- `run_fom_explicit_vs_implict.py`: direct explicit-vs-implicit HDM comparison run.

## Start here
Generate training snapshots (cached in `Results/param_snaps`):

```bash
python3 run_fom_training.py
```

Run one full-order baseline case:

```bash
python3 run_fom.py
```

Run explicit-vs-implicit comparison:

```bash
python3 run_fom_explicit_vs_implict.py
```

## Why fixed sample times
Even though the solver uses CFL-based state-dependent time steps internally, all saved
trajectories are interpolated onto one shared `times` array. This keeps snapshot tensors
consistent across parameters.

## Notes on burgers-style parity
`shallow_waters/core.py` and `shallow_waters/solver.py` intentionally expose:
- `get_snapshot_params(...)`
- `param_to_snap_fn(...)`
- `load_or_compute_snaps(...)`
- `shallow_water_rhs2D(...)`
- `shallow_water_res2D(...)`

so downstream workflows can mirror the burgers structure with minimal renaming.

## Explicit vs implicit
The current default in `shallow_waters/config.py` is:
- `TIME_INTEGRATOR = "implicit_bdf2"`
- `IMPLICIT_NONLINEAR_SOLVER = "newton_krylov"` (recommended)
- `LIMITER = "mc"` (less diffusive than minmod)
- `RIEMANN_FLUX = "hllc"` (sharper contact handling)

You can switch per run in both `run_fom.py` and `run_fom_training.py`:
- `time_integrator="implicit_bdf2"`, `time_integrator="implicit_be"`, or `time_integrator="explicit_rk2"`
- `limiter="mc"` or `limiter="minmod"`
- `riemann_flux="hllc"` or `riemann_flux="hll"`
- `dt_multiplier` (applies to CFL-based `dt`)
- implicit controls: `implicit_nonlinear_solver`, `implicit_max_iter`,
  `implicit_tol`, `implicit_relaxation`

Solver logs include step information and residual monitors:
- explicit: relative step update `step_res`
- implicit BE: residual `be_res` + nonlinear iteration count per step
- implicit BDF2: residual `bdf2_res` + nonlinear iteration count per step
  (and Newton-Krylov linear-iteration diagnostics when enabled)

The implicit solver uses a fixed CFL-based `dt` (scaled by `dt_multiplier`) and
does not perform adaptive time-step retries.

Snapshot cache filenames include solver settings, so explicit and implicit runs
are stored separately even at the same `(mu1, mu2)`.

## Plotting
`run_fom.py` now saves by default on every run:
- top-view map grid (`hdm_maps_*.png`)
- space-time midline maps (`hdm_spacetime_*.png`)
- midline slices in `4 x 2` layout (`hdm_slices_*.png`)
- mass drift (`hdm_mass_*.png`)
- 2D map movie, 2D slice movie, and 3D movie
  (`hdm_movie2d_*.mp4`, `hdm_movie_slices2d_*.mp4`, `hdm_movie3d_*.mp4`)

`run_fom_explicit_vs_implict.py` additionally saves by default on every run:
- explicit-vs-implicit slice comparison (`compare_slices_*.png`)
- explicit-vs-implicit MP4 movies:
  `movie2d_explicit_*.mp4`, `movie2d_implicit_*.mp4`,
  `movie2d_slices_compare_*.mp4`, `movie3d_explicit_*.mp4`, `movie3d_implicit_*.mp4`

Midline static slices use a
`4 x 2` subplot layout:
- left column: x-direction midline slices
- right column: y-direction midline slices
- each row: one of 4 representative time instances

MP4 export is enabled by default (if ffmpeg is available) with:
- `save_movie_mp4=True`
- `movie_fps=<int>`
- `movie_frame_stride=<int or None>`
- `save_movie3d_mp4=True` (3D surface animation)
- `movie3d_elev=<float>`, `movie3d_azim=<float>`

This writes `hdm_movie2d_*.mp4` and `hdm_movie3d_*.mp4`
when ffmpeg is available.
For multi-model comparisons, space-time plots are usually more readable
than many overlaid slices.
