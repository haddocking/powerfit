"""Generate golden reference outputs from the old C/Cython extension.

git worktree add /tmp/pf-baseline 9a11eed
cd /tmp/pf-baseline
uv venv .venv
source .venv/bin/activate
uv pip install -e .
.venv/bin/python <path-to-this-file>/gen_golden.py

# then copy the output .npz files into tests/fixtures/cython_baseline/
"""

import numpy as np
from pathlib import Path

from powerfit_em._extensions import rotate_grid3d
from powerfit_em._powerfit import blur_points, dilate_points

BASELINE_DIR = Path(__file__).parent.parent / "fixtures" / "cython_baseline"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)


def save(name, **arrays):
    """Wrapper function to save an array to disk."""
    np.savez(BASELINE_DIR / f"{name}.npz", **arrays)
    print("wrote", name)


# -----------------------------------------------------------------------------------------#
# generate baseline for `rotate_grid3d`
# -----------------------------------------------------------------------------------------#
def rotmat_x(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def rotmat_z(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


rotate_cases = [
    # sanity baseline
    ("identity_odd_nearest", (7, 7, 7), np.eye(3, dtype=np.float32), 3, True),
    # trilinear path
    ("identity_odd_trilinear", (7, 7, 7), np.eye(3, dtype=np.float32), 3, False),
    # axis rotation
    ("z90_even_trilinear", (8, 8, 8), rotmat_z(90), 3, False),
    # axis rotation to nearest
    ("z90_even_nearest", (8, 8, 8), rotmat_z(90), 3, True),
    # arbitrary angle (37)
    ("x37_odd_trilinear", (9, 9, 9), rotmat_x(37), 4, False),
    # closest voxel with arbitrary angle
    ("x37_odd_nearest", (9, 9, 9), rotmat_x(37), 4, True),
    # full wrap around
    ("radius_near_bound_trilinear", (6, 6, 6), rotmat_z(45), 3, False),
]


for name, shape, rotmat, radius, nearest in rotate_cases:
    grid = rng.random(shape, dtype=np.float64).astype(np.float32)
    out = np.zeros(shape, dtype=np.float32)
    # use `rotate_grid3d` and save the array
    rotate_grid3d(grid, rotmat, radius, out, nearest)
    save(
        f"rotate_grid3d__{name}",
        grid=grid,
        rotmat=rotmat,
        radius=np.array(radius),
        nearest=np.array(nearest),
        out=out,
    )
# -----------------------------------------------------------------------------------------#


# -----------------------------------------------------------------------------------------#
# generate baseline for `blur_points`
# -----------------------------------------------------------------------------------------#
def blur_points_case(name, shape, points_xyz, weights, sigma, wraparound):
    points = np.asarray(points_xyz, dtype=np.float64).T  # -> shape (3, n)
    weights = np.asarray(weights, dtype=np.float64)
    out = np.zeros(shape, dtype=np.float64)
    blur_points(points=points, weights=weights, sigma=float(sigma), out=out, wraparound=wraparound)
    save(
        f"blur_points__{name}",
        points=points,
        param=weights,
        sigma_or_radius=np.array(float(sigma)),
        wraparound=np.array(wraparound),
        shape=np.array(shape),
        out=out,
    )


blur_cases = [
    # dead-center point, baseline Gaussian splat
    ("single_centered", (10, 10, 10), [(5.0, 5.0, 5.0)], [1.0], 1.5, False),
    # point near edge, no wraparound -> kernel clipped at the boundary
    ("single_near_edge_no_wrap", (10, 10, 10), [(0.5, 5.0, 5.0)], [1.0], 1.5, False),
    # same near-edge point, wraparound=True -> exercises the wrap-around branch
    ("single_near_edge_wrap", (10, 10, 10), [(0.5, 5.0, 5.0)], [1.0], 1.5, True),
    # two overlapping points -> checks additive accumulation into `out`
    ("multi_overlapping", (12, 12, 12), [(5.0, 5.0, 5.0), (6.0, 5.0, 5.0)], [1.0, 0.5], 1.2, False),
]
for name, shape, pts, weights, sigma, wrap in blur_cases:
    blur_points_case(name, shape, pts, weights, sigma, wrap)
# -----------------------------------------------------------------------------------------#


# -----------------------------------------------------------------------------------------#
# generate baseline for `dilate_points`
# -----------------------------------------------------------------------------------------#
def dilate_points_case(name, shape, points_xyz, radii, wraparound):
    points = np.asarray(points_xyz, dtype=np.float64).T  # -> shape (3, n)
    radii = np.asarray(radii, dtype=np.float64)
    out = np.zeros(shape, dtype=np.float64)
    dilate_points(points=points, radii=radii, out=out, wraparound=wraparound)
    save(
        f"dilate_points__{name}",
        points=points,
        param=radii,
        sigma_or_radius=np.array(0.0),
        wraparound=np.array(wraparound),
        shape=np.array(shape),
        out=out,
    )


dilate_cases = [
    # dead-center point, baseline hard sphere
    ("single_centered", (10, 10, 10), [(5.0, 5.0, 5.0)], [2.0], False),
    # point near edge, no wraparound -> sphere clipped at the boundary
    ("single_near_edge_no_wrap", (10, 10, 10), [(0.5, 5.0, 5.0)], [2.0], False),
    # same near-edge point, wraparound=True -> exercises the wrap-around branch
    ("single_near_edge_wrap", (10, 10, 10), [(0.5, 5.0, 5.0)], [2.0], True),
    # two overlapping spheres -> checks union behaviour (voxel stays 1, no double-set issue)
    ("multi_overlapping", (12, 12, 12), [(5.0, 5.0, 5.0), (6.0, 5.0, 5.0)], [2.0, 1.5], False),
]
for name, shape, pts, radii, wrap in dilate_cases:
    dilate_points_case(name, shape, pts, radii, wrap)
# -----------------------------------------------------------------------------------------#
