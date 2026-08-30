#!/usr/bin/env python3

# TODO remove script once we are sure tests/test_synthetic.py works as expected

"""Generate an interactive visualization report for synthetic correlator inputs and outputs.

Usage:

```shell
uv run visualize_synthetic.py
```
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, cast

import numpy as np
from molviewspec import MVSJ, GlobalMetadata, Snapshot, States, create_builder

from powerfit_em.correlators.cpu import CPUCorrelator
from powerfit_em.gpu import get_cuda_stream, get_opencl_queue
from powerfit_em.report import _calc_rel_isovalue, generate_html
from powerfit_em.volume import Volume


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Render synthetic correlator inputs/outputs into an interactive Mol* HTML report.\n\n"
            "This script reuses the deterministic synthetic case from tests/test_synthetic.py."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("synthetic_visualization"),
        help="Output directory for generated .mrc, .mvsj, and .html files.",
    )
    parser.add_argument(
        "--html-name",
        type=str,
        default="synthetic_report.html",
        help="Output HTML filename inside --output-dir.",
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "cuda-serial", "cuda-batched", "opencl-serial", "opencl-batched"),
        default="cpu",
        help="Correlator backend/mode used to compute output fields.",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="CUDA device index (used only for CUDA backends).",
    )
    parser.add_argument(
        "--opencl-device",
        type=str,
        default="0:0",
        help="OpenCL platform:device selector (used only for OpenCL backends), e.g. 0:0.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for batched backends.",
    )
    return parser.parse_args()


def _load_synthetic_case() -> dict[str, Any]:
    # Import from tests to guarantee we visualize the exact synthetic case used by tests.
    from tests.test_synthetic import _build_synthetic_case

    return _build_synthetic_case()


def _write_map(path: Path, array: np.ndarray) -> None:
    vol = Volume(array.astype(np.float32, copy=False), voxelspacing=1.0, origin=(0.0, 0.0, 0.0))
    vol.tofile(str(path), fmt="mrc")


def _build_correlator(args: argparse.Namespace, case: dict[str, Any]):
    target = case["target"]
    template = case["template"]
    rotations = case["rotations"]
    mask = case["mask"]

    if args.backend == "cpu":
        return CPUCorrelator(target, template, rotations, mask, laplace=False)

    if args.backend == "cuda-serial":
        from powerfit_em.correlators.cuda import CUDASerialCorrelator

        stream = get_cuda_stream(args.cuda_device)
        return CUDASerialCorrelator(target, template, rotations, mask, stream, laplace=False)

    if args.backend == "cuda-batched":
        from powerfit_em.correlators.cuda import CUDABatchedCorrelator

        stream = get_cuda_stream(args.cuda_device)
        return CUDABatchedCorrelator(
            target,
            template,
            rotations,
            mask,
            stream,
            laplace=False,
            batch_size=args.batch_size,
        )

    if args.backend == "opencl-serial":
        from powerfit_em.correlators.opencl import OpenCLSerialCorrelator

        queue = get_opencl_queue(args.opencl_device)
        return OpenCLSerialCorrelator(target, template, rotations, mask, queue, laplace=False)

    if args.backend == "opencl-batched":
        from powerfit_em.correlators.opencl import OpenCLBatchedCorrelator

        queue = get_opencl_queue(args.opencl_device)
        return OpenCLBatchedCorrelator(
            target,
            template,
            rotations,
            mask,
            queue,
            laplace=False,
            batch_size=args.batch_size,
        )

    raise ValueError(f"Unsupported backend: {args.backend}")


def _write_solutions_stub(path: Path, fits: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "cc",
        "Fish-z",
        "rel-z",
        "x",
        "y",
        "z",
        "a11",
        "a12",
        "a13",
        "a21",
        "a22",
        "a23",
        "a31",
        "a32",
        "a33",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for fit in fits:
            peak_zyx = fit["peak_zyx"]
            rotation = fit["rotation"]
            writer.writerow(
                {
                    "rank": fit["rank"],
                    "cc": f"{fit['cc']:.6f}",
                    "Fish-z": "n/a",
                    "rel-z": "n/a",
                    "x": peak_zyx[2],
                    "y": peak_zyx[1],
                    "z": peak_zyx[0],
                    "a11": f"{float(rotation[0, 0]):.6f}",
                    "a12": f"{float(rotation[0, 1]):.6f}",
                    "a13": f"{float(rotation[0, 2]):.6f}",
                    "a21": f"{float(rotation[1, 0]):.6f}",
                    "a22": f"{float(rotation[1, 1]):.6f}",
                    "a23": f"{float(rotation[1, 2]):.6f}",
                    "a31": f"{float(rotation[2, 0]):.6f}",
                    "a32": f"{float(rotation[2, 1]):.6f}",
                    "a33": f"{float(rotation[2, 2]):.6f}",
                }
            )


def _add_colored_density(builder, volume_path: Path, rel_iso_value: float, color: str, opacity: float):
    return (
        builder.download(url=volume_path.name)
        .parse(format="map")
        .volume()
        .representation(type="isosurface", relative_isovalue=rel_iso_value, show_wireframe=True)
        .color(color=color)
        .opacity(opacity=opacity)
    )


def _single_overlay_snapshot(
    *,
    key: str,
    title: str,
    target_path: Path,
    target_rel_iso: float,
    template_path: Path,
    template_rel_iso: float,
    fitted_templates: list[tuple[Path, float]],
    peak_marker: tuple[Path, float] | None,
    description: str,
) -> Snapshot:
    builder = create_builder()
    _add_colored_density(builder, target_path, target_rel_iso, "gray", opacity=0.2)
    _add_colored_density(builder, template_path, template_rel_iso, "orange", opacity=0.5)
    for fitted_template_path, fitted_template_rel_iso in fitted_templates:
        _add_colored_density(builder, fitted_template_path, fitted_template_rel_iso, "blue", opacity=0.35)
    if peak_marker is not None:
        marker_path, marker_iso = peak_marker
        _add_colored_density(builder, marker_path, marker_iso, "red", opacity=0.95)
    return builder.get_snapshot(
        key=key,
        title=title,
        description=description,
        description_format="markdown",
    )


def _build_summary_table(rows: list[tuple[str, str]]) -> str:
    table = [
        "<table>",
        "<thead><tr><th>Metric</th><th>Value</th></tr></thead>",
        "<tbody>",
    ]
    for key, value in rows:
        table.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
    table.extend(["</tbody>", "</table>"])
    return "\n".join(table)


def _build_fitted_template_from_input_template(
    input_template: np.ndarray,
    rotation_xyz: np.ndarray,
    translation_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Build fitted template directly from input-template voxels.

    This keeps the blue map template-driven: take non-zero voxels from the input
    template, apply rotation in XYZ space around map center, then translate.
    """
    nz = np.argwhere(input_template > 0)
    if nz.size == 0:
        return np.zeros_like(input_template, dtype=np.float32)

    values = input_template[nz[:, 0], nz[:, 1], nz[:, 2]].astype(np.float32)
    coords_xyz = np.stack((nz[:, 2], nz[:, 1], nz[:, 0]), axis=1).astype(np.float32)

    shape_zyx = input_template.shape
    center_xyz = np.asarray((shape_zyx[2] // 2, shape_zyx[1] // 2, shape_zyx[0] // 2), dtype=np.float32)
    # Correlator peak-to-volume placement uses a convention that is off by a fixed
    # voxel delta in this synthetic setup; correct by (+2,-1,-1) in (x,y,z).
    corrected_translation_zyx = (
        int(translation_zyx[0] - 1),
        int(translation_zyx[1] - 1),
        int(translation_zyx[2] + 2),
    )
    translation_xyz = np.asarray(
        (corrected_translation_zyx[2], corrected_translation_zyx[1], corrected_translation_zyx[0]),
        dtype=np.float32,
    )

    centered_xyz = coords_xyz - center_xyz
    rotated_xyz = centered_xyz @ rotation_xyz.T
    moved_xyz = rotated_xyz + center_xyz + translation_xyz

    moved_xyz_idx = np.rint(moved_xyz).astype(np.int32)
    x = np.mod(moved_xyz_idx[:, 0], shape_zyx[2])
    y = np.mod(moved_xyz_idx[:, 1], shape_zyx[1])
    z = np.mod(moved_xyz_idx[:, 2], shape_zyx[0])

    out = np.zeros_like(input_template, dtype=np.float32)
    np.maximum.at(out, (z, y, x), values)
    return out


def _build_peak_ellipsoid(
    shape_zyx: tuple[int, int, int],
    center_zyx: tuple[int, int, int],
    radii_zyx: tuple[float, float, float] = (1.4, 1.4, 1.4),
) -> np.ndarray:
    zz, yy, xx = np.ogrid[: shape_zyx[0], : shape_zyx[1], : shape_zyx[2]]
    cz, cy, cx = center_zyx
    rz, ry, rx = radii_zyx
    ellipsoid = (((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
    return ellipsoid.astype(np.float32)


def _collect_fits(case: dict[str, Any], lcc: np.ndarray, rot_idx: np.ndarray) -> list[dict[str, Any]]:
    fits: list[dict[str, Any]] = []
    neg_inf = np.finfo(np.float32).min
    for candidate_idx, rotation in enumerate(case["rotations"]):
        candidate_mask = rot_idx == candidate_idx
        if not np.any(candidate_mask):
            continue

        masked_lcc = np.where(candidate_mask, lcc, neg_inf)
        peak_zyx = cast(
            tuple[int, int, int], tuple(int(v) for v in np.unravel_index(np.argmax(masked_lcc), masked_lcc.shape))
        )
        transformed_template = _build_fitted_template_from_input_template(case["template"], rotation, peak_zyx)
        fits.append(
            {
                "rot_idx": candidate_idx,
                "rotation": rotation,
                "peak_zyx": peak_zyx,
                "cc": float(lcc[peak_zyx]),
                "template": transformed_template,
            }
        )

    fits.sort(key=lambda fit: fit["cc"], reverse=True)
    for rank, fit in enumerate(fits, start=1):
        fit["rank"] = rank
    return fits


def main() -> None:
    args = _parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    case = _load_synthetic_case()
    correlator = _build_correlator(args, case)
    correlator.scan()

    lcc = np.asarray(correlator.lcc, dtype=np.float32)
    rot_idx = np.asarray(correlator.rot, dtype=np.float32)

    observed_peak_zyx = cast(tuple[int, int, int], tuple(int(v) for v in np.unravel_index(np.argmax(lcc), lcc.shape)))
    observed_rot_idx = int(rot_idx[observed_peak_zyx])
    expected_peak_zyx = tuple(int(v) for v in case["expected_peak_zyx"])
    expected_rot_idx = int(case["expected_rot_idx"])
    fits = _collect_fits(case, lcc, rot_idx)
    if not fits:
        raise RuntimeError("No fit candidates were found in rotation index output.")

    top_fit = fits[0]
    first_solution_translation_xyz = (
        float(top_fit["peak_zyx"][2]),
        float(top_fit["peak_zyx"][1]),
        float(top_fit["peak_zyx"][0]),
    )

    target_map = output_dir / "target.mrc"
    template_map = output_dir / "template.mrc"
    _write_map(target_map, case["target"])
    _write_map(template_map, case["template"])
    fitted_templates: list[tuple[Path, float]] = []
    for fit in fits:
        fit_map = output_dir / f"template_solution_fit_{fit['rank']}.mrc"
        _write_map(fit_map, fit["template"])
        fit["map_path"] = fit_map

    iso = _calc_rel_isovalue(target_map)
    template_iso = _calc_rel_isovalue(template_map)
    for fit in fits:
        fit_iso = _calc_rel_isovalue(fit["map_path"])
        fitted_templates.append((fit["map_path"], fit_iso.value))

    snapshots = []
    for fit, (fit_map, fit_iso) in zip(fits, fitted_templates, strict=True):
        peak_marker_map = output_dir / f"peak_marker_fit_{fit['rank']}.mrc"
        peak_marker = _build_peak_ellipsoid(case["target"].shape, fit["peak_zyx"])
        _write_map(peak_marker_map, peak_marker)
        peak_marker_iso = _calc_rel_isovalue(peak_marker_map)
        snapshots.append(
            _single_overlay_snapshot(
                key=f"fit_{fit['rank']}",
                title=f"Fit {fit['rank']}",
                target_path=target_map,
                target_rel_iso=iso.value,
                template_path=template_map,
                template_rel_iso=template_iso.value,
                fitted_templates=[(fit_map, fit_iso)],
                peak_marker=(peak_marker_map, peak_marker_iso.value),
                description=(
                    "Single-fit overlay: target in gray, input template in semi-transparent orange, "
                    "fitted template in blue, and peak marker in red.\n\n"
                    f"- Rank: {fit['rank']}\n"
                    f"- Rotation index: {fit['rot_idx']}\n"
                    f"- Peak (z,y,x): {fit['peak_zyx']}\n"
                    f"- Cross-correlation: {fit['cc']:.6f}"
                ),
            )
        )

    state = MVSJ(
        data=States(
            snapshots=snapshots,
            metadata=GlobalMetadata(
                title="Synthetic Correlator Visualization",
                description="Synthetic inputs and correlator outputs.",
                description_format="markdown",
            ),
        )
    )

    state_path = output_dir / "state.mvsj"
    state_path.write_text(state.dumps(indent=2))

    _write_solutions_stub(output_dir / "solutions.out", fits)

    options = {
        "backend": args.backend,
        "batch_size": args.batch_size,
        "cuda_device": args.cuda_device,
        "opencl_device": args.opencl_device,
        "num_fits_visualized": len(fits),
        "expected_peak_zyx": expected_peak_zyx,
        "observed_peak_zyx": observed_peak_zyx,
        "expected_rot_idx": expected_rot_idx,
        "observed_rot_idx": observed_rot_idx,
    }

    summary_rows = [
        ("Expected peak (z,y,x)", str(expected_peak_zyx)),
        ("Observed peak (z,y,x)", str(observed_peak_zyx)),
        ("Expected rotation index", str(expected_rot_idx)),
        ("Observed rotation index", str(observed_rot_idx)),
        ("Fits visualized", str(len(fits))),
        ("Output template source", "Input template -> rotate -> translate"),
        ("Best/first solution translation (x,y,z)", str(first_solution_translation_xyz)),
        ("Target vs best-fit MAE", f"{float(np.mean(np.abs(case['target'] - top_fit['template']))):.6f}"),
        ("Target vs best-fit max |diff|", f"{float(np.max(np.abs(case['target'] - top_fit['template']))):.6f}"),
        ("LCC max", f"{float(lcc.max()):.6f}"),
        ("LCC min", f"{float(lcc.min()):.6f}"),
        ("Rotation index min/max", f"{int(np.min(rot_idx))} / {int(np.max(rot_idx))}"),
    ]
    summary_table = _build_summary_table(summary_rows)

    report_html = generate_html(target_map, iso, state_path, options, summary_table)
    report_path = output_dir / args.html_name
    report_path.write_text(report_html)

    rel_out = output_dir.relative_to(Path.cwd()) if output_dir.is_relative_to(Path.cwd()) else output_dir
    rel_report = report_path.relative_to(Path.cwd()) if report_path.is_relative_to(Path.cwd()) else report_path

    print("Synthetic visualization generated.")
    print(f"Output directory: {rel_out}")
    print(f"Report file: {rel_report}")
    print("Expected vs observed:")
    print(f"  peak_zyx: expected={expected_peak_zyx} observed={observed_peak_zyx}")
    print(f"  rot_idx:  expected={expected_rot_idx} observed={observed_rot_idx}")
    print(
        "Open report by serving the directory, for example:\n"
        f"  python3 -m http.server -d {rel_out}\n"
        f"Then open: http://localhost:8000/{args.html_name}"
    )


if __name__ == "__main__":
    main()
