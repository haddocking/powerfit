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
from powerfit_em._extensions import rotate_grid3d
from powerfit_em.report import _add_density_to_builder, _calc_rel_isovalue, generate_html
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


def _write_solution_stub(path: Path, observed_rot_idx: int, observed_peak_zyx: tuple[int, int, int]) -> None:
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
        writer.writerow(
            {
                "rank": 1,
                "cc": "n/a",
                "Fish-z": "n/a",
                "rel-z": "n/a",
                "x": observed_peak_zyx[2],
                "y": observed_peak_zyx[1],
                "z": observed_peak_zyx[0],
                "a11": "n/a",
                "a12": "n/a",
                "a13": "n/a",
                "a21": "n/a",
                "a22": "n/a",
                "a23": "n/a",
                "a31": "n/a",
                "a32": "n/a",
                "a33": "n/a",
            }
        )


def _density_snapshot(key: str, title: str, volume_path: Path, rel_iso_value: float, description: str) -> Snapshot:
    builder = create_builder()
    _add_density_to_builder(builder, volume_path, rel_iso_value)
    return builder.get_snapshot(
        key=key,
        title=title,
        description=description,
        description_format="markdown",
    )


def _add_colored_density(builder, volume_path: Path, rel_iso_value: float, color: str):
    return (
        builder.download(url=volume_path.name)
        .parse(format="map")
        .volume()
        .representation(type="isosurface", relative_isovalue=rel_iso_value, show_wireframe=True)
        .color(color=color)
        .opacity(opacity=0.35)
    )


def _overlay_snapshot(
    *,
    key: str,
    title: str,
    target_path: Path,
    target_rel_iso: float,
    template_path: Path,
    template_rel_iso: float,
    description: str,
) -> Snapshot:
    builder = create_builder()
    _add_density_to_builder(builder, target_path, target_rel_iso)
    _add_colored_density(builder, template_path, template_rel_iso, "blue")
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


def _transform_input_template(
    input_template: np.ndarray,
    rotation: np.ndarray,
    translation_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build output template by applying solution transform to input template.

    Returns:
        (rotated_template, transformed_template)
    """
    rotated_template = np.zeros_like(input_template, dtype=np.float32)
    rotate_grid3d(
        input_template,
        rotation,
        min(input_template.shape) // 2,
        rotated_template,
        False,
    )
    transformed_template = np.roll(
        rotated_template,
        shift=translation_zyx,
        axis=(0, 1, 2),
    ).astype(np.float32, copy=False)
    return rotated_template, transformed_template


def _transform_template_write_fits_style(
    input_template: np.ndarray,
    rotation_xyz: np.ndarray,
    translation_xyz: tuple[float, float, float],
) -> np.ndarray:
    """Apply the same high-level transform sequence as write_fits_to_pdb.

    Sequence: center coordinates, rotate, translate.
    """
    nz = np.argwhere(input_template > 0)
    if nz.size == 0:
        return np.zeros_like(input_template, dtype=np.float32)

    # Convert ZYX voxel indices to XYZ coordinates for rotation matrix usage.
    coords_xyz = np.stack((nz[:, 2], nz[:, 1], nz[:, 0]), axis=1).astype(np.float32)
    values = input_template[nz[:, 0], nz[:, 1], nz[:, 2]].astype(np.float32)

    center_xyz = coords_xyz.mean(axis=0)
    centered_xyz = coords_xyz - center_xyz
    rotated_xyz = centered_xyz @ rotation_xyz.T
    moved_xyz = rotated_xyz + np.asarray(translation_xyz, dtype=np.float32)

    moved_xyz_idx = np.rint(moved_xyz).astype(np.int32)
    shape_zyx = input_template.shape
    shape_xyz = (shape_zyx[2], shape_zyx[1], shape_zyx[0])
    x = np.mod(moved_xyz_idx[:, 0], shape_xyz[0])
    y = np.mod(moved_xyz_idx[:, 1], shape_xyz[1])
    z = np.mod(moved_xyz_idx[:, 2], shape_xyz[2])

    out = np.zeros_like(input_template, dtype=np.float32)
    np.maximum.at(out, (z, y, x), values)
    return out


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

    solution_rot = case["rotations"][observed_rot_idx]
    rotated_template, transformed_template = _transform_input_template(
        case["template"],
        solution_rot,
        observed_peak_zyx,
    )
    first_solution_translation_xyz = (
        float(observed_peak_zyx[2]),
        float(observed_peak_zyx[1]),
        float(observed_peak_zyx[0]),
    )
    transformed_template_write_fits = _transform_template_write_fits_style(
        case["template"],
        solution_rot,
        first_solution_translation_xyz,
    )

    target_map = output_dir / "target.mrc"
    template_map = output_dir / "template.mrc"
    rotated_template_map = output_dir / "template_rotated.mrc"
    transformed_template_map = output_dir / "template_solution.mrc"
    transformed_template_write_fits_map = output_dir / "template_solution_write_fits.mrc"
    diff_target_solution_map = output_dir / "diff_target_minus_template_solution.mrc"
    lcc_map = output_dir / "lcc.mrc"
    rot_map = output_dir / "rotation_index.mrc"

    _write_map(target_map, case["target"])
    _write_map(template_map, case["template"])
    _write_map(rotated_template_map, rotated_template)
    _write_map(transformed_template_map, transformed_template)
    _write_map(transformed_template_write_fits_map, transformed_template_write_fits)
    _write_map(diff_target_solution_map, case["target"] - transformed_template)
    _write_map(lcc_map, lcc)
    _write_map(rot_map, rot_idx)

    iso = _calc_rel_isovalue(target_map)
    template_iso = _calc_rel_isovalue(template_map)
    transformed_template_iso = _calc_rel_isovalue(transformed_template_map)
    transformed_template_write_fits_iso = _calc_rel_isovalue(transformed_template_write_fits_map)
    diff_iso = _calc_rel_isovalue(diff_target_solution_map)

    snapshots = [
        _density_snapshot(
            key="target",
            title="Synthetic Target",
            volume_path=target_map,
            rel_iso_value=iso.value,
            description="Target volume generated by rotating and rolling the synthetic template.",
        ),
        _density_snapshot(
            key="template_rotated",
            title="Template Rotated (No Translation)",
            volume_path=rotated_template_map,
            rel_iso_value=template_iso.value,
            description="Input template after applying the output solution rotation only.",
        ),
        _density_snapshot(
            key="template",
            title="Synthetic Template",
            volume_path=template_map,
            rel_iso_value=template_iso.value,
            description="Asymmetric synthetic template used for deterministic correlator checks.",
        ),
        _overlay_snapshot(
            key="overlay_input",
            title="Target + Input Template",
            target_path=target_map,
            target_rel_iso=iso.value,
            template_path=template_map,
            template_rel_iso=template_iso.value,
            description="Overlay view: target in gray and input template in blue.",
        ),
        _overlay_snapshot(
            key="overlay_solution",
            title="Target + Output Template",
            target_path=target_map,
            target_rel_iso=iso.value,
            template_path=transformed_template_map,
            template_rel_iso=transformed_template_iso.value,
            description=(
                "Overlay view: target in gray and solution-transformed template in blue. "
                "In this synthetic setup they are expected to be visually very similar when aligned."
            ),
        ),
        _overlay_snapshot(
            key="overlay_solution_write_fits",
            title="Target + Output Template (Best/First Solution)",
            target_path=target_map,
            target_rel_iso=iso.value,
            template_path=transformed_template_write_fits_map,
            template_rel_iso=transformed_template_write_fits_iso.value,
            description=(
                "Overlay view using write_fits_to_pdb-style semantics: center input template, "
                "rotate by best/first solution rotation, then translate by first solution (x,y,z)."
            ),
        ),
        _density_snapshot(
            key="diff_target_solution",
            title="Difference: Target - Output Template",
            volume_path=diff_target_solution_map,
            rel_iso_value=diff_iso.value,
            description="Non-zero regions indicate where output template differs from target.",
        ),
        _density_snapshot(
            key="lcc",
            title="LCC Output",
            volume_path=lcc_map,
            rel_iso_value=iso.value,
            description="Local cross-correlation field for the selected backend.",
        ),
        _density_snapshot(
            key="rotation",
            title="Rotation Index Output",
            volume_path=rot_map,
            rel_iso_value=iso.value,
            description="Per-voxel best rotation-candidate index selected by the correlator.",
        ),
    ]

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

    _write_solution_stub(output_dir / "solutions.out", observed_rot_idx, observed_peak_zyx)

    options = {
        "backend": args.backend,
        "batch_size": args.batch_size,
        "cuda_device": args.cuda_device,
        "opencl_device": args.opencl_device,
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
        ("Output template source", "Input template -> rotate -> translate"),
        ("Best/first solution translation (x,y,z)", str(first_solution_translation_xyz)),
        ("Target vs output-template MAE", f"{float(np.mean(np.abs(case['target'] - transformed_template))):.6f}"),
        ("Target vs output-template max |diff|", f"{float(np.max(np.abs(case['target'] - transformed_template))):.6f}"),
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
