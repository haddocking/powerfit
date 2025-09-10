import logging
import os
from pathlib import Path
import csv
import shutil
from textwrap import dedent
from typing import Any

from molviewspec import (
    MVSJ,
    GlobalMetadata,
    Snapshot,
    States,
    create_builder,
)
from molviewspec.builder import Representation, Root, VolumeRepresentation
from molviewspec.molstar_widgets import STORIES_TEMPLATE

logger = logging.getLogger(__name__)


def _add_density_to_builder(builder: Root, density: Path) -> VolumeRepresentation:
    return (
        builder.download(url=density.name)
        .parse(format="map")
        .volume()
        .representation(type="isosurface", relative_isovalue=3, show_wireframe=True)
        .color(color="gray")
        .opacity(opacity=0.2)
    )


def _add_model_to_builder(builder: Root, model: Path) -> Representation:
    return (
        builder.download(url=model.name)
        .parse(format="pdb")
        .model_structure()
        .component()
        # Focus on fitted model with some of the density around it
        .focus(radius_factor=3)
        .representation()
        .color(color="blue")
    )


def _create_snapshot_description(
    solution: dict[str, Any], fitted_model_file: Path
) -> str:
    translation = f"({solution['x']}, {solution['y']}, {solution['z']})"
    rotation = f"(({solution['a11']}, {solution['a12']}, {solution['a13']}), "
    rotation += f"({solution['a21']}, {solution['a22']}, {solution['a23']}), "
    rotation += f"({solution['a31']}, {solution['a32']}, {solution['a33']}))"
    return dedent(f"""
            - Rank: {solution["rank"]}
            - Fitted model: [{fitted_model_file.name}]({fitted_model_file.name})
            - Cross correlation score: {solution["cc"]}
            - Fish-z score: {solution["Fish-z"]}
            - Rel-z score: {solution["rel-z"]}
            - Translation: {translation}
            - Rotation: {rotation}
        """)


def create_snapshot(solution: dict, fitted_model_file: Path, density: Path) -> Snapshot:
    builder = create_builder()
    _add_density_to_builder(builder, density)
    _add_model_to_builder(builder, fitted_model_file)
    description = _create_snapshot_description(solution, fitted_model_file)
    return builder.get_snapshot(
        key=fitted_model_file.stem,
        title=solution["rank"],
        description=description,
        description_format="markdown",
    )


def generate_report(
    directory: str, target: str, num: int, delimiter: str | None = None
):
    """
    Generate a HTML report of the fitting results.

    The report uses Mol* and MolViewSpec stories.

    Args:
        directory: Directory containing the fitting results.
        target: Target volume file name.
        num: Number of fits to include in the report.
        delimiter: Delimiter used in the solutions file. If None, the report raise an error.
    """
    run_dir = Path(directory)

    # To render report all files need to be in the same directory
    logger.warning(f"Copying target file ({target}) to report directory.")
    target_path = run_dir / target
    shutil.copyfile(target, target_path)

    solutions_file = run_dir / "solutions.out"
    solutions = _read_solutions(solutions_file, delimiter)
    snapshots = []
    for i, solution in enumerate(solutions):
        if i >= num:
            # Can only visualize written models
            break
        fitted_model_file = run_dir / f"fit_{i + 1}.pdb"
        snapshot = create_snapshot(solution, fitted_model_file, target_path)
        snapshots.append(snapshot)

    state = MVSJ(
        data=States(
            snapshots=snapshots,
            metadata=GlobalMetadata(
                title="",
                description="Fitted models and density map",
                description_format="markdown",
            ),
        )
    )

    report = run_dir / "report.html"
    template = STORIES_TEMPLATE
    format = "mvsj"
    molstar_version: str = "latest"
    body = (
        template.replace("{{version}}", molstar_version)
        .replace("{{format}}", format)
        .replace("{{state}}", state.dumps(indent=2))
    )
    report.write_text(body)

    rel_report = Path(os.path.relpath(report, Path.cwd()))
    rel_run_dir = Path(os.path.relpath(run_dir, Path.cwd()))
    logger.warning(
        f"Report generated at {rel_report}. Start web server with "
        f"`python3 -m http.server -d {rel_run_dir}`. "
        "Open http://localhost:8000/report.html in a web browser to view the results."
    )


def _read_solutions(path: Path, delimiter: str | None = None) -> list[dict]:
    if delimiter is None:
        raise ValueError("Delimiter must set to produce a report.")
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        solutions = list(reader)
    return solutions
