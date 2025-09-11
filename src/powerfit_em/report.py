from dataclasses import dataclass
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

from powerfit_em.volume import CCP4Parser, MRCParser

logger = logging.getLogger(__name__)

# Copied from molviewspec.molstar_widgets.STORIES_TEMPLATE for customization
POWERFIT_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        html, body {
            height: 100%;
            width: 100%;
        }

        body {
            min-height: 100vh;
            min-width: 100vw;
            display: flex;
            flex-direction: column;
            background: #fff;
        }

        header {
            background: #F6F5F3;
            padding: 1rem;
            border-bottom: 1px solid #ccc;
            display: flex;
            justify-content: space-between;
        }

        #iso-number {
          width: 4rem;
        }

        .container {
            flex: 1 1 auto;
            display: flex;
            flex-direction: row;
            width: 100vw;
            height: 0; /* let flexbox control height */
            min-height: 0;
        }

        #viewer {
            flex: 2 1 0;
            min-width: 0;
            position: relative;
            height: 100%;
        }

        #controls {
            flex: 1 1 0;
            min-width: 320px;
            max-width: 600px;
            padding: 16px;
            padding-bottom: 20px;
            border-left: 1px solid #ccc;
            background: #F6F5F3;
            display: flex;
            flex-direction: column;
            gap: 16px;
            height: 100%;
            box-sizing: border-box;
        }

        @media (orientation:portrait), (max-width: 900px) {
            .container {
                flex-direction: column;
            }
            #viewer {
                flex: 1 1 0;
                min-height: 300px;
                height: 50vh;
            }
            #controls {
                flex: 1 1 0;
                min-width: 0;
                max-width: none;
                border-left: none;
                border-top: 1px solid #ccc;
                height: auto;
            }
            .msp-viewport-controls-buttons {
                display: none;
            }
        }
    </style>
    <script src=\"https://cdn.jsdelivr.net/npm/molstar@{{version}}/build/mvs-stories/mvs-stories.js\"></script>
    <link rel=\"stylesheet\" type=\"text/css\" href=\"https://cdn.jsdelivr.net/npm/molstar@{{version}}/build/mvs-stories/mvs-stories.css\" />
</head>
<body>
    <header>
        <h4>PowerFit Report</h4>
        <a href=\"{{target}}\" target=\"_blank\" rel=\"noreferrer\" >Download density map</a>
        <label>
            Iso value
            <input id="iso-slider" type="range" min="{{iso.min}}" max="{{iso.max}}" value="{{iso.value}}" step="{{iso.step}}" />
            <input id="iso-number" type="number" min="{{iso.min}}" max="{{iso.max}}" value="{{iso.value}}" step="{{iso.step}}" />
        </label>
    </header>
    <div class=\"container\">
        <div id=\"viewer\">
            <mvs-stories-viewer />
        </div>
        <div id=\"controls\">
            <mvs-stories-snapshot-markdown style=\"flex-grow: 1;\" />
        </div>
    </div>

    <script>
        var mvsData = {{state}};

        mvsStories.loadFromData(mvsData, { format: '{{format}}' });

        let listenerAdded = false;
        const slider = document.getElementById('iso-slider');
        const number = document.getElementById('iso-number');

        function updateIsoValue(newIsoValue) {
            const context = mvsStories.getContext();
            const viewer = context.state.viewers.value[0].model;
            const plugin = viewer.plugin;
            // Find and update all isosurface representations in the Mol* viewer
            const state = viewer.plugin.state.data;
            const isosurfaceNodes =  Array.from(viewer.plugin.state.data.cells.values().filter(v => v.obj.label === 'Isosurface'))

            // Update the relativeIsoValue for each isosurface
            const b = state.build();

            isosurfaceNodes.forEach(cell => {
                const params = cell.params?.values;
                if (!params?.type?.params?.isoValue) return;

                const nextParams = {
                    ...params,
                    type: {
                        ...params.type,
                        params: {
                            ...params.type.params,
                            isoValue: {
                                kind: 'relative',
                                relativeValue: newIsoValue
                            }
                        }
                    }
                };

                b.to(cell.transform.ref).update(nextParams);
            });
            plugin.runTask(state.updateTree(b));

            if (!listenerAdded) {
                viewer.subscribe(viewer.model.plugin?.managers.snapshot.events.changed, () => {
                    updateIsoValue(parseFloat(number.value));
                });
                listenerAdded = true;
            }
        }

        // when changing snapshot, the iso value is reset, but the slider/number are not updated
        // TODO after new snapshot is loaded, the iso value in volume to same as slider/number
        slider.addEventListener('input', (event) => {
            const value = parseFloat(event.target.value);
            updateIsoValue(value);
            if (number.value !== event.target.value) {
                number.value = value;
            }
        });
        number.addEventListener('input', (event) => {
            const value = parseFloat(event.target.value);
            updateIsoValue(value);
            if (slider.value !== event.target.value) {
                slider.value = value;
            }
        });
    </script>
</body>
</html>
"""

@dataclass
class Iso:
    value: float
    min: float
    max: float
    step: float


def _calc_rel_isovalue(volume_path: Path) -> Iso:
    """Calculate relative isovalue, min and max from volume file.
    
    Args:
        volume_path: Path to volume file (.mrc or .map or .ccp4)

    Returns:
        A Iso object.
    """
    if (volume_path.suffix in [".ccp4", ".map"]):
        p = CCP4Parser(str(volume_path))
    elif volume_path.suffix == ".mrc":
        p = MRCParser(str(volume_path))
    else:
        logger.warning("Could not determine relative iso value for density map, using default values.")
        return Iso(2, -10, 10, 0.1)  # default values
    # Logic taken from
    # https://github.com/molstar/molstar/blob/e4edb67f62cf92f31ce220f3d250bfd9c8f77572/src/mol-model/volume/volume.ts#L113-L127
    # and ccp4 header mapping from
    # https://github.com/molstar/molstar/blob/e4edb67f62cf92f31ce220f3d250bfd9c8f77572/src/mol-io/reader/ccp4/parser.ts#L70-L94
    amin = p.header['amin']
    amax = p.header['amax']
    amean = p.header['amean']
    sigma = p.header['rms']
    rel_min = (amin - amean) / sigma
    rel_max = (amax - amean) / sigma
    rel_isovalue = 2
    if sigma == 0:
        rel_isovalue = 0
    if rel_isovalue < rel_min:
        rel_isovalue = rel_min
    if rel_isovalue > rel_max:
        rel_isovalue = rel_max
    steps = 100
    step = round((amax - amin) / sigma / steps, 2)    
    return Iso(rel_isovalue, rel_min, rel_max, step)


def _add_density_to_builder(builder: Root, density: Path, rel_iso_value: float) -> VolumeRepresentation:
    return (
        builder.download(url=density.name)
        .parse(format="map")
        .volume()
        # the right isovalue depends on the density
        # You can use molstar to change it in many steps:
        # Controls panel > State Tree > Isosurface > Update 3D representation > ... of type > slide isovalue
        # TODO add a slider in a copy of STORIES_TEMPLATE to easy change it
        .representation(type="isosurface", relative_isovalue=rel_iso_value, show_wireframe=True)
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


def create_snapshot(solution: dict, fitted_model_file: Path, density: Path, rel_iso_value: float) -> Snapshot:
    builder = create_builder()
    _add_density_to_builder(builder, density, rel_iso_value)
    _add_model_to_builder(builder, fitted_model_file)
    description = _create_snapshot_description(solution, fitted_model_file)
    return builder.get_snapshot(
        key=fitted_model_file.stem,
        title=solution["rank"],
        description=description,
        description_format="markdown",
    )

def create_snapshot_with_all_models(fitted_model_files: list[Path], density: Path, rel_iso_value: float) -> Snapshot:
    builder = create_builder()
    _add_density_to_builder(builder, density, rel_iso_value)
    for fitted_model_file in fitted_model_files:
        _add_model_to_builder(builder, fitted_model_file)
    description = "All fitted models"
    return builder.get_snapshot(
        key="all_models",
        title="All models",
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

    iso = _calc_rel_isovalue(target_path)

    solutions_file = run_dir / "solutions.out"
    solutions = _read_solutions(solutions_file, delimiter)
    fitted_model_files = []
    snapshots = []
    for i, solution in enumerate(solutions):
        if i >= num:
            # Can only visualize written models
            break
        fitted_model_file = run_dir / f"fit_{i + 1}.pdb"
        fitted_model_files.append(fitted_model_file)
        snapshot = create_snapshot(solution, fitted_model_file, target_path, iso.value)
        snapshots.append(snapshot)
    snapshots.append(
        create_snapshot_with_all_models(fitted_model_files, target_path, iso.value)
    )

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
    template = POWERFIT_TEMPLATE
    # TODO add download link for density map in template
    format = "mvsj"
    molstar_version: str = "latest"
    body = (
        template.replace("{{version}}", molstar_version)
        .replace("{{format}}", format)
        .replace("{{state}}", state.dumps(indent=2))
        .replace("{{target}}", target_path.name)
        .replace("{{iso.min}}", str(iso.min))
        .replace("{{iso.max}}", str(iso.max))
        .replace("{{iso.value}}", str(iso.value))
        .replace("{{iso.step}}", str(iso.step))
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
