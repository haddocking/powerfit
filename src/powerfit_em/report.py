from dataclasses import dataclass
import logging
import os
from pathlib import Path
import csv
import shutil
from string import Template
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


@dataclass
class Iso:
    value: float
    min: float
    max: float
    step: float


def _calc_rel_isovalue(volume_path: Path) -> Iso:
    """Calculate relative iso value from volume file.

    Args:
        volume_path: Path to volume file (.mrc or .map or .ccp4)

    Returns:
        A Iso object.
    """
    p = None
    if volume_path.suffix in [".ccp4", ".map"]:
        p = CCP4Parser(str(volume_path))
    elif volume_path.suffix == ".mrc":
        p = MRCParser(str(volume_path))
    else:
        logger.warning(
            "Could not determine relative iso value for density map, using default values."
        )
        return Iso(2, -10, 10, 0.1)
    # Logic taken from
    # https://github.com/molstar/molstar/blob/e4edb67f62cf92f31ce220f3d250bfd9c8f77572/src/mol-model/volume/volume.ts#L113-L127
    # and ccp4 header mapping from
    # https://github.com/molstar/molstar/blob/e4edb67f62cf92f31ce220f3d250bfd9c8f77572/src/mol-io/reader/ccp4/parser.ts#L70-L94
    amin = p.header["amin"]
    amax = p.header["amax"]
    amean = p.header["amean"]
    sigma = p.header["rms"]
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


def _add_density_to_builder(
    builder: Root, density: Path, rel_iso_value: float
) -> VolumeRepresentation:
    return (
        builder.download(url=density.name)
        .parse(format="map")
        .volume()
        .representation(
            type="isosurface", relative_isovalue=rel_iso_value, show_wireframe=True
        )
        .color(color="gray")
        .opacity(opacity=0.2)
    )


def _add_model_to_builder(
    builder: Root, model: Path, label: str | None = None
) -> Representation:
    component = (
        builder.download(url=model.name)
        .parse(format="pdb")
        .model_structure()
        .component()
    )
    if label:
        component = component.label(text=label)
    # Focus on fitted model with some of the density around it
    return component.focus(radius_factor=3).representation().color(color="blue")


def _create_snapshot_description(solution: dict[str, Any]) -> str:
    translation = f"({solution['x']}, {solution['y']}, {solution['z']})"
    rotation = f"(({solution['a11']}, {solution['a12']}, {solution['a13']}), "
    rotation += f"({solution['a21']}, {solution['a22']}, {solution['a23']}), "
    rotation += f"({solution['a31']}, {solution['a32']}, {solution['a33']}))"
    fitted_model_file = solution["fitted_model_file"]
    return dedent(f"""
            - Rank: {solution["rank"]}
            - Fitted model: [{fitted_model_file.name}]({fitted_model_file.name})
            - Cross correlation score: {solution["cc"]}
            - Fisher z-score: {solution["Fish-z"]}
            - Relative z-score (z-score/α): {solution["rel-z"]}
            - Sigma difference (z₁-zₙ/α): {solution["sigma_dif"]}
            - Translation: {translation}
            - Rotation: {rotation}
        """)


def create_snapshot(solution: dict, density: Path, rel_iso_value: float) -> Snapshot:
    builder = create_builder()
    fitted_model_file = solution["fitted_model_file"]
    _add_density_to_builder(builder, density, rel_iso_value)
    _add_model_to_builder(builder, fitted_model_file)
    description = _create_snapshot_description(solution)
    return builder.get_snapshot(
        key=fitted_model_file.stem,
        title=solution["rank"],
        description=description,
        description_format="markdown",
    )


def create_snapshot_with_all_models(
    fitted_model_files: list[Path],
    density: Path,
    rel_iso_value: float,
    max_models: int = 100,
) -> Snapshot:
    builder = create_builder()
    _add_density_to_builder(builder, density, rel_iso_value)
    for fitted_model_file in fitted_model_files[:max_models]:
        label = fitted_model_file.stem.replace("fit_", "")
        _add_model_to_builder(builder, fitted_model_file, label=label)
    maxed = len(fitted_model_files) > max_models
    description_first_line = (
        f"First {max_models} fitted models." if maxed else "All fitted models."
    )
    description = dedent(f"""
        {description_first_line}

        The labels correspond to the rank of the fitted model.

        Can take a while to load the models.
    """)
    return builder.get_snapshot(
        key="all_models",
        title=f"First {max_models} fitted models" if maxed else "All fitted models",
        description=description,
        description_format="markdown",
    )


def _read_solutions(path: Path, delimiter: str | None = None) -> list[dict]:
    if delimiter is None:
        raise ValueError("Delimiter must be set to produce a report.")
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        solutions = list(reader)
    # Calculate sigma_dif for each solution and add fitted_model_file
    if solutions:
        best_z = float(solutions[0]["rel-z"])
        for i, solution in enumerate(solutions):
            solution["sigma_dif"] = round(best_z - float(solution["rel-z"]) , 3)
            solution["fitted_model_file"] = Path(path.parent) / f"fit_{i + 1}.pdb"
    return solutions


def generate_html(
    target_path: Path,
    iso: Iso,
    state_path: Path,
    options: dict[str, Any],
    solutions_table: str,
):
    # template was copied from molviewspec.molstar_widgets.STORIES_TEMPLATE and heavily modified
    template = Template(
        dedent("""\
            <!DOCTYPE html>
            <html lang="en">

            <head>
                <style>
                    * {
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }

                    html,
                    body {
                        height: 100%;
                        width: 100%;
                    }

                    body {
                        min-height: 100vh;
                        min-width: 100vw;
                        display: flex;
                        flex-direction: column;
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

                    .options li {
                        list-style-position: inside;
                    }

                    .container {
                        flex: 1 1 auto;
                        display: flex;
                        flex-direction: row;
                        width: 100vw;
                        height: 0;
                        /* let flexbox control height */
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

                    #solutions-table {
                        table {
                            border-collapse: collapse;
                            border: 1px solid #ccc;
                            margin: 1rem;

                            th, td {
                                border: 1px solid rgb(160 160 160);
                                padding: 8px 10px;
                            }
                        }
                        .close-sigma-lt1 {
                            background-color: #b2e5b2; /* lighter green */
                            font-weight: bold;
                        }
                        .close-sigma-1to2 {
                            background-color: #c8f7c8; /* lighter medium green */
                            font-weight: bold;
                        }
                        .close-sigma-2to3 {
                            background-color: #e6ffe6; /* very light green */
                            font-weight: bold;
                        }
                        button {
                            display: inline-block;
                            padding: 0.25rem 0.5rem;
                            text-align: center;
                            font-size: 11px;
                            font-weight: 600;
                            letter-spacing: .1rem;
                            text-transform: uppercase;
                            text-decoration: none;
                            white-space: nowrap;
                            border-radius: 4px;
                            border: 1px solid #bbb;
                            cursor: pointer;
                            box-sizing: border-box
                        }
                    }

                    @media (orientation:portrait),
                    (max-width: 900px) {
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
                <script src="https://cdn.jsdelivr.net/npm/molstar@${molstar_version}/build/mvs-stories/mvs-stories.js"></script>
                <link rel="stylesheet" type="text/css"
                    href="https://cdn.jsdelivr.net/npm/molstar@${molstar_version}/build/mvs-stories/mvs-stories.css" />
            </head>

            <body>
                <header>
                    <h4>PowerFit Report</h4>
                    <details title="Options used for fitting">
                        <summary>Options</summary>
                        <span>The following options were used for fitting:</span>
                        <ul class="options">
                            ${options}
                        </ul>
                    </details>
                    <div>
                        Download:
                        <a href="${target}" target="_blank" rel="noreferrer">density map</a>
                        <a href="solutions.out" target="_blank" rel="noreferrer">solutions</a>
                        <a href="lcc.mrc" target="_blank" rel="noreferrer">cross-correlation</a>
                    </div>
                    <label title="Adjust the iso value of the density map.">
                        Iso value
                        <input id="iso-slider" type="range" min="${iso_min}" max="${iso_max}" value="${iso_value}"
                            step="${iso_step}" />
                        <input id="iso-number" type="number" min="${iso_min}" max="${iso_max}" value="${iso_value}"
                            step="${iso_step}" />
                    </label>
                </header>
                <div class="container">
                    <div id="viewer">
                        <mvs-stories-viewer />
                    </div>
                    <div id="controls">
                        <mvs-stories-snapshot-markdown style="flex-grow: 1;" />
                    </div>
                </div>
               <div id="solutions-table">
               ${solutions}
               </div>
                <script>
                    mvsStories.loadFromURL('${state}', { format: 'mvsj' });

                    let listenerAdded = false;
                    const slider = document.getElementById('iso-slider');
                    const number = document.getElementById('iso-number');
                    let currentIsoValue = parseFloat(number.value);

                    function updateIsoValue(newIsoValue) {
                        const context = mvsStories.getContext();
                        const viewer = context.state.viewers.value[0].model;
                        const plugin = viewer.plugin;
                        // Find and update all isosurface representations in the Mol* viewer
                        const state = viewer.plugin.state.data;
                        const isosurfaceNodes = Array.from(viewer.plugin.state.data.cells.values().filter(v => v.obj.label === 'Isosurface'))

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
                            viewer.subscribe(viewer.plugin.managers.snapshot.events.changed, () => {
                                // wait a frame to apply iso so volume is present
                                requestAnimationFrame(() => {
                                    updateIsoValue(currentIsoValue);
                                });
                            });
                            listenerAdded = true;
                        }
                    }

                    slider.addEventListener('input', (event) => {
                        const newIsoValue = parseFloat(event.target.value);
                        if (newIsoValue === currentIsoValue) {
                            return;
                        }
                        currentIsoValue = newIsoValue;
                        updateIsoValue(currentIsoValue);
                        if (number.value !== event.target.value) {
                            number.value = currentIsoValue;
                        }
                    });
                    number.addEventListener('input', (event) => {
                        const newIsoValue = parseFloat(event.target.value);
                        if (newIsoValue === currentIsoValue) {
                            return;
                        }
                        currentIsoValue = newIsoValue;
                        updateIsoValue(currentIsoValue);
                        if (slider.value !== event.target.value) {
                            slider.value = currentIsoValue;
                        }
                    });

                    function jumpToSnapshot(key) {
                        const context = mvsStories.getContext();
                        const model = context.state.viewers.value[0].model;
                        model.plugin?.managers.snapshot.entryMap.forEach((entry, entryId) => {
                            if (entry.key === key) {
                                const snapshot = model.plugin?.managers.snapshot.setCurrent(entryId);
                                if (snapshot) {
                                    model.plugin.state.setSnapshot(snapshot);
                                }
                            }
                        })
                    }

                    function updateButtonStates(currentKey) {
                        // Enable all buttons first
                        document.querySelectorAll('#solutions-table button').forEach(button => {
                            button.disabled = false;
                            button.style.opacity = '1';
                            button.style.cursor = 'pointer';
                            button.title = 'View this solution in 3D viewer';
                        });

                        // Disable the button for the current snapshot
                        if (currentKey) {
                            const currentButton = document.querySelector(`#solutions-table button[data-snapshot-key="${currentKey}"]`);
                            if (currentButton) {
                                currentButton.disabled = true;
                                currentButton.style.opacity = '0.5';
                                currentButton.style.cursor = 'not-allowed';
                                currentButton.title = 'This solution is currently displayed';
                            }
                        }
                    }

                    function setupSnapshotListener() {
                        const context = mvsStories.getContext();
                        if (!context?.state?.viewers?.value?.[0]?.model) {
                            // Retry after a short delay if context is not ready
                            setTimeout(setupSnapshotListener, 100);
                            return;
                        }

                        const model = context.state.viewers.value[0].model;
                        const plugin = model.plugin;

                        if (plugin?.managers?.snapshot) {
                            // Update button states when snapshot changes
                            plugin.managers.snapshot.subscribe(plugin.managers.snapshot.events.changed, (newState) => {
                                const key = plugin.managers.snapshot.current.key;
                                updateButtonStates(key);
                            });

                            // Set initial state
                            const initialKey = plugin.managers.snapshot.current.key;
                            updateButtonStates(initialKey);
                        }
                    }
                    setupSnapshotListener();
                </script>
            </body>

            </html>
    """)
    )
    li_options = "\n".join(
        [f"<li>{key}: {value}</li>" for key, value in options.items()]
    )

    return template.safe_substitute(
        state=state_path.name,
        target=target_path.name,
        iso_min=round(iso.min, 2),
        iso_max=round(iso.max, 2),
        iso_value=round(iso.value, 2),
        iso_step=round(iso.step, 2),
        options=li_options,
        molstar_version="4.18.0",
        solutions=solutions_table,
    )


def generated_table(solutions: list[dict[str, Any]]) -> str:
    if not solutions:
        return "<p>No solutions found.</p>"
    table = dedent("""\
        <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Cross correlation score</th>
            <th>Fisher z-score</th>
            <th>Relative z-score (z-score/&alpha;)</th>
            <th>Sigma difference (z<sub>1</sub>-z<sub>N</sub>/&alpha;)</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
        """)
    for solution in solutions:
        sigma_dif = solution["sigma_dif"]
        if sigma_dif < 1:
            class_name = "close-sigma-lt1"
        elif sigma_dif < 2:
            class_name = "close-sigma-1to2"
        elif sigma_dif < 3:
            class_name = "close-sigma-2to3"
        else:
            class_name = "close-sigma-gt3"
        snapshot_key = f"fit_{solution['rank']}"
        fitted_model_file = solution["fitted_model_file"]
        table += dedent(f"""\
            <tr class="{class_name}">
              <td><a href="{fitted_model_file.name}" target="_blank" title="Download fitted model">{solution["rank"]}</a></td>
              <td>{solution["cc"]}</td>
              <td>{solution["Fish-z"]}</td>
              <td>{solution["rel-z"]}</td>
              <td>{sigma_dif}</td>
              <td><button data-snapshot-key="{snapshot_key}" onclick="jumpToSnapshot('{snapshot_key}')">View in 3D</button></td>
            </tr>
        """)
    table += "</tbody></table>\n"
    return table


def generate_report(
    directory: str,
    target: str,
    num: int,
    delimiter: str | None,
    options: dict[str, Any],
):
    """
    Generate a HTML report of the fitting results.

    The report uses Mol* and MolViewSpec stories.

    Args:
        directory: Directory containing the fitting results.
        target: Target volume file name.
        num: Number of fits to include in the report.
        delimiter: Delimiter used in the solutions file. If None, the report raise an error.
        options: Options used for fitting, to include in the report.
    """
    run_dir = Path(directory)

    # To render report all files need to be in the same directory
    target_path = run_dir / Path(target).name
    if not target_path.exists():
        rtarget = Path(os.path.relpath(target, Path.cwd()))
        rrun_dir = Path(os.path.relpath(run_dir, Path.cwd()))
        logger.warning(
            f"Copying target file ({rtarget}) to report directory ({rrun_dir})."
        )
        shutil.copyfile(target, target_path)

    iso = _calc_rel_isovalue(target_path)

    solutions_file = run_dir / "solutions.out"
    solutions = _read_solutions(solutions_file, delimiter)
    fitted_model_files = [solution["fitted_model_file"] for solution in solutions[:num]]
    snapshots = []
    for i, solution in enumerate(solutions[:num]):
        snapshot = create_snapshot(solution, target_path, iso.value)
        snapshots.append(snapshot)
    snapshots.append(
        create_snapshot_with_all_models(fitted_model_files, target_path, iso.value)
    )
    # could add snapshot with lcc.mrc,
    # but could not make tutorial map look interesting,
    # so skip for now

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
    state_path = run_dir / "state.mvsj"
    state_path.write_text(state.dumps(indent=2))

    solutions_table = generated_table(solutions[:num])

    report = run_dir / "report.html"
    body = generate_html(target_path, iso, state_path, options, solutions_table)
    report.write_text(body)

    rel_report = Path(os.path.relpath(report, Path.cwd()))
    rel_run_dir = Path(os.path.relpath(run_dir, Path.cwd()))
    logger.warning(
        f"Report generated at {rel_report}. Start web server with "
        f"`python3 -m http.server -d {rel_run_dir}`. "
        "Open http://localhost:8000/report.html in a web browser to view the results."
    )


if __name__ == "__main__":
    import argparse
    from argparse import FileType

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=dedent("""\
            Generate report.html from fitting results.
                           
            Example usage:
            
            powerfit ribosome-KsgA.map 13 KsgA.pdb -d run -n 20 --delimiter , -a 20 -l
            # Later generate report.html with
            python3 -m powerfit_em.report run ribosome-KsgA.map -n 20 --delimiter , --option resolution 13 --option angle 20 --option laplace True
        """),
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory with fitting results.",
    )
    parser.add_argument(
        "target",
        type=str,
        help="Target density map to fit the model in. "
        "Data should either be in CCP4 or MRC format",
    )
    parser.add_argument(
        "-n",
        "--num",
        dest="num",
        type=int,
        default=10,
        metavar="<int>",
        help="Number of models written to file. This number "
        "will be capped if less solutions are found as requested.",
    )
    parser.add_argument(
        "--delimiter",
        dest="delimiter",
        type=str,
        default=",",
        metavar="<str>",
        help="Delimiter used in the 'solutions.out' file. For example use ',' or '\\t'. Defaults to fixed width.",
    )
    parser.add_argument(
        "--option",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=[],
        help="Option used for fitting, to include in the report. "
        "Can be used multiple times. Example: --option resolution 13",
    )
    args = parser.parse_args()

    generate_report(
        args.directory, args.target, args.num, args.delimiter, dict(args.option)
    )
