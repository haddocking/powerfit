import json
import logging
import re
from importlib.util import find_spec
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import binary_erosion

if TYPE_CHECKING:
    from IPython.display import IFrame

logger = logging.getLogger(__name__)


def pyfftw_available() -> bool:
    return find_spec("pyfftw") is not None


def pyodide_available() -> bool:
    """Check if `pyodide` is available."""
    return find_spec("js") is not None


def make_blob_url(data: bytes, mime: str = "application/octet-stream") -> str:
    """Wrap bytes in a JS Blob and return an object URL that can be fetch()'d."""
    if not pyodide_available():
        msg = "make_blob_url() requires running under Pyodide (js module not found)."
        raise RuntimeError(msg)

    import js  # pyright: ignore[reportMissingImports]
    from pyodide.ffi import to_js  # pyright: ignore[reportMissingImports]

    # convrt byte data to a js array
    array = to_js(data)
    # make valid options for the object
    options = to_js({"type": mime}, dict_converter=js.Object.fromEntries)

    blob = js.Blob.new([array], options)

    return js.URL.createObjectURL(blob)


def patch_download_urls(node: dict, base_dir: str | Path) -> None:
    """Rewrite `download` nodes' to blob URLs."""
    if node.get("kind") == "download":
        url = node["params"]["url"]
        data = Path(base_dir, url).read_bytes()
        node["params"]["url"] = make_blob_url(data)

    for child in node.get("children") or []:
        # little recursion here doesn't hurt
        patch_download_urls(child, base_dir)


def blobify_local_refs(html: str, base_dir: str | Path) -> str:
    """Rewrite href/src attributes to blob URLs, so links/downloads work without a web server."""

    # nested function here so we go over `base_dir` more easily
    def repl(match: re.Match) -> str:
        attr, filename = match.group(1), match.group(2)
        if filename.startswith(("http://", "https://", "#")):
            return match.group(0)
        path = Path(base_dir, filename)
        if not path.exists():
            return match.group(0)
        return f'{attr}="{make_blob_url(path.read_bytes())}"'

    return re.sub(r'(href|src)="([^"]+)"', repl, html)


def render_in_jupyterlite(directory: str | Path, width: str = "100%", height: int = 750) -> "IFrame":
    """Render a `generate_report()` output directory in a JupyterLite notebook."""
    if not pyodide_available():
        msg = "render_in_jupyterlite() requires running under Pyodide (js module not found)."
        raise RuntimeError(msg)

    from IPython.display import IFrame  # pyright: ignore[reportMissingImports]

    state = json.loads(Path(directory, "state.mvsj").read_text())
    for snapshot in state["snapshots"]:
        patch_download_urls(snapshot["root"], directory)

    report_html = Path(directory, "report.html").read_text()
    report_html = report_html.replace(
        "mvsStories.loadFromURL('state.mvsj', { format: 'mvsj' });",
        f"mvsStories.loadFromData({json.dumps(json.dumps(state))}, {{ format: 'mvsj' }});",
    )
    report_html = blobify_local_refs(report_html, directory)

    viewer_url = make_blob_url(report_html.encode("utf-8"), "text/html")

    return IFrame(viewer_url, width=width, height=height)


def determine_core_indices(mask):
    """Calculate the core indices of a shape"""

    core_indices = np.zeros(mask.shape)
    eroded_mask = mask > 0
    while eroded_mask.sum() > 0:
        core_indices += eroded_mask
        eroded_mask = binary_erosion(eroded_mask)
    return core_indices


def fisher_sigma(mv, fsc):
    return 1 / sqrt(mv / fsc - 3)


def write_fits_to_pdb(structure, solutions, basename="fit"):
    translated_structure = structure.duplicate()
    center = translated_structure.coor.mean(axis=1)
    translated_structure.translate(-center)
    for n, sol in enumerate(solutions, start=1):
        out = translated_structure.duplicate()
        rot = np.asarray([float(x) for x in sol[6:]]).reshape(3, 3)
        trans = sol[3:6]
        out.rotate(rot)
        out.translate(trans)
        out.tofile(basename + f"_{n:d}.pdb")
