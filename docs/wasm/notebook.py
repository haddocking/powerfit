import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

"""
Marimo WebAssembly notebook.

To build

```shell
uv pip install marimo
uv run marimo export html-wasm notebook.py -o nbout --mode edit
cp powerfit_em-5.0.0-cp314-cp314-pyemscripten_2026_0_wasm32.whl nbout/
python -m http.server --directory nbout
```

Stuck on micropip.install:
Wheel was built with Emscripten vpyemscripten.2026.0 but Pyodide was built with Emscripten v3.1.58

Must wait for marimo to ship with "pyodide": "314.0.0-alpha.1".
"""

@app.cell
async def _():
    try:
        import micropip
        await micropip.install('http://0.0.0.0:8000/powerfit_em-5.0.0-cp314-cp314-pyemscripten_2026_0_wasm32.whl')
    except ModuleNotFoundError:
        pass # We are not running in the browser, so we assume the package is already installed

    from powerfit_em.powerfit import powerfit

    return (powerfit,)


@app.cell
def _():
    # ribosome-KsgA.map 13 KsgA.pdb -a 20 -p 2 -l -d 
    target_volume_fn = "ribosome-KsgA.map"
    template_structure_fn = "KsgA.pdb"
    output_dir = "r"
    return output_dir, target_volume_fn, template_structure_fn


@app.cell
def _(output_dir, powerfit, target_volume_fn, template_structure_fn):
    with open(target_volume_fn, "rb") as target, open(template_structure_fn, "rb") as template:
        powerfit(
            target_volume=target,
            template_structure=template,
            resolution=13,
            angle=20,
            rust=True,
            directory=output_dir,
        )
    return


@app.cell
def _():
    from pathlib import Path

    return (Path,)


@app.cell
def _(Path, output_dir):
    list(Path(output_dir).glob('*'))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
