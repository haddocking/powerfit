// Script to test wasm wheel
//
// Prepare
// 1. Get wasm wheel in cwd by compiling it locally see ../../CONTRIBUTING.md#build-wasm-wheel-locally or install from GitHub releases page.
// 2. Put input files in cwd
//
// Run with `pnpm i && node runme.mjs` in this directory.
//
// Should have generated r/solutions.out and r/lcc.mrc files.

import { readdirSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const { loadPyodide } = await import("pyodide");
const cwd = process.cwd();
const wheelName = readdirSync(cwd).find((name) => name.startsWith("powerfit_em-") && name.endsWith(".whl"));
if (!wheelName) {
    throw new Error("No powerfit_em-*.whl file found in current directory.");
}

let pyodide = await loadPyodide();
await pyodide.loadPackage("pygments") // Use pygments shipped with pyodide
await pyodide.loadPackage("micropip");
const micropip = pyodide.pyimport("micropip");
await micropip.install(pathToFileURL(join(cwd, wheelName)).href);
const pf = pyodide.pyimport("powerfit_em.powerfit");
const builtins = pyodide.pyimport("builtins");

pyodide.mountNodeFS("/data", cwd);

const target = builtins.open("/data/emd_1046.map.gz", "rb");
const template = builtins.open("/data/9A2G.cif.gz", "rb");

const rel_output_dir = "r";
const py_output_dir = `/data/${rel_output_dir}`;
const js_output_dir = join(cwd, rel_output_dir);

try {
	pyodide.globals.set("target", target);
	pyodide.globals.set("template", template);
	await pyodide.runPythonAsync(`
import powerfit_em.powerfit as pf
pf.powerfit(
    target,
    13.0,
    template,
    angle=20,
    delimiter=",",
    num=0,
    directory="/data/r",
    rust=True,
)
`);
} finally {
	pyodide.globals.delete("target");
	pyodide.globals.delete("template");
	target.close();
	template.close();
}

console.log("Output dir:", js_output_dir);

// Commands takes 10-20 seconds to run
