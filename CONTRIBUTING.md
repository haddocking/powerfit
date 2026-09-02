# Contributing guidelines

We welcome any kind of contribution to our software, from simple comment or question to a full fledged [pull request](https://help.github.com/articles/about-pull-requests/). Please read and follow our [Code of Conduct](https://github.com/haddocking/powerfit/blob/master/CODE_OF_CONDUCT.md).

A contribution can be one of the following cases:

1. you have a question;
1. you think you may have found a bug (including unexpected behavior);
1. you want to make some kind of change to the code base (e.g. to fix a bug, to add a new feature, to update documentation);
1. you want to make a new release of the code base.

The sections below outline the steps in each case.

## You have a question

1. use the search functionality [here](https://github.com/haddocking/powerfit/issues) to see if someone already filed the same issue;
2. if your issue search did not yield any relevant results, make a new issue;
3. apply the "Question" label; apply other labels when relevant.

## You think you may have found a bug

1. use the search functionality [here](https://github.com/haddocking/powerfit/issues) to see if someone already filed the same issue;
1. if your issue search did not yield any relevant results, make a new issue, making sure to provide enough information to the rest of the community to understand the cause and context of the problem. Depending on the issue, you may want to include:
    - the [SHA hashcode](https://help.github.com/articles/autolinked-references-and-urls/#commit-shas) of the commit that is causing your problem;
    - some identifying information (name and version number) for dependencies you're using;
    - information about the operating system;
1. apply relevant labels to the newly created issue.

## You want to make some kind of change to the code base

1. (**important**) announce your plan to the rest of the community *before you start working*. This announcement should be in the form of a (new) issue;
1. (**important**) wait until some kind of consensus is reached about your idea being a good idea;
1. if needed, fork the repository to your own Github profile and create your own feature branch off of the latest main commit. While working on your feature branch, make sure to stay up to date with the main branch by pulling in changes, possibly from the 'upstream' repository (follow the instructions [here](https://help.github.com/articles/configuring-a-remote-for-a-fork/) and [here](https://help.github.com/articles/syncing-a-fork/));
1. install dependencies (see the [development documentation](#development));
1. make sure the existing tests still work by running ``pytest``;
1. add your own tests (if necessary);
1. update or expand the documentation;
1. [push](http://rogerdudler.github.io/git-guide/) your feature branch to (your fork of) the powerfit repository on GitHub;
1. create the pull request, e.g. following the instructions [here](https://help.github.com/articles/creating-a-pull-request/).

In case you feel like you've made a valuable contribution, but you don't know how to write or run tests for it, or how to generate the documentation: don't let this discourage you from making the pull request; we can help you! Just go ahead and submit the pull request, but keep in mind that you might be asked to append additional commits to your pull request.

## You want to make a new release of the code base

To create a release you need write permission on the repository.

1. Check the author list in [`CITATION.cff`](https://github.com/haddocking/powerfit/blob/master/CITATION.cff)
1. Bump the version in [rust/Cargo.toml](https://github.com/haddocking/powerfit/blob/master/rust/Cargo.toml) under `[package].version`.
1. In [installation.md](docs/installation.md) adjust docker command to use new version.
1. Merge the changes into the main branch.
1. Run regression tests to verify baseline stability across execution profiles:
   ```shell
   pytest -k powerfit_regression -vv --powerfit="--help"
   pytest -k powerfit_regression -vv # Default on 1 CPU
   pytest -k powerfit_regression -vv --powerfit="--nproc 6 --progressbar" # With 6 CPUs
   pytest -k powerfit_regression -vv --powerfit="--gpu" # With auto detected GPU
   ```
   All tests must pass with numerically matching results (within rounding tolerance). If the baseline fixture requires updates, see [Baseline fixture maintenance](#baseline-fixture-maintenance) section under Development.
1. Go to the [GitHub release page](https://github.com/haddocking/powerfit/releases)
1. Press draft a new release button
1. Fill tag, title and description field. For tag use version from `rust/Cargo.toml` (`[package].version`) and prepend with "v" character. The Python package version in `pyproject.toml` is dynamic and follows Cargo. For description use "Rigid body fitting of high-resolution structures in low-resolution cryo-electron microscopy density maps." line plus press "Generate release notes" button.
1. Press the Publish Release button
1. Wait until [Build and upload to PyPI](https://github.com/haddocking/powerfit/actions/workflows/pypi-publish.yml) has completed
1. Verify new release is on [PyPi](https://pypi.org/project/powerfit-em/#history)
1. Verify Zenodo version was added to https://doi.org/10.5281/zenodo.14185749
1. Wait until [Create and publish a Docker image](https://github.com/haddocking/powerfit/actions/workflows/docker-publish.yml) has completed.
1. Verify [new Docker images](https://github.com/haddocking/powerfit/pkgs/container/powerfit)

## Contributing to documentation

Whenever you have changed something in the codebase, this also needs to be reflected in the documentation.
To work on the PowerFit documentation you need to install the documentation version of using:

```shell
pip install -e .[docs]
```

Build the documentation site with the following command:

```shell
cd docs
make
```

The site will be built on `site/`

To serve it locally so you can verify the changes:

```
make serve
```

Open <http://0.0.0.0:8000>


`make` requires `uv` and Rust (via [rustup](https://rustup.rs/)) `>=1.98`, see `make help` for individual targets.

# Development

To develop PowerFit, you need to install the development version of it using.

```shell
pip install -e .[dev]
```

Tests can be run using

```shell
pytest
```

GPU integration tests (marked `requires_cuda` or `requires_opencl`) are automatically skipped when the required hardware or packages are absent. CI runs with `--extra opencl --extra dev` (POCL only) so CUDA tests and OpenCL tests that need a real GPU device are skipped. On a local machine with a GPU, all tests run.

To run OpenCL on **C**PU install use `pip install -e .[pocl]` and make sure no other OpenCL platforms, like 'AMD Accelerated Parallel Processing' or 'NVIDIA CUDA', are installed .

The Docker container, that works for CPU and OpenCL backends, can be build with

```shell
docker build -t ghcr.io/haddocking/powerfit:v5.0.0 .
```

The Docker container, that works for NVIDIA GPUs via CUDA version 13, can be build with

```shell
docker build -t ghcr.io/haddocking/powerfit-cuda13:v5.0.0 -f Dockerfile.cuda13 .
```

For CUDA version 12, use

```shell
docker build -t ghcr.io/haddocking/powerfit-cuda12:v5.0.0 -f Dockerfile.cuda12 .
```

The Docker container, that works for AMD gpus, can be build with

```shell
docker build -t ghcr.io/haddocking/powerfit-rocm:v5.0.0 -f Dockerfile.rocm .
```

The binary wheels can be build for all supported platforms by running the
https://github.com/haddocking/powerfit/actions/workflows/pypi-publish.yml GitHub action and downloading the artifacts.
The workflow is triggered by a push to the main branch, a release or can be manually triggered.

### Rust extension

The CPU version of rotate grid function and `--rust` variant are handled by a Rust extension in the `rust/` directory.
To build the Rust extension, you need to have [Rust](https://rustup.rs/) installed and run:

```shell
uv run maturin develop --release
```

### Linting & formatting

To lint the Python code, run

```shell
ruff check
```
Use `--fix` to automatically fix some of the issues.

To format the Python code, run

```shell
ruff format
```

Cargo commands do not work when uv venv is active, so make sure to deactivate it before running the commands below.

To format the Rust code in rust/ directory, run

```shell
cargo fmt --manifest-path rust/Cargo.toml --all
```

To lint the Rust code, run

```shell
cargo clippy --manifest-path rust/Cargo.toml --all-targets
```

Use `--fix` to automatically apply clippy suggestions when possible:

```shell
cargo clippy --manifest-path rust/Cargo.toml --all-targets --fix --allow-dirty --allow-staged
```

To run the Rust tests, run

```shell
cargo test --manifest-path rust/Cargo.toml --all --locked --verbose
```

Note: unlike the other cargo commands above, `cargo test` needs the uv venv active (`source .venv/bin/activate`) so it picks up the venv's numpy; otherwise some tests fail.

### Baseline fixture maintenance

The regression test in `test_powerfit_regression.py` compares `solutions.out` against a cached baseline at `tests/fixtures/solutions.out`. The baseline should remain stable across different execution profiles (CPU nproc 1/N and GPU backends).

**If the baseline fixture needs updating:**

1. Run the regression test to generate new output:
   ```shell
   pytest -k powerfit_regression -vv
   ```

2. Inspect the test failure output to understand what changed and verify it is expected.
3. Manually copy the generated `solutions.out` from the test's temporary directory into `tests/fixtures/solutions.out`.
4. Run the test again, see [step 5 of "You want to make a new release of the code base" section](#you-want-to-make-a-new-release-of-the-code-base) for example commands.
5. Commit the updated baseline fixture file as part of your change.

### Build wasm wheel locally

Normally `wasm` wheels are built by CI and published to PyPI, but you can also build them locally with `docs/Makefile`:

```shell
cd docs
make wasm-wheel-build
```

Expected output (filename pattern) in `docs/lite/pypi/`:

```text
powerfit_em-<version>-cp314-cp314-pyemscripten_2026_0_wasm32.whl
```
