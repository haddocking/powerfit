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
1. Bump the version in [src/powerfit_em/__init__.py](https://github.com/haddocking/powerfit/blob/master/src/powerfit_em/__init__.py)
1. In [installation.md](docs/installation.md) adjust docker command to use new version.
1. Merge the changes into the main branch.
1. Run regression tests to verify baseline stability across execution profiles:
   ```shell
   pytest -k powerfit_regression -vv --powerfit="--help"
   pytest -k powerfit_regression -vv # Default on 1 CPU
   pytest -k powerfit_regression -vv --powerfit="--nproc 6" # With 6 CPUs
   pytest -k powerfit_regression -vv --powerfit="--gpu --no-progressbar" # With auto detected GPU
   ```
   All tests must pass with numerically matching results (within rounding tolerance). If the baseline fixture requires updates, see [Baseline fixture maintenance](#baseline-fixture-maintenance) section under Development.
1. Go to the [GitHub release page](https://github.com/haddocking/powerfit/releases)
1. Press draft a new release button
1. Fill tag, title and description field. For tag use version from pyproject.toml and prepend with "v" character. For description use "Rigid body fitting of high-resolution structures in low-resolution cryo-electron microscopy density maps." line plus press "Generate release notes" button.
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

Start the live-reloading docs server with:

```shell
mkdocs serve
```

Build the documentation site with:

```shell
mkdocs build
# The site will be built in the 'site/' directory.
# You can preview it with
python3 -m http.server -d site
```

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

The Docker container, that works for NVIDIA GPUs via CUDA, can be build with

```shell
docker build -t ghcr.io/haddocking/powerfit-cuda:v5.0.0 -f Dockerfile.cuda .
```

The Docker container, that works for AMD gpus, can be build with

```shell
docker build -t ghcr.io/haddocking/powerfit-rocm:v5.0.0 -f Dockerfile.rocm .
```

The binary wheels can be build for all supported platforms by running the
https://github.com/haddocking/powerfit/actions/workflows/pypi-publish.yml GitHub action and downloading the artifacts.
The workflow is triggered by a push to the main branch, a release or can be manually triggered.

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

To check Cython code, run

```shell
cython-lint src/powerfit_em/_powerfit.pyx
```

To format the C code, run

```shell
clang-format -i src/powerfit_em/_extensions.c
```

To lint the C code, run

```shell
clang-tidy src/powerfit_em/_extensions.c -- \
    -I"$(python -c 'from sysconfig import get_paths; print(get_paths()["include"])')" \
    -I"$(python -c 'import numpy; print(numpy.get_include())')"
```

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
