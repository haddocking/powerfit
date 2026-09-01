#=======================================================================================================#
# Makefile for building the PowerFit docs site, including the client-side JupyterLite/wasm demo
#
# Run from docs/
#=======================================================================================================#

.ONESHELL:

ROOT               := ..
WHEELHOUSE         := wheelhouse
LITE_DIR           := docs/lite
LITE_PYPI          := $(LITE_DIR)/pypi
SITE_DIR           := site
PYODIDE_XBUILDENV  := 314.0.0
PY_TAG             := cp314
PYEMSCRIPTEN_TAG   := pyemscripten_2026_0_wasm32
# Parse version from cargo so we know which to download
PKG_VERSION        := $(shell grep -m1 '^version' $(ROOT)/rust/Cargo.toml | sed -E 's/version = "(.*)"/\1/')
# Use `test.pypi` for development and `pypi` for pdo
# WASM_INDEX_URL     := https://test.pypi.org/simple/
WASM_INDEX_URL     := https://pypi.org/simple/

.DEFAULT_GOAL := site

.PHONY: help install wasm-wheel wasm-wheel-build mkdocs lite site serve clean

help:
	@echo "make site             - full build: wasm wheel + MkDocs site + JupyterLite"
	@echo "make install          - install '.[docs]' (mkdocs + JupyterLite + wasm build tooling)"
	@echo "make wasm-wheel       - fetch the wasm wheel from PyPI, falling back to a local build"
	@echo "make wasm-wheel-build - always build the wasm wheel locally, stage it into lite/pypi/"
	@echo "make mkdocs           - build the MkDocs site only (into ../site/)"
	@echo "make lite             - build the JupyterLite site into ../site/lite/ (needs wasm-wheel + mkdocs first)"
	@echo "make serve            - serve the built site at http://localhost:8000"
	@echo "make clean            - remove build outputs"

install:
	cd $(ROOT)
	uv sync --extra docs

wasm-wheel: install
	cd $(ROOT)
	mkdir -p $(LITE_PYPI)
	# Try to find the whell on PyPi
	echo "Looking for powerfit-em==$(PKG_VERSION) ($(PYEMSCRIPTEN_TAG)) on PyPI..."
	if uvx pip download --no-deps --only-binary=:all: \
	    --index-url $(WASM_INDEX_URL) \
	    --python-version 3.14 --implementation cp --abi $(PY_TAG) \
	    --platform $(PYEMSCRIPTEN_TAG) \
	    --dest $(LITE_PYPI) \
	    "powerfit-em==$(PKG_VERSION)"; then \
	  echo "Fetched wasm wheel from PyPI, skipping local build."; \
	# If its not there then build it
	else \
	  echo "Not found on PyPI - building wasm wheel locally instead."; \
	  $(MAKE) -C $(CURDIR) wasm-wheel-build; \
	fi

wasm-wheel-build: install
	cd $(ROOT)
	# add the `wasm32` target
	rustup target add wasm32-unknown-emscripten
	rustup update stable
	rustc --version
	# install the build env
	uv run pyodide xbuildenv install $(PYODIDE_XBUILDENV)
	# make sure destination is clear
	rm -rf $(WHEELHOUSE) && mkdir -p $(WHEELHOUSE)
	# build
	uv run pyodide build . --outdir $(WHEELHOUSE)
	# move to a location jupyterlite needs it
	mkdir -p $(LITE_PYPI)
	cp $(WHEELHOUSE)/*.whl $(LITE_PYPI)/

mkdocs: install
	cd $(ROOT)
	uv run mkdocs build

lite:
	cd $(ROOT)
	uv run jupyter lite build --lite-dir $(LITE_DIR) --contents content --output-dir $(SITE_DIR)/lite

site: wasm-wheel mkdocs lite
	@echo "Done. Serve with: make serve"

serve:
	cd $(ROOT)
	python3 -m http.server --directory $(SITE_DIR) 8000

clean:
	cd $(ROOT)
	rm -rf $(WHEELHOUSE) $(SITE_DIR) $(LITE_PYPI)/*.whl $(LITE_DIR)/.jupyterlite.doit.db
