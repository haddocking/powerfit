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
PYODIDE_XBUILDENV  := 314.0.0a1

.DEFAULT_GOAL := site

.PHONY: help install wasm-wheel mkdocs lite site serve clean

help:
	@echo "make site        - full build: wasm wheel + MkDocs site + JupyterLite"
	@echo "make install     - install '.[docs]' (mkdocs + JupyterLite + wasm build tooling)"
	@echo "make wasm-wheel  - build the wasm wheel, stage it into lite/pypi/"
	@echo "make mkdocs      - build the MkDocs site only (into ../site/)"
	@echo "make lite        - build the JupyterLite site into ../site/lite/ (needs wasm-wheel + mkdocs first)"
	@echo "make serve       - serve the built site at http://localhost:8000"
	@echo "make clean       - remove build outputs"

install:
	cd $(ROOT)
	uv sync --extra docs

wasm-wheel: install
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
