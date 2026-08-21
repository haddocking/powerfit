#=======================================================================================================#
# Makefile for building the PowerFit docs site, including the client-side JupyterLite/wasm demo
#
# Run from docs/
#=======================================================================================================#

.ONESHELL:

ROOT               := ..
WHEELHOUSE         := wheelhouse
SITE_DIR           := site
PYODIDE_XBUILDENV  := 314.0.0

.DEFAULT_GOAL := site

.PHONY: help install wasm-wheel mkdocs site serve clean

help:
	@echo "make site        - full build: wasm wheel + MkDocs site (JupyterLite demo embedded via mkdocs-jupyterlite)"
	@echo "make install     - install '.[docs]' (mkdocs + JupyterLite + wasm build tooling)"
	@echo "make wasm-wheel  - build the wasm wheel into wheelhouse/"
	@echo "make mkdocs      - build the MkDocs site (into ../site/); works without wasm-wheel"
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

mkdocs: install
	cd $(ROOT)
	uv run mkdocs build

site: wasm-wheel mkdocs
	@echo "Done. Serve with: make serve"

serve:
	cd $(ROOT)
	python3 -m http.server --directory $(SITE_DIR) 8000

clean:
	cd $(ROOT)
	rm -rf $(WHEELHOUSE) $(SITE_DIR)
