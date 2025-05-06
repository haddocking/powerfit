FROM python:3.12 AS build
# Downgraded to python 3.12 so binary wheel for siphash24 is available

WORKDIR /src

RUN apt update && \
apt install -y libclfft-dev git && \
pip install build setuptools wheel cython

COPY . .

RUN python -m build --wheel

RUN pip wheel -w dist --no-deps --no-use-pep517 gpyfft@git+https://github.com/geggo/gpyfft@v0.8.0

FROM python:3.12-slim

RUN apt update && apt install -y libclfft2

COPY --from=build /src/dist/*.whl /opt/

RUN PFWHL_FILE=$(find /opt -name "powerfit*.whl" | head -n 1) && \
    GFWHL_FILE=$(find /opt -name "gpyfft*.whl" | head -n 1) && \
    pip install "${PFWHL_FILE}[opencl]" "${GFWHL_FILE}"

ENTRYPOINT [ "powerfit" ]
