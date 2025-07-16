FROM python:3.12 AS build
# Downgraded to python 3.12 so binary wheel for siphash24 is available

WORKDIR /src

RUN pip install build setuptools wheel cython

COPY . .

RUN python -m build --wheel

FROM python:3.12-slim

RUN apt update && apt install -y g++ ocl-icd-opencl-dev ocl-icd-libopencl1

COPY --from=build /src/dist/*.whl /opt/

RUN PFWHL_FILE=$(find /opt -name "powerfit*.whl" | head -n 1) && \
    pip install "${PFWHL_FILE}[opencl]"

ENTRYPOINT [ "powerfit" ]
