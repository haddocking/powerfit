FROM python:3.12-bookworm AS build
# Downgraded to python 3.12 so binary wheel for siphash24 is available

WORKDIR /src

RUN apt update && apt install -y ocl-icd-opencl-dev

RUN pip install build

COPY . .

RUN python -m build --wheel

# Build wheel for pyvkfft in build container to reduce final image size
RUN pip wheel -w dist --no-deps pyvkfft

FROM python:3.12-slim-bookworm

RUN apt update && apt install -y ocl-icd-libopencl1
RUN apt install -y intel-opencl-icd

COPY --from=build /src/dist/*.whl /opt/

RUN PFWHL_FILE=$(find /opt -name "powerfit*.whl" | head -n 1) && \
    FTTWHL_FILE=$(find /opt -name "pyvkfft*.whl" | head -n 1) && \
    pip install "${PFWHL_FILE}[opencl]" "${FTTWHL_FILE}"

ENTRYPOINT [ "powerfit" ]
