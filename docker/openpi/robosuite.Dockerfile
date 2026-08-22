# Robosuite client image for screening frozen OpenPI checkpoints on robomimic's
# NutAssemblySquare and ToolHang tasks. These are NOT part of the LIBERO
# benchmark: they are native robosuite environments, so this image installs
# robosuite directly instead of the LIBERO submodule used by libero.Dockerfile.
#
# The frozen policy server (pi05_libero, or any other OpenPI checkpoint) is
# served separately by the unmodified upstream openpi_server image; only the
# small dependency-free openpi-client package crosses into this image.

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y \
    make \
    g++ \
    clang \
    python3.10-dev \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libegl1 \
    libglew-dev \
    libglfw3-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

WORKDIR /app

ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv

COPY ./packages/openpi-client /tmp/openpi-client

RUN uv venv --python 3.10 $UV_PROJECT_ENVIRONMENT
RUN uv pip install \
    "numpy<2.0.0" \
    "mujoco==2.3.3" \
    robosuite==1.4.1 \
    imageio[ffmpeg] \
    typing_extensions \
    /tmp/openpi-client

RUN mkdir -p /usr/share/glvnd/egl_vendor.d && echo '{"file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" }}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

CMD ["/bin/bash"]
