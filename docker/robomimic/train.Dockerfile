# Image for training/evaluating a robomimic BC-RNN baseline on robosuite's
# NutAssemblySquare task. This is the frozen "broad but imprecise" base
# policy for the Thread B (RLT-faithful precision task) experiment -- a real
# demonstration-trained policy on a tight-tolerance insertion task, not a
# zero-shot VLA. Separate from docker/openpi/* since it has no OpenPI
# dependency at all: this is an offline training + local-rollout image.

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y \
    make \
    g++ \
    clang \
    cmake \
    python3.8-dev \
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

RUN uv venv --python 3.8 $UV_PROJECT_ENVIRONMENT
RUN uv pip install \
    "numpy<2.0.0" \
    "mujoco==2.3.3" \
    robosuite==1.4.1 \
    robomimic==0.3.0 \
    h5py \
    tensorboardX \
    imageio[ffmpeg] \
    egl_probe

# robomimic's EnvRobosuite wrapper unconditionally imports the legacy
# mujoco_py package (only ever used for one exception type in a rollout
# try/except), which is incompatible with the modern `mujoco` bindings
# robosuite 1.4.1 actually uses. Stub it out rather than installing the
# real (fragile, unmaintained) mujoco_py.
COPY docker/robomimic/stubs/mujoco_py /.venv/lib/python3.8/site-packages/mujoco_py

RUN mkdir -p /usr/share/glvnd/egl_vendor.d && echo '{"file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" }}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

CMD ["/bin/bash"]
