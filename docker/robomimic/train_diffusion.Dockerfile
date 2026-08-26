# Image for training/evaluating a Diffusion Policy baseline on robosuite's
# NutAssemblySquare task -- a second, parallel frozen base policy alongside
# the BC-RNN baseline in train.Dockerfile, for direct comparison. Separate
# image because Diffusion Policy support only exists on robomimic's GitHub
# master, not the robomimic==0.3.0 PyPI release train.Dockerfile pins; that
# image and its results/bc_rnn_baseline/ lineage stay untouched.

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80retries && \
    echo 'Acquire::http::Timeout "20";' >> /etc/apt/apt.conf.d/80retries && \
    echo 'Acquire::https::Timeout "20";' >> /etc/apt/apt.conf.d/80retries
RUN apt-get update
RUN apt-get install -y \
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
    libxext6 \
    git

WORKDIR /app

ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv
# torch's CUDA wheels (pulled in transitively) are large and this host's
# network is slow; uv's default 30s per-request timeout is too aggressive
# here and kills otherwise-healthy downloads mid-transfer.
ENV UV_HTTP_TIMEOUT=600

RUN uv venv --python 3.8 $UV_PROJECT_ENVIRONMENT
# robomimic pinned to a specific GitHub master commit (Aug 2026, git log:
# "Lazy CLIP download (#303)") since diffusion_policy support was never
# published to PyPI (latest PyPI release is 0.3.0, which predates it).
# diffusers/transformers/huggingface_hub are robomimic master's own
# diffusion_policy.py requirements. Vendored as a local, pre-cloned source
# tree (docker/robomimic/robomimic_src/, gitignored) rather than a
# `git+https://...` pip dependency: a one-shot git fetch of the full repo
# from inside the Docker build proved unreliable on this host's network
# (a 781-second attempt died to a mid-transfer TLS error) -- installing
# from a local path that was already cloned (with its own retry loop) on
# the host avoids repeating that fragile fetch on every build attempt.
COPY docker/robomimic/robomimic_src /opt/robomimic_src
RUN uv pip install \
    "numpy<2.0.0" \
    "mujoco==2.3.3" \
    robosuite==1.4.1 \
    /opt/robomimic_src \
    "diffusers==0.11.1" \
    "transformers==4.41.2" \
    "huggingface_hub==0.23.4" \
    h5py \
    tensorboardX \
    imageio[ffmpeg] \
    egl_probe

# Same mujoco_py compatibility stub as train.Dockerfile -- robomimic's
# EnvRobosuite wrapper imports it only for one unused exception type,
# unrelated to which algo module is in use.
COPY docker/robomimic/stubs/mujoco_py /.venv/lib/python3.8/site-packages/mujoco_py

RUN mkdir -p /usr/share/glvnd/egl_vendor.d && echo '{"file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" }}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

CMD ["/bin/bash"]
