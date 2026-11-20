FROM osrf/ros:jazzy-desktop

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV ROS_DISTRO=jazzy
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV UV_PROJECT_ENVIRONMENT=/opt/omnigda-venv
ENV UV_LINK_MODE=copy
ENV PATH="/opt/omnigda-venv/bin:/root/.local/bin:${PATH}"

# System, Vulkan/EGL rendering, and ROS 2 extras
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    cmake \
    pkg-config \
    ffmpeg \
    libvulkan1 \
    vulkan-tools \
    mesa-vulkan-drivers \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    python3-pip \
    python3-venv \
    python3-colcon-common-extensions \
    ros-jazzy-cv-bridge \
    ros-jazzy-vision-msgs \
    ros-jazzy-tf2-geometry-msgs \
    ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /uvx /usr/local/bin/

# Create Python 3.12 venv inheriting ROS 2 system-site-packages (rclpy, cv_bridge, etc.)
RUN uv venv --system-site-packages --python /usr/bin/python3.12 /opt/omnigda-venv

WORKDIR /workspace
COPY pyproject.toml README.md /workspace/
RUN mkdir -p /workspace/src/omnigda && touch /workspace/src/omnigda/__init__.py

# Install Python stack with uv (locked & reproducible)
RUN uv sync --extra dev --no-install-project

# Ensure ROS 2 and TensorRT libraries are always discoverable
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /etc/bash.bashrc
RUN printf '#!/usr/bin/env bash\nset -e\nsource /opt/ros/jazzy/setup.bash\nif [ -f "${PWD}/ros2_ws/install/setup.bash" ]; then\n  source "${PWD}/ros2_ws/install/setup.bash"\nfi\nexport PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"\nexec "$@"\n' > /usr/local/bin/omnigda-entrypoint.sh \
    && chmod +x /usr/local/bin/omnigda-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/omnigda-entrypoint.sh"]
CMD ["bash"]
