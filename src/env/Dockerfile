# Single Franka image for this Panda: ROS 2 Jazzy, the FER libfranka fork,
# and panda_cartesian_controller. Do not layer a second image on top.
FROM ros:jazzy-ros-base-noble
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
ARG FRANKA_ROS2_REF=c0e1f8206211
ARG LIBFRANKA_REF=55e4ab7654
ENV FRANKA_ROS2_REF=${FRANKA_ROS2_REF} LIBFRANKA_REF=${LIBFRANKA_REF}
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates curl pkg-config \
    python3-venv python3-dev python3-pip python3-colcon-common-extensions python3-rosdep \
    libeigen3-dev libpoco-dev libfmt-dev libspdlog-dev \
    libgl1 libglib2.0-0 libportaudio2 libsndfile1 ffmpeg libusb-1.0-0 \
    ros-jazzy-rmw-cyclonedds-cpp ros-jazzy-control-msgs ros-jazzy-xacro \
    ros-jazzy-joint-state-publisher ros-jazzy-joint-state-broadcaster \
    ros-jazzy-controller-manager ros-jazzy-robot-state-publisher ros-jazzy-rviz2 \
    && rm -rf /var/lib/apt/lists/*
# Explicit dependencies of the pinned official single-arm packages. Keep this
# layer separate from source builds; no mutable rosdep index is required.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-moveit-core ros-jazzy-moveit-msgs ros-jazzy-pinocchio \
    ros-jazzy-realtime-tools ros-jazzy-backward-ros \
    ros-jazzy-generate-parameter-library ros-jazzy-joint-state-publisher-gui \
    libtinyxml2-dev && rm -rf /var/lib/apt/lists/*
COPY src/env/panda_cartesian_controller /tmp/panda_cartesian_controller
COPY src/env/build_franka.sh /usr/local/bin/build_franka
RUN bash /usr/local/bin/build_franka \
    && bash -lc 'source /opt/ros/jazzy/setup.bash && source /opt/franka_ws/install/setup.bash && python3 -c "from franka_msgs.msg import FrankaRobotState; assert len(FrankaRobotState().o_t_ee)==16"'
# The locked Teleop XR Git fork builds its WebXR assets during installation.
COPY --from=node:22-bookworm-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:22-bookworm-slim /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx
# rclpy is an apt-built CPython 3.12 extension: use the system interpreter.
COPY --from=ghcr.io/astral-sh/uv:0.7.20 /uv /uvx /usr/local/bin/
RUN python3 -m venv --system-site-packages /opt/data_collect_venv
COPY pyproject.toml uv.lock README.md /opt/data_collect_project/
COPY src /opt/data_collect_project/src
COPY config /opt/data_collect_project/config
COPY vr_collect.py inference.py /opt/data_collect_project/
RUN --mount=type=cache,target=/root/.cache/uv \
    source /opt/ros/jazzy/setup.bash \
    && source /opt/franka_ws/install/setup.bash \
    && cd /opt/data_collect_project \
    && UV_HTTP_TIMEOUT=300 UV_PROJECT_ENVIRONMENT=/opt/data_collect_venv uv sync \
       --locked --python /usr/bin/python3 --no-managed-python \
    && /opt/data_collect_venv/bin/python -c 'import rclpy; from franka_msgs.msg import FrankaRobotState'
COPY src/env/entrypoint.sh /usr/local/bin/data_collect_entrypoint
RUN chmod +x /usr/local/bin/data_collect_entrypoint
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/data_collect_entrypoint"]
CMD ["bash"]
