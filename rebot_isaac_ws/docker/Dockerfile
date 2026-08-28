# reBot B601-DM + Isaac for Manipulation 4.5 development container.
#
# Base: TensorRT 25.08 on Ubuntu 24.04 (Noble). Chosen because Isaac ROS 4.5 is
# Noble/Jazzy-only and is tested against CUDA 13.0 / TensorRT 10.13.x, both of
# which this image already carries:
#   VERSION_ID=24.04, nvcc 13.0, libnvinfer10 10.13.2.6-1+cuda13.0,
#   libcudnn9-cuda-13 9.12.0.46-1
#
# The host stays Ubuntu 22.04 / Humble and gets no Noble apt repositories.
FROM nvcr.io/nvidia/tensorrt:25.08-py3@sha256:81c48cacaecbc586e0ad1e9c15f7cf7769a1a146b6a12f3bbaf83f9c5c16e623

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# Apt repositories. All four are required; the dependency closure of
# ros-jazzy-isaac-ros-manipulation-bringup spans all of them:
#   core ROS (ros2launch, rviz2, tf2-ros, realsense2-camera, moveit) -> (1)
#   cumotion / nvblox / foundationpose                               -> (2)
#   cuda-toolkit-13-0                                                -> (3)
#   libnvvpi4, vpi4-dev, nvsci                                       -> (4)
# Verified with `apt-get install -s`: 1904 packages, zero unmet dependencies.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg ca-certificates lsb-release software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN install -d -m 0755 /etc/apt/keyrings

# (1) ROS 2 Jazzy.
#
# Two deliberate deviations from the upstream instructions, both verified:
#
#   a) The key comes from ros/rosdistro, NOT from packages.ros.org/ros.key.
#      The copy served by packages.ros.org is EXPIRED — apt reports
#      "EXPKEYSIG F42ED6FBAB17C654 Open Robotics" and refuses the repo as
#      unsigned. rosdistro serves the same key id with expiry renewed to 2030.
#
#   b) The sources line uses http://, not https://. packages.ros.org
#      DNS-resolves to ftp.osuosl.org, which presents a certificate for
#      *.osuosl.org, so TLS verification fails with
#      "no alternative certificate subject name matches target host name".
#      Reproduced from the host and from inside a container, over both IPv4
#      and IPv6, against the system CA bundle -- it is an upstream SAN
#      mismatch, not corporate TLS interception. Plain HTTP returns 200 and
#      apt still verifies the Release file against the GPG key above, so
#      package integrity is preserved.
#
# Note: --dearmor writes (never `tee -a`); appending corrupts the keyring if a
# layer is rebuilt.
RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      | gpg --dearmor > /etc/apt/keyrings/ros.gpg \
 && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/ros.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
      > /etc/apt/sources.list.d/ros2.list

# (2) Isaac ROS 4.5.
RUN curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key \
      | gpg --dearmor > /etc/apt/keyrings/nvidia-isaac-ros.gpg \
 && echo "deb [signed-by=/etc/apt/keyrings/nvidia-isaac-ros.gpg] https://isaac.download.nvidia.com/isaac-ros/release-4.5 noble main" \
      > /etc/apt/sources.list.d/nvidia-isaac-ros.list

# (3) CUDA 13 / TensorRT. ros-jazzy-isaac-ros-common hard-depends on
#     cuda-toolkit-13-0, which the base image does not provide.
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
      | gpg --dearmor > /etc/apt/keyrings/cuda.gpg \
 && echo "deb [signed-by=/etc/apt/keyrings/cuda.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
      > /etc/apt/sources.list.d/cuda.list

# (4) L4T "Jetson host components". Serves amd64 despite the /jetson/ path and
#     is the ONLY source of libnvvpi4, vpi4-dev and nvsci, which
#     isaac-ros-common, -nitros, -gxf, -nitros-image-type and -vpi-utils
#     hard-depend on. NVIDIA's docs list it without explanation; omitting it
#     makes the whole manipulation tree unresolvable.
RUN curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
      | gpg --dearmor > /etc/apt/keyrings/nvidia-jetson.gpg \
 && echo "deb [signed-by=/etc/apt/keyrings/nvidia-jetson.gpg] https://repo.download.nvidia.com/jetson/x86_64/noble r38.4 main" \
      > /etc/apt/sources.list.d/nvidia-jetson.list

# Fail the build loudly if any repo is unsigned/unreachable, rather than
# silently resolving fewer packages later.
RUN apt-get update 2>&1 | tee /tmp/aptupdate.log \
 && ! grep -qiE "^E:|is not signed|EXPKEYSIG|NO_PUBKEY" /tmp/aptupdate.log

# ---------------------------------------------------------------------------
# ROS 2 Jazzy + MoveIt + build tooling.
# ---------------------------------------------------------------------------
RUN apt-get install -y --no-install-recommends \
        ros-jazzy-ros-base \
        ros-jazzy-moveit \
        ros-jazzy-rmw-cyclonedds-cpp \
        ros-jazzy-xacro \
        ros-jazzy-ros2-control \
        ros-jazzy-ros2-controllers \
        ros-jazzy-joint-state-publisher \
        ros-jazzy-joint-state-broadcaster \
        ros-jazzy-joint-trajectory-controller \
        ros-jazzy-rviz2 \
        ros-jazzy-rqt-image-view \
        ros-jazzy-tf-transformations \
        ros-jazzy-urdfdom-py \
        ros-jazzy-ros-testing \
        python3-colcon-common-extensions python3-rosdep python3-vcstool \
        python3-pip python3-git python3-pytest python3-pytest-cov \
        liburdfdom-tools \
        git wget nano less iputils-ping net-tools usbutils udev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# isaac-ros-cli, and the environment mode it gates on.
#
# This step is MANDATORY and must precede the manipulation install, even though
# NVIDIA's container instructions do not mention it.
#
# The manipulation dependency closure pulls ~16 `python3-*-pip-shim` packages
# (numpy, torch, torchvision, transformers, sam2, ...). Each is an empty .deb
# whose maintainer scripts shell out to isaac-ros-cli:
#     preinst  -> /usr/lib/isaac-ros-cli/check-pip-shim-readiness
#     postinst -> /usr/lib/isaac-ros-cli/install-pip-shim "<pkg>==<ver>"
# and readiness reads ISAAC_ROS_ENVIRONMENT from
# /etc/isaac-ros-cli/environment.conf, which ships as `uninitialized`. In that
# state preinst exits 1, dpkg aborts every shim, and the whole transaction fails:
#     Error: Isaac ROS environment configuration is not set!
#     dpkg: error processing archive .../python3-numpy-pip-shim...deb (--unpack)
#
# `apt-get install -s` does NOT catch this -- simulation never runs maintainer
# scripts, so the dependency closure resolves cleanly and the install still fails.
#
# Mode choice. `isaac-ros init` accepts docker|venv|baremetal:
#   docker     -- preinst rejects it outright (it means "delegate to a container",
#                 and we ARE the container).
#   venv       -- installs into /var/lib/isaac-ros-cli/isaac-ros, which is NOT on
#                 the interpreter path that ros2 launch uses, so ROS nodes would
#                 not see torch/numpy.
#   baremetal  -- system pip3 with --break-system-packages --ignore-installed.
# `baremetal` is correct here: inside a disposable container the system
# interpreter IS the target environment, and ROS nodes import from it directly.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends isaac-ros-cli \
 && isaac-ros init baremetal --yes \
 && /usr/lib/isaac-ros-cli/check-pip-shim-readiness \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Isaac ROS 4.5 manipulation stack (binary). The source overlay for the
# packages we must patch is built later from the mounted workspace (Stage 1).
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-isaac-ros-manipulation-bringup \
        ros-jazzy-isaac-ros-manipulation-asset-bringup \
    && rm -rf /var/lib/apt/lists/*

# Pinocchio (Seeed SDK kinematics) and the pyvers version NVIDIA requires.
# Installed AFTER the shims so their pinned constraints win on any overlap.
#
# numpy is pinned EXPLICITLY and listed FIRST. Without the pin, pip resolves
# `pin` against numpy 2.5.1, installs it, then downgrades to 1.26.4 to satisfy
# the shim constraint -- and leaves 2.5.1's package tree in /usr/local shadowing
# Debian's 1.26.4, with two numpy dist-info dirs as the only trace. Every
# apt-built extension (cv2, and through it isaac_ros_manipulation's
# manipulator_types) is compiled against the NumPy 1.x C ABI, so they all fail
# with "numpy.core.multiarray failed to import". The failure surfaces far from
# its cause, so pin rather than repair after the fact.
#
# --force-reinstall on numpy alone rewrites the shadowing copy in place; the
# pip-shim layer has already put 1.26.4 in Debian's tree.
RUN pip install --break-system-packages --no-cache-dir --force-reinstall \
        "numpy==1.26.4" \
 && pip install --break-system-packages --no-cache-dir \
        pyvers==0.2.2 pin \
 && pip install --break-system-packages --no-cache-dir --force-reinstall \
        "numpy==1.26.4" \
 && python3 -c "import numpy, cv2; \
assert numpy.__version__.startswith('1.26'), numpy.__version__; \
print('numpy', numpy.__version__, 'cv2', cv2.__version__)"

# rosdep overlay carrying NVIDIA's extra rosdep keys, needed to resolve
# dependencies of the manipulation packages when building from source.
RUN rosdep init 2>/dev/null || true \
 && curl -fsSL -o /etc/ros/rosdep/sources.list.d/nvidia-isaac.yaml \
      https://raw.githubusercontent.com/NVIDIA-ISAAC-ROS/isaac-ros-cli/release-4.5/docker/rosdep/extra_rosdeps.yaml \
 && echo "yaml file:///etc/ros/rosdep/sources.list.d/nvidia-isaac.yaml" \
      > /etc/ros/rosdep/sources.list.d/00-nvidia-isaac.list \
 && rosdep update --rosdistro jazzy || true

# ---------------------------------------------------------------------------
# Environment.
#
# ROS_DOMAIN_ID and RMW must match the Isaac Sim side on the host or discovery
# silently fails. Isaac Sim owns /clock; container nodes set use_sim_time per
# launch profile (true for sim, false for real), never globally here.
# ---------------------------------------------------------------------------
ENV ISAAC_ROS_WS=/workspaces/rebot_isaac_ws \
    ROS_DISTRO=jazzy \
    ROS_DOMAIN_ID=42 \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    ISAAC_ROS_ACCEPT_EULA=1 \
    MANIPULATOR_INSTALL_ASSETS=1 \
    FOUNDATIONSTEREO_MODEL_RES=low_res \
    PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/src/tensorrt/bin:${PATH}

WORKDIR /workspaces/rebot_isaac_ws
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
