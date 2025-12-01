#!/usr/bin/env bash

# Copyright 2025 Proyectos y Sistemas de Mantenimiento SL (eProsima).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Installation script for Vulcanexus HRI
# Usage:
#   install_hri [--venv /path/to/venv] [-y|--yes]

set -euo pipefail

VENV_PATH=""
ASSUME_YES="false"
INSTALLATION_PATH="/opt/vulcanexus/${VULCANEXUS_DISTRO:-kilted}"
SUDO=""

print_help()
{
  cat <<EOF
Usage: install_hri [options]

Options:
  --venv PATH     Use the Python and pip from the virtual environment at PATH
  -y, --yes       Do not ask for confirmation (assume 'yes')
  -h, --help      Show this help message

This script:
  - Installs required APT dependencies (including ROS2/HRI packages).
  - Downloads and installs the Python requirements for Vulcanexus HRI.
  - Builds a ROS2 workspace (if it exists) and sources the environment.
EOF
}

# -------------------- Argument parsing --------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)
        if [[ $# -lt 2 ]]; then
            echo "Error: --venv requires a path to a virtual environment." >&2
            exit 1
        fi
        VENV_PATH="$2"
        shift 2
        ;;
        -y|--yes)
        ASSUME_YES="true"
        shift
        ;;
        -h|--help)
        print_help
        exit 0
        ;;
        *)
        echo "Unknown option: $1" >&2
        print_help
        exit 1
        ;;
    esac
done

# -------------------- Interactive confirmation --------------------
if [[ "$ASSUME_YES" != "true" ]]; then
    echo "This script will:"
    echo "  - Install required APT dependencies."
    echo "  - Install Python HRI dependencies (pip)."
    echo "  - Build and source the ROS 2 workspace for HRI."
    echo "This script assumes there is a valid Vulcanexus installation at $INSTALLATION_PATH."
    echo "If Vulcanexus was installed from sources, install HRI from sources as well."
    echo
    read -r -p "Do you want to continue? [y/N]: " REPLY
    case "$REPLY" in
        [yY]|[yY][eE][sS])
        ;;
        *)
        echo "Installation canceled by user."
        exit 0
        ;;
    esac
fi

# -------------------- Helper for APT (sudo/no sudo) --------------------
set_sudo()
{
    if [[ "$(id -u)" -ne 0 ]]; then
        if command -v sudo >/dev/null 2>&1; then
            SUDO="sudo"
        else
            echo "You are not root and sudo is not available. Installer would not be able to install APT packages." >&2
            return 1
        fi
    fi
    return 0
}

run_apt()
{
    local packages=("$@")

    if ! command -v apt-get >/dev/null 2>&1; then
        echo "apt-get is not available. Skipping APT package installation."
        return 0
    fi

    local SUDO=""
    if [[ "$(id -u)" -ne 0 ]]; then
        if command -v sudo >/dev/null 2>&1; then
            SUDO="sudo"
        else
            echo "You are not root and sudo is not available. Cannot install APT packages." >&2
            return 1
        fi
    fi

    echo "Updating APT indices..."
    $SUDO apt-get update -y

    echo "Installing required APT packages..."
    $SUDO apt-get install -y "${packages[@]}"
}

# -------------------- Python selection (global o venv) --------------------
PYTHON_BIN="python3"

if [[ -n "$VENV_PATH" ]]; then
    if [[ ! -d "$VENV_PATH" ]]; then
        echo "Error: Virtual environment $VENV_PATH does not exist." >&2
        exit 1
    fi
    if [[ ! -x "$VENV_PATH/bin/python" ]]; then
        echo "Error: $VENV_PATH/bin/python not found." >&2
        exit 1
    fi
    PYTHON_BIN="$VENV_PATH/bin/python"
    echo "Using virtual environment Python: $PYTHON_BIN"
else
    echo "Using system Python: $PYTHON_BIN"
fi

# -------------------- APT packages installation --------------------
APT_COMMON_PACKAGES=(
    build-essential
    cmake
    git
    python3-pip
    wget
)

DEPS_APT_PACKAGES=(
    "libusb-1.0-0-dev"
    "portaudio19-dev"
    "libportaudio2"
    "libportaudiocpp0"
    "alsa-utils"
    "ffmpeg"
    # HRI-API
    "libopencv-dev"
    "libmagicenum-dev"
    "ros-${VULCANEXUS_DISTRO}-vision-opencv"
    "ros-${VULCANEXUS_DISTRO}-pybind11-vendor"
    "ros-${VULCANEXUS_DISTRO}-tf2-ros"
)

if ! set_sudo; then
    exit 1
fi

echo "Installing APT packages..."
run_apt "${APT_COMMON_PACKAGES[@]}" "${DEPS_APT_PACKAGES[@]}"

# -------------------- Python HRI dependencies installation --------------------
if [[ -z "${VULCANEXUS_DISTRO:-}" ]]; then
    echo "Error: environment variable VULCANEXUS_DISTRO is not set." >&2
    echo "Source Vulcanexus installation or manually: export VULCANEXUS_DISTRO=kilted" >&2
    exit 1
fi

REQ_URL="https://raw.githubusercontent.com/eProsima/agile-hri/refs/heads/${VULCANEXUS_DISTRO}/hri_requirements.txt"
TMP_REQ="$(mktemp)"

echo "Downloading requirements file:"
echo "  $REQ_URL"
if ! wget -qO "$TMP_REQ" "$REQ_URL"; then
    echo "Error downloading $REQ_URL" >&2
    rm -f "$TMP_REQ"
    exit 1
fi

echo "Installing Python HRI dependencies from $TMP_REQ ..."
"$PYTHON_BIN" -m pip install \
    --break-system-packages \
    --ignore-installed \
    -r "$TMP_REQ"

COLCON_INSTALLED_BY_SCRIPT="false"

if ! "$PYTHON_BIN" -m pip show colcon-common-extensions >/dev/null 2>&1; then
    echo "Installing colcon-common-extensions ..."
    "$PYTHON_BIN" -m pip install --break-system-packages --no-cache-dir colcon-common-extensions
    COLCON_INSTALLED_BY_SCRIPT="true"
fi

rm -f "$TMP_REQ"

echo "Python HRI dependencies installed successfully."

# -------------------- Building --------------------
HRI_ROS_WS=$(mktemp -d)
mkdir -p "$HRI_ROS_WS/src"

if [[ -d "$HRI_ROS_WS/src" ]]; then
    cd "$HRI_ROS_WS/src"
    echo "Cloning Vulcanexus HRI packages into $HRI_ROS_WS/src ..."
    git clone -b "$VULCANEXUS_DISTRO" https://github.com/eProsima/agile-hri.git

    echo "Building ROS2 workspace at $HRI_ROS_WS ..."
    cd "$HRI_ROS_WS"
    colcon build

    VULCANEXUS_PREFIX="/opt/vulcanexus/${VULCANEXUS_DISTRO}"

    HRI_PACKAGES=(
        hri_msgs
        hri_id_manager
        hri_face_detect
        hri_pose_detect
        hri_emotion_detect
        hri_detection_display
        hri_stt
        hri_tts
    )

    echo "Installing HRI packages..."
    $SUDO mkdir -p "${VULCANEXUS_PREFIX}/share"

    for pkg in "${HRI_PACKAGES[@]}"; do
        SRC_PKG_DIR="$HRI_ROS_WS/install/${pkg}"

        if [[ ! -d "$SRC_PKG_DIR" ]]; then
            echo "  - Package ${pkg} not found in ${SRC_PKG_DIR}, skipping."
            continue
        fi

        # Copy include (as it did: install/hri_msgs/include -> /opt/vulcanexus/##distro##)
        if [[ -d "$SRC_PKG_DIR/include" ]]; then
            echo "  - Copying ${pkg}/include -> ${VULCANEXUS_PREFIX}/include"
            $SUDO cp -r --update=none "$SRC_PKG_DIR/include" "${VULCANEXUS_PREFIX}/" || true
        fi

        # Copy lib (install/hri_msgs/lib -> /opt/vulcanexus/##distro##)
        if [[ -d "$SRC_PKG_DIR/lib" ]]; then
            echo "  - Copying ${pkg}/lib -> ${VULCANEXUS_PREFIX}/lib"
            $SUDO cp -r --update=none "$SRC_PKG_DIR/lib" "${VULCANEXUS_PREFIX}/" || true
        fi

        # Copy share (install/hri_msgs/share/* -> /opt/vulcanexus/##distro##/share)
        if [[ -d "$SRC_PKG_DIR/share" ]]; then
            echo "  - Copying ${pkg}/share/* -> ${VULCANEXUS_PREFIX}/share"
            $SUDO cp -r --update=none "$SRC_PKG_DIR/share/"* "${VULCANEXUS_PREFIX}/share/" 2>/dev/null || true
        fi
    done

    cd /
    rm -rf "$HRI_ROS_WS"
fi

if [ "$COLCON_INSTALLED_BY_SCRIPT" = "true" ]; then
    echo "Uninstalling colcon-common-extensions ..."
    "$PYTHON_BIN" -m pip uninstall -y colcon-common-extensions --break-system-packages
fi

echo "====================================================="
echo "  Vulcanexus HRI was installed successfully!"
echo
echo "  Remember to source Vulcanexus again in this shell "
echo "  before using HRI functionalities. New shells will"
echo "  already be sourced."
echo "====================================================="
