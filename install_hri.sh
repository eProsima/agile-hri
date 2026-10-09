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
UNINSTALL="false"
UNINSTALL_DEPS="true"
PYTHON_BIN="python3"

HRI_STATE_DIR="${INSTALLATION_PATH}/.hri_installer"
APT_STATE_FILE="${HRI_STATE_DIR}/apt_installed_by_hri.txt"
PIP_STATE_FILE="${HRI_STATE_DIR}/pip_installed_by_hri.txt"
REQ_PKGS=()
APT_NEW_PACKAGES=()
PIP_NEW_PACKAGES=()

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

print_help()
{
  cat <<EOF
Usage: install_hri [options]

Options:
  --venv PATH     Use the Python and pip from the existing virtual environment at PATH
  -y, --yes       Do not ask for confirmation (assume 'yes')
  -u, --uninstall Uninstall HRI environment installed by this script
  --skip-deps     Skip dependency removal during uninstallation
  -h, --help      Show this help message

This script:
  - Installs required APT dependencies (including ROS2/HRI packages).
  - Downloads and installs the Python requirements for Vulcanexus HRI.
  - Builds a ROS2 workspace (if it exists) and sources the environment.
EOF
}

get_installed_pkgs()
{
    if [[ -f "$APT_STATE_FILE" ]]; then
        mapfile -t APT_NEW_PACKAGES < "$APT_STATE_FILE"
    fi

    if [[ -f "$PIP_STATE_FILE" ]]; then
        mapfile -t PIP_NEW_PACKAGES < "$PIP_STATE_FILE"
    fi
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
        -u|--uninstall)
        UNINSTALL="true"
        shift
        ;;
        --skip-deps)
        UNINSTALL_DEPS="false"
        shift
        ;;
        *)
        echo "Unknown option: $1" >&2
        print_help
        exit 1
        ;;
    esac
done

# -------------------- Interactive confirmation --------------------
if [[ "$ASSUME_YES" != "true" && "$UNINSTALL" != "true" ]]; then
    echo "This script will:"
    echo "  - Install required APT dependencies."
    echo "  - Install Python HRI dependencies (pip)."
    echo "  - Build and install the ROS 2 workspace for HRI."
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

if [[ "$UNINSTALL" == "true" ]]; then
    get_installed_pkgs
fi
if [[ "$ASSUME_YES" != "true" && "$UNINSTALL" == "true" ]]; then
    echo "This script will:"
    echo "  - Remove APT dependencies installed by the installer."
    echo "  - Remove Python HRI dependencies (pip) installed by the installer."
    echo "  - Clean up the ROS 2 workspace for HRI."
    echo "This script assumes HRI was installed from the installer."
    echo "If HRI was installed from sources, do not use this script."
    echo
    if [[ "$UNINSTALL_DEPS" == "true" ]]; then
        echo "Add the argument --skip-deps to skip dependency removal."
        echo
        echo "APT Packages to be removed: ${APT_NEW_PACKAGES[*]:-None}"
        echo "PIP Packages to be removed: ${PIP_NEW_PACKAGES[*]:-None}"
        echo
    else
        echo "APT and Pip dependency removal will be skipped."
        echo
    fi
    read -r -p "Do you want to continue? [y/N]: " REPLY
    case "$REPLY" in
        [yY]|[yY][eE][sS])
        ;;
        *)
        echo "Uninstallation canceled by user."
        exit 0
        ;;
    esac
fi

# --------------------------- Helpers ---------------------------
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

    echo "Updating APT indices..."
    $SUDO apt-get update -y

    echo "Installing required APT packages..."
    $SUDO apt-get install -y "${packages[@]}"
}

clean_pip_requirements()
{
    local req_file="$1"

    # Read the requirements file and extract package names
    while IFS= read -r line; do
        # Remove comments and empty lines
        line="${line%%#*}"
        line="$(echo "$line" | xargs || true)"
        [[ -z "$line" ]] && continue

        # Get only package name (remove version specifiers)
        name="${line%%[<>=! ]*}"
        [[ -n "$name" ]] && REQ_PKGS+=("$name")
    done < "$TMP_REQ"
}

set_python_bin()
{
    if [[ -n "$VENV_PATH" ]]; then
        if [[ ! -d "$VENV_PATH" ]]; then
            echo "Error: Virtual environment $VENV_PATH does not exist." >&2
            exit 1
        fi
        if [[ ! -x "$VENV_PATH/bin/python" ]]; then
            echo "Error: $VENV_PATH/bin/python not found." >&2
            exit 1
        fi
        PYTHON_BIN="$(realpath "$VENV_PATH")/bin/python"
        echo "Using virtual environment Python: $PYTHON_BIN"
    else
        echo "Using system Python: $PYTHON_BIN"
    fi
}

# -------------------- Installation method --------------------
install()
{
    # -------------------- APT packages installation --------------------
    local APT_PACKAGES=(
        # Common tools
        build-essential
        cmake
        git
        python3-pip
        wget
        # Dependencies
        libusb-1.0-0-dev
        portaudio19-dev
        libportaudio2
        libportaudiocpp0
        alsa-utils
        ffmpeg
        # HRI-API
        libopencv-dev
        libmagicenum-dev
        ros-${VULCANEXUS_DISTRO}-vision-opencv
        ros-${VULCANEXUS_DISTRO}-pybind11-vendor
        ros-${VULCANEXUS_DISTRO}-tf2-ros
    )

    if ! set_sudo; then
        exit 1
    fi

    echo "Installing APT packages..."
    for pkg in "${APT_PACKAGES[@]}"; do
        if ! dpkg -s "$pkg" >/dev/null 2>&1; then
            APT_NEW_PACKAGES+=("$pkg")
        fi
    done

    run_apt "${APT_PACKAGES[@]}"

    # Record installed APT packages
    if ((${#APT_NEW_PACKAGES[@]} > 0)); then
        $SUDO mkdir -p "$HRI_STATE_DIR"
        printf '%s\n' "${APT_NEW_PACKAGES[@]}" | $SUDO tee "$APT_STATE_FILE" >/dev/null
        echo "Recorded APT packages installed by this script in $APT_STATE_FILE"
    fi
    echo "Installed APT packages: ${APT_NEW_PACKAGES[*]:-None}"

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

    set_python_bin
    clean_pip_requirements "$TMP_REQ"

    COLCON_INSTALLED_BY_SCRIPT="false"

    if ! "$PYTHON_BIN" -m pip show colcon-common-extensions >/dev/null 2>&1; then
        echo "Adding colcon-common-extensions to install requirements ..."
        echo "colcon-common-extensions" >> "$TMP_REQ"
        COLCON_INSTALLED_BY_SCRIPT="true"
    fi

    echo "Installing Python HRI dependencies from $TMP_REQ ..."
    PIP_NEW_PACKAGES=()
    for pkg in "${REQ_PKGS[@]}"; do
        if ! "$PYTHON_BIN" -m pip show "$pkg" >/dev/null 2>&1; then
            PIP_NEW_PACKAGES+=("$pkg")
        fi
    done

    "$PYTHON_BIN" -m pip install \
        --break-system-packages \
        --ignore-installed \
        -r "$TMP_REQ"

    if ((${#PIP_NEW_PACKAGES[@]} > 0)); then
        $SUDO mkdir -p "$HRI_STATE_DIR"
        printf '%s\n' "${PIP_NEW_PACKAGES[@]}" > "$PIP_STATE_FILE"
        echo "Recorded pip packages installed by this script in $PIP_STATE_FILE"
    fi
    echo "Installed pip packages:"
    cat "$PIP_STATE_FILE"

    rm -f "$TMP_REQ"

    echo "Python HRI dependencies installed successfully."

    # -------------------- Building --------------------
    HRI_ROS_WS=$(mktemp -d)
    mkdir -p "$HRI_ROS_WS/src"

    if [[ -d "$HRI_ROS_WS/src" ]]; then
        cd "$HRI_ROS_WS/src"
        echo "Cloning Vulcanexus HRI packages into $HRI_ROS_WS/src ..."
        git clone -b "$VULCANEXUS_DISTRO" https://github.com/eProsima/agile-hri.git

        echo "Cmake version is: $(cmake --version | head -n 1)"

        echo "===== COLCON ====="
        colcon version-check

        echo "===== ROS 2 / AMENT PACKAGES ====="
        python3 -c "import ament_package; print('ament_package', ament_package.__version__, ament_package.__file__)" 2>/dev/null || true
        python3 -c "import ament_index_python; print('ament_index_python', ament_index_python.__version__, ament_index_python.__file__)" 2>/dev/null || true
        python3 -c "import rosidl_parser; print('rosidl_parser', rosidl_parser.__version__, rosidl_parser.__file__)" 2>/dev/null || true
        python3 -c "import rosidl_adapter; print('rosidl_adapter', rosidl_adapter.__version__, rosidl_adapter.__file__)" 2>/dev/null || true

        echo "Building ROS2 workspace at $HRI_ROS_WS ..."
        cd "$HRI_ROS_WS"
        "$PYTHON_BIN" -m colcon build --packages-up-to vulcanexus_hri_cpp --event-handlers=console_direct+ --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_VERBOSE_MAKEFILE=ON || true
        ls -R "$HRI_ROS_WS"/install/hri_msgs/
        cat "$HRI_ROS_WS"/build/vulcanexus_hri_cpp/compile_commands.json
        echo "Reached build end."
        "$PYTHON_BIN" -m colcon graph
        return 1

        echo "Installing HRI packages..."
        $SUDO mkdir -p "${INSTALLATION_PATH}/share"

        for pkg in "${HRI_PACKAGES[@]}"; do
            SRC_PKG_DIR="$HRI_ROS_WS/install/${pkg}"

            if [[ ! -d "$SRC_PKG_DIR" ]]; then
                echo "  - Package ${pkg} not found in ${SRC_PKG_DIR}, skipping."
                continue
            fi

            # Copy include (as it did: install/hri_msgs/include -> /opt/vulcanexus/##distro##)
            if [[ -d "$SRC_PKG_DIR/include" ]]; then
                echo "  - Copying ${pkg}/include -> ${INSTALLATION_PATH}/include"
                $SUDO cp -r --update=none "$SRC_PKG_DIR/include" "${INSTALLATION_PATH}/" || true
            fi

            # Copy lib (install/hri_msgs/lib -> /opt/vulcanexus/##distro##)
            if [[ -d "$SRC_PKG_DIR/lib" ]]; then
                echo "  - Copying ${pkg}/lib -> ${INSTALLATION_PATH}/lib"
                $SUDO cp -r --update=none "$SRC_PKG_DIR/lib" "${INSTALLATION_PATH}/" || true
            fi

            # Copy share (install/hri_msgs/share/* -> /opt/vulcanexus/##distro##/share)
            if [[ -d "$SRC_PKG_DIR/share" ]]; then
                echo "  - Copying ${pkg}/share/* -> ${INSTALLATION_PATH}/share"
                $SUDO cp -r --update=none "$SRC_PKG_DIR/share/"* "${INSTALLATION_PATH}/share/" 2>/dev/null || true
            fi
        done

        cd /
        rm -rf "$HRI_ROS_WS"
    fi

    if [ "$COLCON_INSTALLED_BY_SCRIPT" = "true" ]; then
        echo "Uninstalling colcon-common-extensions and its deps ..."
        "$PYTHON_BIN" -m pip uninstall -y --break-system-packages colcon-common-extensions colcon-argcomplete colcon-bash colcon-cd colcon-cmake colcon-core colcon-defaults colcon-devtools colcon-library-path colcon-metadata colcon-notification colcon-output colcon-package-information colcon-package-selection colcon-parallel-executor colcon-powershell colcon-python-setup-py colcon-recursive-crawl colcon-ros colcon-test-result colcon-zsh
    fi

    echo "====================================================="
    echo "  Vulcanexus HRI was installed successfully!"
    echo
    echo "  Remember to source Vulcanexus again in this shell "
    echo "  before using HRI functionalities. New shells will"
    echo "  already be sourced."
    echo "====================================================="
}


# -------------------- Uninstallation method --------------------
uninstall()
{
    if ! set_sudo; then
        return 1
    fi

    echo "Uninstalling HRI environment for distro ${VULCANEXUS_DISTRO} ..."

    set_python_bin

    # Dependencies removal
    if [[ "$UNINSTALL_DEPS" == "true" ]]; then
        # Pip packages removal first to avoid uninstalling pip with apt
        echo "Removing pip dependencies packages"
        "$PYTHON_BIN" -m pip uninstall -y --break-system-packages "${PIP_NEW_PACKAGES[@]}" || true
        rm -f "$PIP_STATE_FILE"

        # APT packages removal
        echo "Removing APT dependencies packages"
        if ((${#APT_NEW_PACKAGES[@]} > 0)); then
            $SUDO apt-get remove -y "${APT_NEW_PACKAGES[@]}" || true
        else
            echo "No APT packages to remove."
        fi
        rm -f "$APT_STATE_FILE"
    else
        echo "Skipping dependency removal as per user request."
    fi

    # Extra files that do not follow basic ROS 2 package structure (libs, conf files, etc.)
    EXTRA_FILES_REMOVAL=(
        "lib/libfacedetection.so"  # Installed by hri_face_detect
    )

    # Installer state cleanup
    for pkg in "${HRI_PACKAGES[@]}"; do
        echo "Removing HRI package: $pkg"
        # Include dirs
        $SUDO rm -rf "${INSTALLATION_PATH}/include/${pkg}" || true

        # Lib dirs
        $SUDO rm -rf "${INSTALLATION_PATH}/lib/${pkg}" || true
        $SUDO rm -rf "${INSTALLATION_PATH}/lib/lib${pkg}"* || true
        $SUDO rm -rf "${INSTALLATION_PATH}/lib/python3.12/site-packages/${pkg}"* || true

        # Share dirs
        $SUDO rm -rf "${INSTALLATION_PATH}/share/${pkg}" || true

        $SUDO rm -rf "${INSTALLATION_PATH}/share/ament_index/resource_index/packages/${pkg}" || true
        $SUDO rm -rf "${INSTALLATION_PATH}/share/ament_index/resource_index/parent_prefix_path/${pkg}" || true
        $SUDO rm -rf "${INSTALLATION_PATH}/share/ament_index/resource_index/rosidl_interfaces/${pkg}" || true
        $SUDO rm -rf "${INSTALLATION_PATH}/share/ament_index/resource_index/package_run_dependencies/${pkg}" || true

        $SUDO rm -rf "${INSTALLATION_PATH}/share/colcon-core/packages/${pkg}" || true
    done

    for file in "${EXTRA_FILES_REMOVAL[@]}"; do
        rm -f "${INSTALLATION_PATH}/${file}" || true
    done

    echo "HRI uninstall completed."
}

# -------------------- Main flow --------------------
if [[ "$UNINSTALL" == "true" ]]; then
    uninstall
    exit 0
fi

install
