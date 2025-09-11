#!/usr/bin/env bash
set -e

# Where to download the requirements.txt
REQ_URL="https://raw.githubusercontent.com/eProsima/agile-hri/refs/heads/${VULCANEXUS_DISTRO}/hri_requirements.txt"
TMP_REQ="/tmp/hri_requirements.txt"

echo "Getting Vulcanexus HRI Python requirements from GitHub..."
wget -qO "$TMP_REQ" "$REQ_URL"

echo "Temporarily allowing pip to install global packages..."
python3 -m pip config set global.break-system-packages true

echo "Installing Vulcanexus HRI Python dependencies from $TMP_REQ..."
python3 -m pip install --break-system-packages --ignore-installed --no-cache-dir -r "$TMP_REQ"
rm "$TMP_REQ"

python3 -m pip config set global.break-system-packages false

echo "All Vulcanexus HRI Python dependencies installed successfully!"
