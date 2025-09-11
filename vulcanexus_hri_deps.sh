#!/usr/bin/env bash
set -e

# Where to download the requirements.txt
REQ_URL="https://raw.githubusercontent.com/eProsima/agile-hri/refs/heads/main/hri_requirements.txt"
TMP_REQ="/tmp/hri_requirements.txt"

echo "Getting Vulcanexus HRI Python requirements from GitHub..."
wget -qO "$TMP_REQ" "$REQ_URL"

echo "Installing Vulcanexus HRI Python dependencies from $TMP_REQ..."
python3 -m pip install --no-cache-dir -r "$TMP_REQ"
rm "$TMP_REQ"

echo "All Vulcanexus HRI Python dependencies installed successfully!"
