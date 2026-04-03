#!/bin/bash
# Install Dioramatic module to Move
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

if [ ! -d "dist/dioramatic" ]; then
    echo "Error: dist/dioramatic not found. Run ./scripts/build.sh first."
    exit 1
fi

echo "=== Installing Dioramatic Module ==="

# Deploy to Move - audio_fx subdirectory
echo "Copying module to Move..."
ssh ableton@move.local "mkdir -p /data/UserData/schwung/modules/audio_fx/dioramatic"
scp -r dist/dioramatic/* ableton@move.local:/data/UserData/schwung/modules/audio_fx/dioramatic/

# Set permissions so Module Store can update later
echo "Setting permissions..."
ssh ableton@move.local "chmod -R a+rw /data/UserData/schwung/modules/audio_fx/dioramatic"

echo ""
echo "=== Install Complete ==="
echo "Module installed to: /data/UserData/schwung/modules/audio_fx/dioramatic/"
echo ""
echo "Restart Schwung to load the new module."
