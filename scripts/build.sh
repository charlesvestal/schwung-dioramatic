#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="move-anything-builder"

if [ -z "$CROSS_PREFIX" ] && [ ! -f "/.dockerenv" ]; then
    echo "=== Dioramatic Module Build (via Docker) ==="
    if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
        echo "Building Docker image..."
        docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile" "$REPO_ROOT"
    fi
    docker run --rm \
        -v "$REPO_ROOT:/build" \
        -u "$(id -u):$(id -g)" \
        -w /build \
        "$IMAGE_NAME" \
        ./scripts/build.sh
    exit 0
fi

CROSS_PREFIX="${CROSS_PREFIX:-aarch64-linux-gnu-}"
cd "$REPO_ROOT"
echo "=== Building Dioramatic Module ==="
mkdir -p build dist/dioramatic

${CROSS_PREFIX}gcc -Ofast -shared -fPIC \
    -march=armv8-a -mtune=cortex-a72 \
    -fomit-frame-pointer -fno-stack-protector \
    -DNDEBUG \
    src/dsp/dioramatic.c \
    -o build/dioramatic.so \
    -Isrc/dsp \
    -lm

cat src/module.json > dist/dioramatic/module.json
[ -f src/help.json ] && cat src/help.json > dist/dioramatic/help.json
cat build/dioramatic.so > dist/dioramatic/dioramatic.so
chmod +x dist/dioramatic/dioramatic.so

cd dist && tar -czvf dioramatic-module.tar.gz dioramatic/ && cd ..
echo "=== Build Complete: dist/dioramatic/ ==="
