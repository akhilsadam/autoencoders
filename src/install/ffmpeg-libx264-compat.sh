#!/usr/bin/env bash
set -e

INSTALL_DIR="$HOME/software/ffmpeg-libx264-compat"

echo "=== Installing static FFmpeg with libx264 support ==="
echo "Target directory: $INSTALL_DIR"
echo

# Create parent directory
mkdir -p "$HOME/software"
cd "$HOME/software"

# Download latest static build
echo "Downloading FFmpeg static build..."
wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz

# Extract
echo "Extracting archive..."
tar -xf ffmpeg-release-amd64-static.tar.xz

# Find extracted directory
EXTRACTED_DIR=$(find . -maxdepth 1 -type d -name "ffmpeg-*-amd64-static" | head -n 1)

if [ -z "$EXTRACTED_DIR" ]; then
    echo "ERROR: Could not find extracted FFmpeg directory."
    exit 1
fi

# Remove existing install if present
rm -rf "$INSTALL_DIR"

# Move to final location
mv "$EXTRACTED_DIR" "$INSTALL_DIR"

# Clean up archive
rm -f ffmpeg-release-amd64-static.tar.xz

# Add to PATH if not already present
if ! grep -q 'ffmpeg-libx264-compat' "$HOME/.bashrc"; then
    echo "export PATH=$INSTALL_DIR:\$PATH" >> "$HOME/.bashrc"
    echo "Added FFmpeg to PATH in ~/.bashrc"
fi

# Activate for current session
export PATH="$INSTALL_DIR:$PATH"

echo
echo "=== Verifying Installation ==="
echo "FFmpeg location:"
which ffmpeg
echo

echo "Checking version:"
ffmpeg -version | head -n 3
echo

echo "Checking for libx264 support:"
ffmpeg -encoders | grep libx264 || echo "libx264 NOT found!"
echo

echo "Testing CRF encoding..."
ffmpeg -f lavfi -i testsrc -c:v libx264 -crf 23 -t 2 test.mp4 -y

echo
echo "Installation complete."
echo "Open a new shell or run: source ~/.bashrc"
