#!/bin/bash
# Install external packages from package_requirements.txt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOENCODERS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PACKAGES_DIR="${AUTOENCODERS_ROOT}/packages"
PYTHON_PATH="${1}"
FORCE_REINSTALL="${2:-false}"

# Convert relative path to absolute if needed
if [[ ! "$PYTHON_PATH" = /* ]]; then
    PYTHON_PATH="${AUTOENCODERS_ROOT}/${PYTHON_PATH}"
fi

mkdir -p "${PACKAGES_DIR}"

while read -r name url; do
    [[ "$name" =~ ^#.*$ || -z "$name" ]] && continue
    
    if [ ! -d "${PACKAGES_DIR}/${name}" ]; then
        echo "📦 Cloning ${name}..."
        git clone "${url}" "${PACKAGES_DIR}/${name}"
        echo "📦 Installing ${name}..."
        cd "${PACKAGES_DIR}/${name}" && uv pip install -e . --python "${PYTHON_PATH}" --extra local
    else
        echo "✓ ${name} already exists, pulling latest..."
        if [ "$FORCE_REINSTALL" = "true" ]; then
            cd "${PACKAGES_DIR}/${name}" && git pull
            echo "📦 Reinstalling ${name}..."
            uv pip install -e . --python "${PYTHON_PATH}"
        else
            (cd "${PACKAGES_DIR}/${name}" && git pull) &
        fi
    fi
done < "${SCRIPT_DIR}/package_requirements.txt"
wait
echo "✅ Packages ready"

