#!/bin/bash

# Install TripoSG
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d "TripoSG/.git" ]; then
    echo "TripoSG repo already installed."
else
    git clone https://github.com/VAST-AI-Research/TripoSG
fi

ENV_NAME="mocapanything"
# Check if environment exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Conda environment '$ENV_NAME' already exists."
    conda activate $ENV_NAME
else
    echo "Creating conda environment '$ENV_NAME'..."
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
fi

if [-d ]

pip install -r TripoSG/requirements.txt

