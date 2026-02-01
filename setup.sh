#!/bin/bash
# Setup script for CS336 Assignment 5 environment

set -e  # Exit on error

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH for current session
export PATH="$HOME/.local/bin:$PATH"

echo "Running uv sync..."
uv sync

echo "Setup complete!"
