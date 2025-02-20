#!/bin/bash

# Define the base URL for the checkpoints
BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/"

# Define available models
declare -A models
models["tiny"]="sam2_hiera_tiny.pt"
models["small"]="sam2_hiera_small.pt"
models["base_plus"]="sam2_hiera_base_plus.pt"
models["large"]="sam2_hiera_large.pt"

# Function to display usage
usage() {
    echo "Usage: $0 [tiny|small|base_plus|large|all]"
    echo "Example: $0 tiny small"
    echo "         This will download the tiny and small models."
    echo "         Use 'all' to download all available models."
    exit 1
}

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
    usage
fi

# Function to download a model
download_model() {
    local model_key=$1
    local model_file=${models[$model_key]}
    
    if [ -z "$model_file" ]; then
        echo "Invalid model name: $model_key"
        usage
    fi

    local url="${BASE_URL}${model_file}"

    echo "Downloading $model_file..."
    wget "$url" || { echo "Failed to download $model_file"; exit 1; }
    echo "$model_file downloaded successfully."
}

# Handle "all" argument separately
if [[ " $* " =~ " all " ]]; then
    for key in "${!models[@]}"; do
        download_model "$key"
    done
else
    for model_key in "$@"; do
        download_model "$model_key"
    done
fi

echo "Download process completed."
