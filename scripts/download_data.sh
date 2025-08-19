#!/bin/bash

#=========================================#
PROJ_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_DIR=${PROJ_DIR}/data

mkdir -p ${DATA_DIR}
cd ${DATA_DIR}

#=========================================#
# Check if authenticated with Hugging Face
echo "Checking Hugging Face authentication..."
HF_STATUS=$(uv run hf whoami 2>&1)
if echo "$HF_STATUS" | grep -q "Not logged in"; then
    echo "❌ Not logged in to Hugging Face!"
    echo "Please run: uv run hf login"
    echo "You'll need a Hugging Face account and token from https://huggingface.co/settings/tokens"
    exit 1
fi

echo "✅ Authenticated with Hugging Face"
echo "🔒 Downloading data..."

#=========================================#
# DummyData
#=========================================#
mkdir -p DummyData
echo "Downloading DummyData dataset..."
uv run hf download --repo-type dataset vedantpuri/PDESurrogates DummyData.tar.gz --local-dir .
echo "Extracting DummyData dataset..."
tar -xzf DummyData.tar.gz
rm -rf DummyData.tar.gz

#=========================================#
#