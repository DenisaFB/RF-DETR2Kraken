#!/bin/bash
set -e

if ! conda env list | grep -q "^layout_env "; then
  echo "Creating layout_env..."
  conda env create -f environment_layout.yml
else
  echo "layout_env already exists."
fi

if ! conda env list | grep -q "^kraken_env "; then
  echo "Creating kraken_env..."
  conda env create -f environment_kraken.yml
else
  echo "kraken_env already exists."
fi

echo "Environments ready."