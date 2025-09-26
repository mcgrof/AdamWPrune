#!/bin/bash
# SPDX-License-Identifier: MIT
# Generate a smart project name based on model and environment

MODEL=$1
TRACKER=$2

# Get hostname and IP
HOSTNAME=$(hostname -s 2>/dev/null || echo "unknown")
IP=$(hostname -I 2>/dev/null | awk '{print $1}' | sed 's/\./-/g' || echo "0-0-0-0")
DIR=$(basename $(pwd))

# Create a checksum from hostname, IP, and directory
CHECKSUM=$(echo "${HOSTNAME}-${IP}-${DIR}" | md5sum | cut -c1-5)

# Generate project name based on model
if [ -z "$MODEL" ]; then
    MODEL="adamwprune"
fi

# Clean model name (remove special characters)
MODEL_CLEAN=$(echo "$MODEL" | sed 's/[^a-zA-Z0-9-]/-/g')

# Generate project name
PROJECT="${MODEL_CLEAN}-${CHECKSUM}"

echo "$PROJECT"
