#!/bin/bash
# SPDX-License-Identifier: MIT
# Check if a specific tracker is enabled in TRACKER variable
# Supports: TRACKER=wandb, TRACKER=trackio, TRACKER=both, TRACKER=wandb,trackio

TRACKER_NAME=$1
TRACKER_VALUE=$2

if [ -z "$TRACKER_NAME" ] || [ -z "$TRACKER_VALUE" ]; then
    echo "n"
    exit 0
fi

# Convert to lowercase for comparison
TRACKER_NAME_LOWER=$(echo "$TRACKER_NAME" | tr '[:upper:]' '[:lower:]')
TRACKER_VALUE_LOWER=$(echo "$TRACKER_VALUE" | tr '[:upper:]' '[:lower:]')

# Check if tracker is in the value
# Handles: wandb, trackio, both, wandb,trackio, trackio,wandb
if [ "$TRACKER_VALUE_LOWER" = "both" ]; then
    echo "y"
elif [ "$TRACKER_VALUE_LOWER" = "none" ]; then
    echo "n"
elif echo "$TRACKER_VALUE_LOWER" | grep -q "$TRACKER_NAME_LOWER"; then
    echo "y"
else
    echo "n"
fi