#!/bin/bash
# SPDX-License-Identifier: MIT
# Check if a variable was set via CLI (make command line)

VAR_NAME=$1

if [ -z "$VAR_NAME" ]; then
    echo "n"
    exit 0
fi

# Check if the variable is set in the environment
VAR_VALUE="${!VAR_NAME}"

if [ -n "$VAR_VALUE" ]; then
    echo "y"
else
    echo "n"
fi