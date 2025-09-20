#!/bin/bash
# SPDX-License-Identifier: MIT
# Append or use Makefile variable value

VAR_VALUE=$1

if [ -n "$VAR_VALUE" ]; then
    echo "$VAR_VALUE"
else
    echo ""
fi