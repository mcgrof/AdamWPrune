#!/bin/bash
# SPDX-License-Identifier: MIT
# GPU detection script for multi-GPU training

# Detect GPU vendor and count
detect_gpu_count() {
    local gpu_count=0

    # Try NVIDIA first
    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
        if [ "$gpu_count" -gt 0 ]; then
            echo "$gpu_count"
            return 0
        fi
    fi

    # Try AMD ROCm
    if command -v rocm-smi &> /dev/null; then
        # Count unique GPU device IDs (e.g., GPU[0], GPU[1])
        gpu_count=$(rocm-smi --showproductname 2>/dev/null | grep -oP '^GPU\[\K\d+' | sort -u | wc -l)
        if [ "$gpu_count" -gt 0 ]; then
            echo "$gpu_count"
            return 0
        fi
    fi

    # Try rocminfo for AMD
    if command -v rocminfo &> /dev/null; then
        gpu_count=$(rocminfo 2>/dev/null | grep -c "Name:.*gfx")
        if [ "$gpu_count" -gt 0 ]; then
            echo "$gpu_count"
            return 0
        fi
    fi

    # Try lspci as fallback
    if command -v lspci &> /dev/null; then
        # Count NVIDIA GPUs
        local nvidia_count=$(lspci 2>/dev/null | grep -i "vga.*nvidia\|3d.*nvidia" | wc -l)
        # Count AMD GPUs
        local amd_count=$(lspci 2>/dev/null | grep -i "vga.*amd\|3d.*amd\|vga.*radeon\|3d.*radeon" | wc -l)
        gpu_count=$((nvidia_count + amd_count))
        if [ "$gpu_count" -gt 0 ]; then
            echo "$gpu_count"
            return 0
        fi
    fi

    # Default to 1 if no GPUs detected (assume single GPU system)
    echo "1"
    return 0
}

# Detect GPU vendor
detect_gpu_vendor() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi --list-gpus &> /dev/null; then
            echo "nvidia"
            return 0
        fi
    fi

    if command -v rocm-smi &> /dev/null; then
        echo "amd"
        return 0
    fi

    if command -v rocminfo &> /dev/null; then
        echo "amd"
        return 0
    fi

    # Use lspci as fallback
    if command -v lspci &> /dev/null; then
        if lspci 2>/dev/null | grep -qi "nvidia"; then
            echo "nvidia"
            return 0
        fi
        if lspci 2>/dev/null | grep -qi "amd\|radeon"; then
            echo "amd"
            return 0
        fi
    fi

    echo "unknown"
    return 1
}

# Main logic
case "$1" in
    count)
        detect_gpu_count
        ;;
    vendor)
        detect_gpu_vendor
        ;;
    *)
        echo "Usage: $0 {count|vendor}"
        exit 1
        ;;
esac
