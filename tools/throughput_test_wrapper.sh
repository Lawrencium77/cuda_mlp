#!/bin/bash

# This script is used to fix GPU blocks before obtaining throughput/latency statistics.
# See https://blog.speechmatics.com/cuda-timings#fixed-clocks

DEVICE=${CUDA_VISIBLE_DEVICES:-0}
CLOCK_SPEED=1590

function set_clock_speed {
    echo "Setting clock speed to $CLOCK_SPEED MHz..."
    sudo nvidia-smi -pm ENABLED -i $DEVICE
    sudo nvidia-smi -lgc $CLOCK_SPEED -i $DEVICE
}

function reset_clock_speed {
    echo "Resetting clock speed..."
    sudo nvidia-smi -pm ENABLED -i $DEVICE
    sudo nvidia-smi -rgc -i $DEVICE
}

trap reset_clock_speed EXIT # Ensures reset_clock_speed is called
set_clock_speed
"$@"
