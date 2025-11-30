#!/bin/bash

echo "=== PCIe Topology Tree ==="
lspci -tv | grep -E "NVIDIA|VGA|Display|RTX" -B 5 -A 5

echo -e "\n=== GPU PCIe Addresses ==="
lspci | grep NVIDIA
