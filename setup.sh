#!/bin/bash
# setup.sh (Based on sources 188, 260-263)

echo "--- Starting GA-LSTM UAV-NS2 Setup ---"

# 1. Update package lists
echo "[SETUP] Updating package lists..."
sudo apt-get update -y

# 2. Install System Dependencies (NS2, Python, GCC/G++)
echo "[SETUP] Installing system dependencies (NS2, Python3, pip, g++)..."
sudo apt-get install -y ns2 python3 python3-pip g++

# 3. Install Python Dependencies
echo "[SETUP] Installing Python dependencies from requirements.txt..."
pip3 install -r requirements.txt

# 4. Create required directories (implied by source 159)
echo "[SETUP] Creating 'uav_positions' directory..."
mkdir -p uav_positions

echo "--- Setup Complete ---"
echo "You can now run the test suite: python3 test_system.py"
echo "Or run the main simulation: python3 hybrid_path_planner.py"