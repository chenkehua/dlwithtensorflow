#!/bin/bash

OPENCV_DIR="/paddle-infer/opencv-3.4.7"
# Create build directory if it doesn't exist
mkdir -p build

# Navigate to the build directory
cd build

# Run cmake
cmake ..

# Compile the project
make

# Return to the original directory
cd ..
