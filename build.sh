#!/bin/sh
# Builds zero on Linux: bin/zero (command line) and bin/zero-gui (GTK 3 window, loaded at run time with dlopen:
# no GTK development files are needed to build, only libgtk-3.so.0 to run the window).
# Needs g++ 13+ or clang++ 17+ (C++20). Usage: ./build.sh [cli|gui]
set -e
cd "$(dirname "$0")"
CXX="${CXX:-g++}"
mkdir -p bin
# the kernels, embedded into the exe as a byte array
sh scripts/embed_kernels.sh kernels/zero.cl src/kernels_gen.h
FLAGS="-std=c++20 -O2 -Wall -Wextra -pthread -Isrc"
if [ "$1" != "gui" ]; then
  echo "building bin/zero"
  $CXX $FLAGS -o bin/zero src/main.cpp -ldl
fi
if [ "$1" != "cli" ]; then
  echo "building bin/zero-gui"
  $CXX $FLAGS -DZERO_GUI -o bin/zero-gui src/main.cpp -ldl
fi
echo done
