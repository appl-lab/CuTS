#!/bin/sh
cudabinpath=''
cudalibpath=''
mkdir build
cd build
export LD_LIBRARY_PATH=$cudalibpath:$LD_LIBRARY_PATH
export PATH=$cudabinpath:$PATH
pwd
cmake ..
make -j
cd ..