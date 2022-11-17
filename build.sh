#!/bin/bash

set -e
cd ~/sph/rtnn/src/build
make
echo ==== BUILT OPTIX =====
cp lib/liboptixNSearch.so ~/sph/src/
cd ~/sph
make
echo ==== BUILT TRIFORCE ====
echo
echo ==========
echo
./Triforce