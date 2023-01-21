#!/bin/bash
./clean.sh

# set -e

# build optix
cd ~/sph/rtnn/src/build
make
echo ==== BUILT OPTIX =====
cp lib/liboptixNSearch.so ~/sph/src/
cd ~/sph

# set constants for no RTNN
rm src/constants.h
touch src/constants.h
echo $'#define ENABLE_RTNN 0\n#define NEIGHBORS_PATH "neighbors-manual"' >> src/constants.h
echo ""
# run no RTNN
make
# timeout 10 ./Triforce --kokkos-threads=1
./Triforce --kokkos-threads=1

echo === Now running with RTNN ====

# set constants for RTNN
rm src/constants.h
touch src/constants.h
echo $'#define ENABLE_RTNN 1\n#define NEIGHBORS_PATH \"neighbors-rtnn\"' >> src/constants.h
# run with rtnn
make
# timeout 10 ./Triforce --kokkos-threads=1
./Triforce --kokkos-threads=1