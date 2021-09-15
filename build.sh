#!/usr/bin/env bash

cmake5="cmake -DCMAKE_C_COMPILER=/home/baotong/GCC-10.2.0/bin/gcc -DCMAKE_CXX_COMPILER=/home/baotong/GCC-10.2.0/bin/g++"
if [[ "$#" -ne 0 && $1 == "debug" ]]
then
    mkdir -p build_debug;
    cd build_debug;
    cmake -DCMAKE_BUILD_TYPE=Debug ..;
else
    mkdir -p build;
    cd build;
    $cmake5 -DCMAKE_BUILD_TYPE=Release -DDESC_CAP=16 ..;
fi
make;
cd ..;