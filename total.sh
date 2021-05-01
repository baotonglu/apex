#!/bin/bash

#APEX evaluation
./full.sh longitudes-200M.bin.data binary double 100000000 200000000 full 1 alex 1 0 > apex-longitudes-full.data
./full.sh longitudes-200M.bin.data binary double 100000000 200000000 insert 1 alex 1 0 > apex-longitudes-insert.data
./full.sh longitudes-200M.bin.data binary double 90000000 200000000 erase 1 alex 1 0 > apex-longitudes-erase.data