#!/bin/bash

#APEX evaluation
./run.sh longitudes-200M.bin.data binary double 100000000 200000000 search 1 apex 1 0 > apex-longitudes-search.data
./run.sh longitudes-200M.bin.data binary double 100000000 200000000 insert 1 apex 1 0 > apex-longitudes-insert.data
