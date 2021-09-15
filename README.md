<meta name="robots" content="noindex">

# APEX: High Performance Learned Index on Persistent Memory

Persistent memory friendly learned index.

More details are described in our [preprint](https://arxiv.org/abs/2105.00683).

## What's included

- APEX - the source code of APEX
- Mini benchmark framework

Fully open-sourced under MIT license.

## Building

### Dependencies
We tested our build with Linux Kernel 5.10.11 and GCC 10.2.0. You must ensure that your Linux kernel version >= 4.17 and glibc >=2.29 for proper build. 

### Compiling
Assuming to compile under a `build` directory:
```bash
git clone https://github.com/baotonglu/apex.git
cd apex
./build.sh
```

## Running benchmark

We run the tests in a single NUMA node with 24 physical CPU cores. We pin threads to physical cores compactly assuming thread ID == 2 * core ID (e.g., for a dual-socket system, we assume cores 0, 2, 4, ... are located in socket 0).  Check out also the `total.sh` and `full.sh` script for example benchmarks and easy testing of the index. 

## Competitors
Here hosts source codes which are used in comparision with APEX , including LB+-Tree [1], DPTree [2], uTree [3], FPTree [4], BzTree [5] and FAST+FAIR [6].

[1] https://github.com/schencoding/lbtree<br/>
[2] https://github.com/zxjcarrot/DPTree-code<br/>
[3] https://github.com/thustorage/nvm-datastructure<br/>
[4] https://github.com/sfu-dis/fptree<br/>
[5] https://github.com/sfu-dis/bztree<br/>
[6] https://github.com/DICL/FAST_FAIR

## Datasets
- [Longitudes (200M 8-byte floats)](https://drive.google.com/file/d/1zc90sD6Pze8UM_XYDmNjzPLqmKly8jKl/view?usp=sharing)
- [Longlat (200M 8-byte floats)](https://drive.google.com/file/d/1mH-y_PcLQ6p8kgAz9SB7ME4KeYAfRfmR/view?usp=sharing)
- [Lognormal (190M 8-byte ints)](https://drive.google.com/file/d/1y-UBf8CuuFgAZkUg_2b_G8zh4iF_N-mq/view?usp=sharing)
- [YCSB (200M 8-byte ints)](https://drive.google.com/file/d/1Q89-v4FJLEwIKL3YY3oCeOEs0VUuv5bD/view?usp=sharing)
- [FB (200M 8-byte ints)](https://github.com/learnedsystems/SOSD)
- [TPCE (259M 8-byte ints)](https://github.com/sfu-dis/ermia/tree/master/benchmarks/tpce_keys)


## Acknowledgements
Our implementation is based on the code of [ALEX](https://github.com/microsoft/ALEX).
