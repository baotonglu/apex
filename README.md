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

## Acknowledgements

Our implementation is based on the code of [ALEX](https://github.com/microsoft/ALEX).
Longitudes dataset is open-sourced in ALEX's repo.
