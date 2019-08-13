# GraceQ/tensor 
A high-performance tensor library for quantum physics community.

_"If it isn't fast, it isn't graceful"_

## Design Goals

- Clear and easy-used tensor objects for quantum physics purposes. We believe usability is also a kind of high-performance.
- Faster tensor operation and arithmetic. Hard issues in quantum physics ask for this goal.
- Specific optimization for kinds of HPC hardware architectures. You can use consistent APIs to handle computing power on shared memory, distributed memory and heterogeneous (especially for GPU) systems.

## Installation

Directly use
```shell
git clone git clone --recurse-submodules https://github.com/gracequantum/tensor.git gqten
cd gqten
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
Find library file `libgqten.a` at `./lib`

Build unittests
```bash
cmake .. -DGQTEN_BUILD_UNITTEST=ON
make
```
Run unittests
```shell
make test
```
or
```sh
ctest
```


## Minimal tutorial

## TODO list

This TODO list is *not* sorted by expected completion order.

- Complete distributed-memory parallel tensor numerical functions.
- Support constructing tensor in the graphic memory and GPU accelerated tensor numerical functions.
- Support dense tensor objects and numerical functions performing on them under conformed APIs.


## License

GraceQ/tensor is freely available under the [LGPL v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html) licence.

## How to cite

Cite GraceQ/tensor as
> "GraceQ/tensor: A high-performance tensor library for quantum physics community.", https://github.com/gracequantum/tensor .

## Acknowledgments

The author(s) highly acknowledge the following people and organization(s) (sorted in alphabetical order):

Cheng Peng, Chunyu Sun, D. N. Sheng, Grace Song, Hong-Chen Jiang, itensor.org, Le Zhao, Shuo Yang, Thomas P. Devereaux, Wayne Zheng, Xiaoyu Dong, Yifan Jiang, Zheng-Yu Weng

You can not meet this library without anyone of them.
