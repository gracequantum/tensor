# GraceQ/tensor 
A high-performance tensor library for the quantum physics community

_"If it isn't fast, it isn't graceful"_

## Design Goals

- Clear and easy-used tensor objects for quantum physics purposes. We believe usability is also a kind of high-performance.
- Faster tensor operation and arithmetic. Hard issues in quantum physics ask for this goal.
- Specific optimization for kinds of HPC hardware architectures. You can use consistent APIs to handle computing power on conventional shared/distributed memory computing systems and heterogeneous (especially for GPU) computing systems.
- Designed as an infrastructure in this field.

## Installation

### Common usage

GraceQ/tensor library depends [hptt](https://github.com/springer13/hptt) to perform dense tensor transpose on shared memory computing system. And it has been integrated as a git submodule. So you can use the `--recurse-submodules` option to clone it recursively.

```
git clone --recurse-submodules https://github.com/gracequantum/tensor.git gqten
```

Then you can use the CMake tool to build the library.

```
cd gqten
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

Finally, you can get the library file `libgqten.a` at `./lib/libgqten.a` and the dependence `hptt` library file `libhptt.a` at `./external/hptt/libhptt.a`.

### Build unittests

GraceQ/tensor uses [Google Test](https://github.com/google/googletest) as its unittest framework. You should install the Google Test first and then turn on the `GQTEN_BUILD_UNITTEST` option.

```
cd gqten
mkdir build && cd build
cmake .. -DGQTEN_BUILD_UNITTEST=ON
make
```

Run the unittests using

```
make test
```
or
```
ctest
```
GraceQ/tensor has been tested on Linux and MacOS X systems.

### Timing mode

Set `GQTEN_TIMING_MODE=ON` to turn on the timing mode when you build the library. Then massive timing information will be printed, when you call tensor numerical functions.


## Minimal tutorial

### Using GraceQ/tensor

It is easy to use the GraceQ/tensor.

```cpp
#include "gqten/gqten.h"
using namespace gqten;
```

### Sparse tensor object with U1 quantum number

As an example, we will create  ![S^z](https://latex.codecogs.com/svg.latex?S%5Ez) operator living in a two-dimensional spin 1/2 Hilbert space.

First, we define the quantum numbers.
```cpp
auto qn_up = QN({QNNameVal("Sz",  1)});     // qn_up: Up quantum number
auto qn_dn = QN({QNNameVal("Sz", -1)});     // qn_dn: Down quantum number
```
The `QNNameVal` object carry the name string of the quantum number and the value of the quantum number which must be an integer. The constructor of `QN` class accepts a vector of `QNNameVal` objects to create a quantum number with any number of different U1 quantum numbers.

Then we define the quantum number sector. We know that a spin 1/2 Hilbert space has two quantum number sectors. Each one of them has one dimension.
```cpp
auto qnsct_up = QNSector(qn_up, 1);         // qnsct_up: Quantum number sector with up spin
auto qnsct_dn = QNSector(qn_dn, 1);         // qnsct_dn: quantum number sector with down spin
```

We know that physical operator like ![S^z](https://latex.codecogs.com/svg.latex?S%5Ez) can be represented as a tensor with two legs. Now we define the leg which is called `Index` in GraceQ/tensor.
```cpp
auto idx_in  = Index({qnsct_up, qnsct_dn}, IN);       // idx_in: Index with in-direction quantum number flow.
auto idx_out = Index({qnsct_up, qnsct_dn}, OUT);      // idx_out: Index with out-direction quantum number flow.
```

Once we have the legs with quantum number informations, we can create the tensor object to represent the ![S^z](https://latex.codecogs.com/svg.latex?S%5Ez) operator and set the non-zero elements.
```cpp
auto sz_op = GQTensor({idx_in, idx_out});       // Create GraceQ/tensor sparse tensor object.
sz_op({0, 0}) =  0.5;                           // Set the spin up element.
sz_op({1, 1}) = -0.5;                           // Set the spin down element.
```

### Tensor numerical functions

#### Contraction

Tensor contraction is implemented as a _numpy-like_ API in GraceQ/tensor. The following example performs ![C=AcdotB](https://latex.codecogs.com/svg.latex?C%20%3D%20A%20%5Ccdot%20B) matrices dot product using the tensor contraction.
```cpp
// Treat ta and tb ...

GQTensor tc;

Contract(
    &ta,         // Tensor A
    &tb,         // Tensor B
    {{1}, {0}},     // Contract the 1th leg of A with 0th leg of B
    &tc);    // Contraction result, tensor C
```

#### Linear combination

Tensor linear combination is defined as

![T_{res}=T_{res}+acdotA+bcdotB+ccdotC+cdots](https://latex.codecogs.com/svg.latex?T_%7Bres%7D%20%3D%20T_%7Bres%7D%20&plus;%20a%20%5Ccdot%20A%20&plus;%20b%20%5Ccdot%20B%20&plus;%20c%20%5Ccdot%20C%20&plus;%20%5Ccdots)

where all the treated tensors must have the same shape.
```cpp
GQTensor tres, ta, tb, tc;
double a, b, c;

// Some treatments ...

LinearCombine(
    {a, b, c},
    {&ta, &tb, &tc},
    &tres);
```

#### Singular value decomposition with cutoff

GraceQ/tensor performs the generic tensor SVD with singular value spectrum cutoff by some constraints using the following API.
```cpp
GQTensor t;
long ldims, rdims;
QN ldiv, rdiv;
double cutoff;
long Dmin, Dmax;

// Some treatments ...

GQTensor u, s, vt;
double trunc_err;
long D;

Svd(                // Perform T = Ucut * Scut * VTcut with singular value spectrum cutoff
    &t,             // Input tensor T
    ldims,          // ldims number of legs counted from begin of the leg list will be reshaped as the rows of the effective matrix.
    rdims,          // rdims number of legs counted from the end of the leg list will be reshaped as the columns of the effective matrix.
    ldiv, rdiv,     // Quantum number shifts by the tensor Ucut and tensor VTcut.
    cutoff,         // Maximal tolerance truncation error.
    Dmin,           // Minimal number of singular values kept.
    Dmax,           // Maximal number of singular values kept.
    &u, &s, &vt,    // Results. Tensor Ucut, Scut and VTcut.
    &trunc_err,     // Real truncation error.
    &D);            // Real number of singular values kept.
```
The singular value spectrum cutoff policies are
- If the raw number of singular values of the effective matrix is smaller than the `Dmin`, keep all the spectrum.
- If the spectrum touches the `cutoff` first but its number is smaller than the `Dmin`, keep `Dmin` number of singular values.
- If the spectrum touches the `cutoff` first and its number is larger than the `Dmin` but smaller than the `Dmax`, keep number of singular values which satisfies the `cutoff`.
- If the spectrum touches the `Dmax` first but `cutoff` not satisfied, keep `Dmax` number of singular values.

## TODO list

This TODO list is *not* sorted by expected completion order.

- Complete distributed-memory parallel tensor numerical functions.
- Support constructing tensor in the graphic memory and GPU accelerated tensor numerical functions.
- Support dense tensor objects and their related numerical functions under consistent APIs.
- Support tensor objects with complex number elements under consistent APIs.
- ...


## License

GraceQ/tensor is freely available under the [LGPL v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html) licence.

## How to cite

Cite GraceQ/tensor as
> "GraceQ/tensor: A high-performance tensor library for the quantum physics community", https://github.com/gracequantum/tensor .

## Acknowledgments

The author(s) highly acknowledge the following people, project(s) and organization(s) (sorted in alphabetical order):

ALPS project, Cheng Peng, Chunyu Sun, D. N. Sheng, Grace Song, Hong-Chen Jiang, itensor.org, Le Zhao, Shuo Yang, Thomas P. Devereaux, Wayne Zheng, Xiaoyu Dong, Yifan Jiang, Zheng-Yu Weng

You can not meet this library without anyone of them.
