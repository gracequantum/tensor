# GraceQ/tensor
_If it isn't fast, it isn't graceful_


## Features
- Support symmetry-blocked sparse real/complex tensor with any number of Abelian symmetries.
- Support user to define their own Abelian symmetry (quantum number).
- Support **asynchronous** high-performance tensor numerical manipulations.
- Header-only library.


## Design Goals
- Clear and easy-used tensor objects for quantum physics purposes. We believe usability is also a kind of high-performance.
- Faster tensor operation and arithmetic. Hard issues in quantum physics ask for this goal.
- Specific optimization for kinds of HPC hardware architectures. You can use consistent APIs to handle computing power on conventional shared/distributed memory computing systems and heterogeneous (especially for GPU) computing systems.


## Newest version
- version [0.2-alpha.0](https://github.com/gracequantum/tensor/releases)


## Current developers and maintainers
- Rong-Yang Sun <sun-rongyang@outlook.com>

> Note: For a complete list of the contributors, see CONTRIBUTORS.txt


## Development homepage
- On GitHub: [gracequantum/tensor](https://github.com/gracequantum/tensor)


## User guide
> In the following code blocks, if it is available, you can click the class or function name to jump to the detail documentation.

### Download GraceQ/tensor
You can always get the latest usable version using `git` from the default branch on [GitHub](https://github.com/gracequantum/tensor).
```
git clone https://github.com/gracequantum/tensor.git gqten
```
You can also download the release version from [release page](https://github.com/gracequantum/tensor/releases) and uncompress the file to get the source code.

### Installation
From now on, we suppose the root directory of the GraceQ/tensor source code is `gqten`. You can use [CMake](https://cmake.org/) 3.12 or higher version to install it.
```
cd gqten
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<your_gqten_installation_root>
make install
```
Where `<your_gqten_installation_root>` is the installation root directory which you have write access. For example, `~/.local` is a good choice.

GraceQ/tensor depends [hptt](https://github.com/springer13/hptt) to perform dense tensor transpose on shared memory computing system and has integrated it at `external/hptt`. GraceQ/tensor build and install hptt library using the same CMake setting by default. If you want to use external hptt library, set `GQTEN_USE_EXTERNAL_HPTT_LIB=ON`. the hptt library will also be installed to `<your_gqten_installation_root>` if `GQTEN_USE_EXTERNAL_HPTT_LIB` is set to `OFF` (default value).

GraceQ/tensor also depends Intel's [MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html) to perform heavy dense vector and matrix numerical calculations on shared memory computing system. So you should also install it before using GraceQ/tensor.

### Using GraceQ/tensor
It is easy to use the GraceQ/tensor.
```cpp
#include "gqten/gqten.h"
```
When you compile your program, you can use following compile flags to gain the best performance.
```
-std=c++11 -g -O3 -DNDEBUG
```
GraceQ/tensor needs hptt and MKL during linking process. So you should use the following statements when you link the library.
```
-lhptt <your_mkl_linking_flags>
```
We highly recommend that you use [MKL Link Line Advisor](https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/) to set `<your_mkl_linking_flags>`. A possible complete compiling command may looks like
```
g++ \
	-std=c++11 -g -O3 -DNDEBUG \
	-I<your_gqten_installation_root>/include -L<your_gqten_installation_root>/lib \
	-lhptt <your_mkl_linking_flags> \
	-o <your_main_program_name> <your_main_program_file_name>
```

### Symmetry-blocked sparse tensor
The central object in GraceQ/tensor is its symmetry-blocked sparse tensor, `GQTensor`. It is designed as a class template, `GQTensor<ElemT, QNT>`. The first template parameter `ElemT` sets the type of the tensor element. Real number(`GQTEN_Double`) and complex number(`GQTEN_Complex`) are supported now. The second template parameter `QNT` (Quantum Number Type) sets the _symmetry_ of the tensor. We will introduce how to define a quantum number later. Up to now, GraceQ/tensor supports the total symmetry \f$ G_{1} \otimes G_{2} \otimes G_{3} \otimes \cdots \f$, where \f$ G_{1} \f$, \f$ G_{2} \f$ ... are global Abelian symmetries the system has. For example, they may be the \f$ U(1) \f$ total particle number conservation or \f$ \mathbb{Z}_{n} \f$ discrete translational invariant.

#### Symmetry and quantum number
The objects in quantum physics, like states and operators, are defined in the Hilbert space. If the system has some symmetry, the Hilbert space of the system will be split as a series of unconnected regions which labeled by quantum number. For example, a plain spinless Fermion tight-binding model keeps the \f$ U(1) \f$ total Fermion number \f$ N \f$ conservation; a plain Heisenberg model keeps the \f$ U(1) \f$ total spin at z direction \f$ S^{z} \f$ conservation; a model living on a n circumference cylinder with only nearest neighbor interactions keeps the \f$ \mathbb{Z}_{n} \f$ discrete translational invariant. And a system can also keeps several kinds symmetries. For example, a plain \f$ t \f$-\f$ J \f$ model keeps \f$ U(1) \otimes U(1) \f$ total particle number and total z direction spin conservation.

In GraceQ/tensor, the quantum number value class \ref gqten::QNVal represents a specific symmetry. User can inherit it and define their own symmetry. Only Abelian symmetries are supported now. GraceQ/tensor has offered \f$ U(1) \f$ quantum number value \ref gqten::U1QNVal. For example, you can use this class to set a \f$ U(1) \f$ quantum number value labeled by the charge `n`:
```cpp
int n = val_of_qnval;
auto u1qnval_n = gqten::U1QNVal(n);
```
The type of total symmetry of the system is represented by \ref gqten::QN which take a series of quantum number value classes as the template parameters to support many symmetries case. For example, we can define the quantum number type as
```cpp
// Define symmetry for a plain Heisenberg system
using U1QN = gqten::QN<gqten::U1QNVal>;

// Define symmetry for a plain Hubbard system
using U1U1QN = gqten::QN<gqten::U1QNVal, gqten::U1QNVal>;
```
and create a quantum number with specific name and quantum number value pairs by using \ref gqten::QNCard as
```cpp
U1QN u1qn_n1({gqten::QNCard("N", gqten::U1QNVal(1))});
U1QN u1qn_n0({gqten::QNCard("N", gqten::U1QNVal(0))});

U1U1QN n1szm1(
		{
			 	 gqten::QNCard("N",  gqten::U1QNVal( 1)),
			 	 gqten::QNCard("Sz", gqten::U1QNVal(-1))
		}
);
```

#### Linear space split and labeled by symmetry sectors
We know the representation of the symmetry (groups) can label a specific linear space (i.e. the representation space). For example, A representation of the \f$ U(1) \f$ group with charge \f$ n \f$ label a one-dimensional representation space. The representations with different charges (degeneracy is allowed) can span a higher-dimensional linear space,
\f[
\mathbb{V} = \bigoplus_{n} \mathbb{V}^{n} d_{n}~,
\f]
where \f$ \mathbb{V}^{n} \f$ is the representation space with charge \f$ n \f$ (for Abelian group, it is a one-dimensional space), and \f$ d_{n} \f$ is the degeneracy of this representation. So, for Abelian group, the dimension of \f$ \mathbb{V} \f$ is \f$ \sum_{n} d_{n}\f$. The combination of a representation space and its degeneracy \f$ \mathbb{V}^{n} d_{n} \f$ is called a quantum number sector. In GraceQ/tensor, quantum number sector is represented by \ref gqten::QNSector
```cpp
// Define quantum number sector type
using QNSctT = gqten::QNSector<U1QN>;

// Create a quantum number sector labeled by particle number equals 1 with 2 fold degeneracy
QNSctT u1qnsct_n1_2(n1, 2);
```
So, we can use a series of quantum number sectors to describe a linear space.

#### Tensor as a linear mapping between linear spaces
A rank \f$ k \f$ tensor can be seen as a linear mapping between \f$ k \f$ linear spaces. Each linear space is labeled by an index of the tensor, \f$ i_{0}, i_{1}, \cdots, i_{k-1} \f$. As a mapping, we can define the _incoming_ spaces and _outgoing_ spaces. And the tensor construct the mapping from _incoming_ spaces to _outgoing_ spaces. So a linear space can own a direction and be defined by an index of a tensor. In GraceQ/tensor, a linear space with a direction is described by \ref gqten::Index
```cpp
// Define the index type
using IndexT = gqten::Index<U1QN>;

// The local Hilbert space for a spinless particle with outgoing direction
IndexT local_spinless_out(
		{QNSctT(n0, 1), QNSctT(n1, 1)},
		gqten::GQTenIndexDirType::OUT
);
```
where you can use a vector of quantum number sectors \ref gqten::QNSector and the direction of the space (also the direction of the index) \ref gqten::GQTenIndexDirType to initialize a linear space (tensor index). You can use \ref gqten::InverseIndex function to inverse the direction of an index.
```cpp
auto local_spinless_in = gqten::InverseIndex(local_spinless_out);
local_spinless_in.GetDir() == gqten::GQTenIndexDirType::IN;			// true
```

Once we have constructed the mapping spaces, the mapping, i.e. a tensor, can be defined straightforwardly. In GraceQ/tensor, a tensor is described by \ref gqten::GQTensor. This class template has two template parameters. The first one describes the type of the tensor elements, \ref gqten::GQTEN_Double for real number, and \ref gqten::GQTEN_Complex for complex number. The second one describes the type of the symmetry which needs be a \ref gqten::QN.
```cpp
// Define type of a tensor with U(1) symmetry and real number elements
using Tensor = gqten::GQTensor<gqten::GQTEN_Double, U1QN>;
```
Then, we can use mapping linear spaces, i.e. the tensor indexes, to initialize a tensor instance.
```cpp
// Default initialize a blank default tensor
// Note: the default tensor is **not** a rank 0 tensor (scalar)
Tensor default_ten;

// Initialize a rank 0 tensor, i.e. a scalar, using an empty indexes vector
Tensor scalar_ten({});

// Initialize a rank 1 tensor which can describe a local spinless particle state
Tensor local_spinless_state({local_spinless_out});

// Initialize a rank 2 tensor which can describe an operator on a local spinless particle site
Tensor local_spinless_op({local_spinless_in, local_spinless_out});

// Initialize a rank 3 tensor with one incoming index and two outgoing indexes
Tensor rank3_ten({local_spinless_in, local_spinless_out, local_spinless_out});
```

### Basic tensor operations
#### Get/set tensor element, get basic information, and I/O
The following sample code shows how to get/set the tensor element, how to get the basic information of a tensor, and how to read/write a tensor from/to a file. You can compile and run it to see what will happened.
```cpp
#include "gqten/gqten.h"

#include <iostream>
#include <fstream>
#include <string>


int main() {
  using QNT = gqten::QN<gqten::U1QNVal>;
  using QNSctT = gqten::QNSector<QNT>;
  using IndexT = gqten::Index<QNT>;
  using ElemtT = gqten::GQTEN_Double;
  using Tensor = gqten::GQTensor<ElemtT, QNT>;

  QNT pn0({gqten::QNCard("N", gqten::U1QNVal(0))});                                                                                                      
  QNT pn1({gqten::QNCard("N", gqten::U1QNVal(1))});                                                                                                      
  IndexT idx_in({QNSctT(pn0, 1), QNSctT(pn1, 1)}, gqten::GQTenIndexDirType::IN);                                                                         
  IndexT idx_out = gqten::InverseIndex(idx_in);                                                                                                          
                                                                                                                                                         
  // Define on-site particle number operator                                                                                                             
  Tensor ntot({idx_in, idx_out});                                                                                                                        
                                                                                                                                                         
  // Set tensor element using the coordinates by way of operator()                                                                                       
  // The index of the coordinate starting from 0                                                                                                         
  ntot(1, 1) = 1.0;                                                                                                                                      
                                                                                                                                                         
  // Get tensor element using the coordinates by way of operator()
  std::cout << "ntot(0, 0) = " << ntot(0, 0) << std::endl;
  std::cout << "ntot(0, 1) = " << ntot(0, 1) << std::endl;
  std::cout << "ntot(1, 0) = " << ntot(1, 0) << std::endl;
  std::cout << "ntot(1, 1) = " << ntot(1, 1) << std::endl;

  // Get the rank of the tensor
  std::cout << "The rank of ntot is " << ntot.Rank() << std::endl;

  // Get the shape of the tensor
  auto shape = ntot.GetShape();
  std::cout <<
      "The shape of ntot is" << " (" << shape[0] << ", " << shape[1] << ")"
  << std::endl;

  // Get the total number of elements, i.e. its size,  of the tensor
  std::cout << "The size of ntot is " << ntot.size() << std::endl;

  // Get the quantum number variation, i.e. the quantum number divergence, the tensor give rise to
  // If the tensor can not give rise to a uniform quantum number variation or
  // the tensor is a rank 0 tensor (scalar), an instance equals QNT() will be returned
  QNT qndiv = ntot.Div();
  if (qndiv == pn0) {
    std::cout << "The quantum number variation of ntot is QN<U1QNVal>(0)" << std::endl;
  }

  // Read/write a tensor from/to a stream
  // Write a tensor to file stream, i.e. output a tensor to hard disk
  std::string file = "ntot.gqten";
  std::ofstream out(file, std::ofstream::binary);
  out << ntot;
  out.close();

  // Read a tensor from a file stream, i.e. input a tensor from hard disk
  std::ifstream in(file, std::ifstream::binary);
  Tensor ntot_cpy;
  in >> ntot_cpy;
  in.close();
  std::cout << "Whether ntot_cpy equals ntot? " << (ntot_cpy == ntot) << std::endl;


  return 0;
}
```

#### Randomize, normalize, complex conjugate and complex convert
The following runnable sample code shows how to generate a random tensor by giving the quantum number variation, how to normalize a tensor, how to complex conjugate a tensor , and how to convert a real tensor to a complex tensor.
```cpp
#include "gqten/gqten.h"

#include <iostream>
#include <type_traits>


int main(void) {
  using QNT = gqten::QN<gqten::U1QNVal>;
  using QNSctT = gqten::QNSector<QNT>;
  using IndexT = gqten::Index<QNT>;
  using RealTensor = gqten::GQTensor<gqten::GQTEN_Double, QNT>;
  using CplxTensor = gqten::GQTensor<gqten::GQTEN_Complex, QNT>;

  QNT szp1({gqten::QNCard("Sz", gqten::U1QNVal(1))});
  QNT szm1({gqten::QNCard("Sz", gqten::U1QNVal(-1))});                                                                                                   
  QNT sz0({gqten::QNCard("Sz", gqten::U1QNVal(0))});                                                                                                     
  IndexT idx_in({QNSctT(szp1, 1), QNSctT(szm1, 1)}, gqten::GQTenIndexDirType::IN);                                                                       
  IndexT idx_out = gqten::InverseIndex(idx_in);                                                                                                          
                                                                                                                                                         
  // Generate a random tensor                                                                                                                            
  RealTensor rand_ten({idx_in, idx_out});                                                                                                                
  // Randomize tensor elements in the range of [0, 1], and hold the quantum number variation as sz0                                                      
  rand_ten.Random(sz0);                                                                                                                                  
  if (rand_ten.Div() == sz0) {                                                                                                                           
    std::cout << "The quantum number variation of rand_ten is QN<U1QNVal>(0)" << std::endl;                                                              
  }                                                                                                                                                      
  std::cout << "rand_ten(0, 0) = " << rand_ten(0, 0) << std::endl;
  std::cout << "rand_ten(0, 1) = " << rand_ten(0, 1) << std::endl;
  std::cout << "rand_ten(1, 0) = " << rand_ten(1, 0) << std::endl;
  std::cout << "rand_ten(1, 1) = " << rand_ten(1, 1) << std::endl;

  // Normalize a tensor,
  // and return the Euclidean norm (square root of the sum of all the square of the elements)
  auto ten_norm = rand_ten.Normalize();
  std::cout << "The norm of the rand_ten is " << ten_norm << std::endl;
  std::cout << "Normalized rand_ten(0, 0) = " << rand_ten(0, 0) << std::endl;
  std::cout << "Normalized rand_ten(0, 1) = " << rand_ten(0, 1) << std::endl;
  std::cout << "Normalized rand_ten(1, 0) = " << rand_ten(1, 0) << std::endl;
  std::cout << "Normalized rand_ten(1, 1) = " << rand_ten(1, 1) << std::endl;
  // Normalize a normalized tensor will get a norm equals 1
  ten_norm = rand_ten.Normalize();
  std::cout << "The norm of the rand_ten is " << ten_norm << std::endl;

  // Get the complex conjugate of a tensor
  // For real tensor, this operation only flip the direction of each indexes
  auto rand_ten_dag = gqten::Dag(rand_ten);
  auto rand_ten_dag_indexes = rand_ten_dag.GetIndexes();
  if (rand_ten_dag_indexes[0] == gqten::InverseIndex(idx_in)) {
    std::cout << "The direction of the first index of rand_ten_dag is OUT" << std::endl;
  }
  if (rand_ten_dag_indexes[1] == gqten::InverseIndex(idx_out)) {
    std::cout << "The direction of the first index of rand_ten_dag is IN" << std::endl;
  }
  // For complex tensor, this operation also complex conjugate each tensor elements
  CplxTensor cplx_rand_ten({idx_in, idx_out});
  cplx_rand_ten.Random(sz0);
  auto cplx_rand_ten_dag = gqten::Dag(cplx_rand_ten);
  std::cout << "cplx_rand_ten(0, 0) = " << gqten::GQTEN_Complex(cplx_rand_ten(0, 0)) << std::endl;
  std::cout << "cplx_rand_ten(0, 1) = " << gqten::GQTEN_Complex(cplx_rand_ten(0, 1)) << std::endl;
  std::cout << "cplx_rand_ten(1, 0) = " << gqten::GQTEN_Complex(cplx_rand_ten(1, 0)) << std::endl;
  std::cout << "cplx_rand_ten(1, 1) = " << gqten::GQTEN_Complex(cplx_rand_ten(1, 1)) << std::endl;
  std::cout << "cplx_rand_ten_dag(0, 0) = " << gqten::GQTEN_Complex(cplx_rand_ten_dag(0, 0)) << std::endl;
  std::cout << "cplx_rand_ten_dag(0, 1) = " << gqten::GQTEN_Complex(cplx_rand_ten_dag(0, 1)) << std::endl;
  std::cout << "cplx_rand_ten_dag(1, 0) = " << gqten::GQTEN_Complex(cplx_rand_ten_dag(1, 0)) << std::endl;
  std::cout << "cplx_rand_ten_dag(1, 1) = " << gqten::GQTEN_Complex(cplx_rand_ten_dag(1, 1)) << std::endl;

  // Convert a real tensor to complex tensor
  auto rand_ten_to_cplx = gqten::ToComplex(rand_ten);
  if (std::is_same<decltype (rand_ten_to_cplx), CplxTensor>::value) {
    std::cout << "The type of rand_ten_to_cplx is CplxTensor" << std::endl;
  }
  return 0;
}
```

### Basic tensor arithmetics
Up to now, following basic arithmetics are supported by \ref gqten::GQTensor.
- Unary negate
```cpp
Tensor ten(...);

// Some other operations

auto neg_ten = -ten;
```
- Sum two tensors together
```cpp
Tensor ten_a(...);
Tensor ten_b(...);

// Some other operations

auto a_add_b = ten_a + ten_b;
```
- Add to another tensor and assign back
```cpp
Tensor ten_a(...);
Tensor ten_b(...);

// Some other operations

ten_a += ten_b;			// Equal to ten_a = ten_a + ten_b, but more efficient
```
- Multiply by a scalar
```cpp
Tensor ten(...);
GQTEN_Double scalar;

// Some other operations

auto ten_2 = ten * scalar;
auto ten_3 = scalar * ten;
```
- Multiply by a scalar and assign back
```cpp
Tensor ten(...);
GQTEN_Double scalar;

// Some other operations

ten *= scalar;		// Equal to ten = ten * scalar, but more efficient
```

### Tensor numerical manipulations
#### Tensor transpose
Since the data in memory are stored in a certain order, we need the indexes transpose manipulation. Because the cost of a tensor transpose is about O(N), the order of tensor indexes will influence the performance. In GraceQ/tensor, we can perform in-place tensor transpose by the member function of GQTensor, \ref gqten::GQTensor::Transpose.
```cpp
Tensor ten({idx0, idx1, idx2});			// T_{ijk}

// Some other operations

// In-place tensor transpose: T_{ijk} -> T_{jik}
ten.Transpose({1, 0, 2});			// {1, 0, 2} is the new order of previous indexes
```

#### Tensor contraction
Tensor contraction is implemented as a [_numpy-like_ API](https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html) in GraceQ/tensor \ref gqten::Contract.
```cpp
Tensor ta({idx0_in, idx1_out, idx2_out});
Tensor tb({idx1_in, idx0_out, idx3_out});

// Treat ta and tb ...

Tensor tc;

// Contract the 0th index of A with 1th index of B, and the 1th index of A with 0th index of B
gqten::Contract(
		&ta,
		&tb,
		{{0, 1}, {1, 0}},			// {to-be-ctrct-axes-in-ta, to-be-ctrct-axes-in-tb} the only difference between contracted indexes pair is their index direction
		&tc
);

tc.GetIndexes();		// Will return {idx2_out, idx3_out}
```

#### Tensor decomposition: SVD
The truncated tensor singular value decomposition (SVD) is performed in GraceQ/tensor \ref gqten::SVD.
```cpp
Tensor t(...);

// Some treatments ...

size_t ldims = ...;
QN lqndiv(...);
gqten::GQTEN_Double trunc_err = ...;
size_t Dmin = ...;
size_t Dmax = ...;

Tensor u, vt;
RealTensor s;
gqten::GQTEN_Double actual_trunc_err;
size_t D;


// Perform T = Ucut * Scut * VTcut with singular value spectrum cutoff
gqten::SVD(
    &t,                   // Input tensor T
    ldims,                // ldims number of indexes counted from begin of the index list
    lqndiv,               // Quantum number variation of the tensor Ucut
    trunc_err,            // Maximal tolerance truncation error
    Dmin,                 // Minimal number of singular values kept
    Dmax,                 // Maximal number of singular values kept
    &u, &s, &vt,          // Results. Tensor Ucut, Scut and VTcut
    &actual_trunc_err,    // Actual truncation error
    &D                    // Actual number of singular values kept
);
```
The singular value spectrum truncation policies are
- If the raw number of singular values of the effective matrix is smaller than the `Dmin`, keep all the spectrum.
- If the spectrum touches the `trunc_err` first but its number is smaller than the `Dmin`, keep `Dmin` number of singular values.
- If the spectrum touches the `trunc_err` first and its number is larger than the `Dmin` but smaller than the `Dmax`, keep number of singular values which satisfies the `trunc_err`.
- If the spectrum touches the `Dmax` first but `trunc_err` not satisfied, keep `Dmax` number of singular values.

### Utilities for tensor manipulation
#### Tensor index combination
To combine two indexes of a tensor, you can generate an index combiner according to these two indexes using \ref gqten::IndexCombine
```cpp
Index index_1(...);
Index index_2(...);
auto combined_index_direction = gqten::GQTenIndexDirType::IN;     // or OUT. Define the direction of the new combined index

auto combiner = IndexCombine<TenElemT, QNT>(
                    index_1,    // The first index to be combined
                    index_2,    // The second index to be combined
                    combined_index_direction
                );
```
The `combiner` is a tensor initialized as `GQTensor<TenElemT, QNT>( {InverseIndex(idx1), InverseIndex(idx2), new_idx} )` and only elements corresponding to index combination equal to 1. To combine two specific indexes of a tensor, you can create the corresponding combiner and contract the tensor with the combiner.

### Advanced topics
#### Customize symmetry (quantum number value)
...

#### Asynchronous tensor numerical manipulations
...


## Developer guide
...


## License
GraceQ/tensor is freely available under the [LGPL v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html) licence.


## How to cite
You can cite the GraceQ/tensor where you use it as a support to this project. Please cite GraceQ/tensor as
> GraceQuantum.org . GraceQ/tensor: A high-performance tensor computation framework for the quantum physics community. For a complete list of the contributors, see CONTRIBUTORS.txt .


## Acknowledgments
We highly acknowledge the following people, project(s) and organization(s) (sorted in alphabetical order):

ALPS project, Chunyu Sun, Donna Sheng, Grace Song, Hong-Chen Jiang, Hong-Hao Tu, Hui-Ke Jin, itensor.org, Le Zhao, Shuo Yang, Thomas P. Devereaux, Wayne Zheng, Xiaoyu Dong, Yi Zhou, Yifan Jiang, Yifeng Chen, Zheng-Yu Weng

You can not meet this project without anyone of them. And the basic part of this project (before version 0.1) was developed by Rong-Yang Sun and Cheng Peng, when Rong-Yang Sun was a visiting student at Stanford University. So R.-Y. Sun want to give special thanks to his co-advisors Hong-Chen Jiang, Prof. Thomas P. Devereaux and their postdoctors Yifan Jiang and Cheng Peng.
