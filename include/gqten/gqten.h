// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-25 14:33
* 
* Description: GraceQ/tensor project. The main header file.
*/
#ifndef GQTEN_GQTEN_H
#define GQTEN_GQTEN_H


#define GQTEN_VERSION_MAJOR 0
#define GQTEN_VERSION_MINOR 2
#define GQTEN_VERSION_PATCH 0
#define GQTEN_VERSION_SUFFIX "alpha"
// GQTEN_VERSION_DEVSTR to describe the development status, for example the git branch
#define GQTEN_VERSION_DEVSTR


#include "gqten/framework/consts.h"
#include "gqten/framework/value_t.h"
#include "gqten/gqtensor_all.h"


namespace gqten {


//// Tensor numerical functions.
//// Tensors contraction.
//template <typename TenElemType>
//void Contract(
    //const GQTensor<TenElemType> *, const GQTensor<TenElemType> *,
    //const std::vector<std::vector<long>> &,
    //GQTensor<TenElemType> *);

//// This API just for forward compatibility, it will be deleted soon.
//// TODO: Remove these API.
//inline DGQTensor *Contract(
    //const DGQTensor &ta, const DGQTensor &tb,
    //const std::vector<std::vector<long>> &axes_set) {
  //auto res_t = new DGQTensor();
  //Contract(&ta, &tb, axes_set, res_t);
  //return res_t;
//}

//inline ZGQTensor *Contract(
    //const ZGQTensor &ta, const ZGQTensor &tb,
    //const std::vector<std::vector<long>> &axes_set) {
  //auto res_t = new ZGQTensor();
  //Contract(&ta, &tb, axes_set, res_t);
  //return res_t;
//}

//inline ZGQTensor *Contract(
    //const DGQTensor &ta, const ZGQTensor &tb,
    //const std::vector<std::vector<long>> &axes_set) {
  //auto res_t = new ZGQTensor();
  //auto zta = ToComplex(ta);
  //Contract(&zta, &tb, axes_set, res_t);
  //return res_t;
//}

//inline ZGQTensor *Contract(
    //const ZGQTensor &ta, const DGQTensor &tb,
    //const std::vector<std::vector<long>> &axes_set) {
  //auto res_t = new ZGQTensor();
  //auto ztb = ToComplex(tb);
  //Contract(&ta, &ztb, axes_set, res_t);
  //return res_t;
//}


// Tensors linear combination.
// Do the operation: res += (coefs[0]*ts[0] + coefs[1]*ts[1] + ...).
//[> TODO: Support scalar (rank 0) tensor case. <]
template <typename TenElemType>
void LinearCombine(
    const std::vector<TenElemType> &,
    const std::vector<GQTensor<TenElemType> *> &,
    GQTensor<TenElemType> *);

template <typename TenElemType>
void LinearCombine(
    const std::size_t,
    const TenElemType *,
    const std::vector<GQTensor<TenElemType> *> &,
    GQTensor<TenElemType> *);

inline void LinearCombine(
    const std::size_t size,
    const double *dcoefs,
    const std::vector<GQTensor<GQTEN_Complex> *> &zts,
    GQTensor<GQTEN_Complex> *res) {
  auto zcoefs = new GQTEN_Complex [size];
  for (size_t i = 0; i < size; ++i) {
    zcoefs[i] = dcoefs[i];
  }
  LinearCombine(size, zcoefs, zts, res);
  delete [] zcoefs;
}


// Tensor SVD.
template <typename TenElemType>
void Svd(
    const GQTensor<TenElemType> *,
    const long, const long,
    const QN &, const QN &,
    const double, const long, const long,
    GQTensor<TenElemType> *,
    GQTensor<GQTEN_Double> *,
    GQTensor<TenElemType> *,
    double *, long *);


// These APIs just for forward compatibility, it will be deleted soon.
// TODO: Remove these APIs.
template <typename TenElemType>
struct SvdRes {
  SvdRes(
      GQTensor<TenElemType> *u,
      GQTensor<GQTEN_Double> *s,
      GQTensor<TenElemType> *v,
      const double trunc_err, const long D) :
      u(u), s(s), v(v), trunc_err(trunc_err), D(D) {}
  GQTensor<TenElemType> *u;
  GQTensor<GQTEN_Double> *s;
  GQTensor<TenElemType> *v;
  const double trunc_err;
  const long D;
};


template <typename TenElemType>
inline SvdRes<TenElemType> Svd(
    const GQTensor<TenElemType> &t,
    const long ldims, const long rdims,
    const QN &ldiv, const QN &rdiv,
    const double cutoff, const long Dmin, const long Dmax) {
  auto pu =  new GQTensor<TenElemType>();
  auto ps =  new GQTensor<GQTEN_Double>();
  auto pvt = new GQTensor<TenElemType>();
  double trunc_err;
  long D;
  Svd(
      &t,
      ldims, rdims,
      ldiv, rdiv,
      cutoff, Dmin, Dmax,
      pu, ps, pvt,
      &trunc_err, &D);
  return SvdRes<TenElemType>(pu, ps, pvt,trunc_err, D);
}


template <typename TenElemType>
inline SvdRes<TenElemType> Svd(
    const GQTensor<TenElemType> &t,
    const long ldims, const long rdims,
    const QN &ldiv, const QN &rdiv) {
  auto t_shape = t.shape;
  long lsize = 1;
  long rsize = 1;
  for (std::size_t i = 0; i < t_shape.size(); ++i) {
    if (i < ldims) {
      lsize *= t_shape[i];
    } else {
      rsize *= t_shape[i];
    }
  }
  auto D = ((lsize >= rsize) ? lsize : rsize);
  return Svd(
      t,
      ldims, rdims,
      ldiv, rdiv,
      0, D, D);
}


// Tensor expansion.
template <typename TenElemType>
void Expand(
    const GQTensor<TenElemType> *,
    const GQTensor<TenElemType> *,
    const std::vector<size_t>,
    GQTensor<TenElemType> *
);


// Index combiner generator.
template <typename TenElemType>
GQTensor<TenElemType> IndexCombine(
    const Index &,
    const Index &,
    const std::string &new_idx_dir = OUT
);


// Tensor transpose function multi-thread controller.
int GQTenGetTensorTransposeNumThreads(void);

void GQTenSetTensorTransposeNumThreads(const int);


// Timer.
class Timer {
public:
  Timer(const std::string &);

  void Restart(void);
  double Elapsed(void);
  double PrintElapsed(std::size_t precision = 5);

private:
  double start_;
  std::string notes_;

  double GetWallTime(void);
};
} /* gqten */
#endif /* ifndef GQTEN_GQTEN_H */
