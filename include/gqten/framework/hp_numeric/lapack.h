// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-12-02 14:15
* 
* Description: GraceQ/tensor project. High performance LAPACK related functions
* based on MKL.
*/

/**
@file lapack.h
@brief High performance LAPACK related functions based on MKL.
*/
#ifndef GQTEN_FRAMEWORK_HP_NUMERIC_LAPACK_H
#define GQTEN_FRAMEWORK_HP_NUMERIC_LAPACK_H


#include "gqten/framework/value_t.h"

#include <algorithm>    // min
#include <cstring>      // memcpy, memset

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert

#include "mkl.h"


namespace gqten {


namespace hp_numeric {


inline void MatSVD(
    GQTEN_Double *mat,
    const size_t m, const size_t n,
    GQTEN_Double* &u,
    GQTEN_Double* &s,
    GQTEN_Double* &vt
) {
  auto lda = n;
  size_t ldu, ldvt;
  if (m >= n) {
    ldu = n;
    ldvt = n;
    s = (GQTEN_Double *) malloc(n * sizeof(GQTEN_Double));
    vt = (GQTEN_Double *) malloc((ldvt * n) * sizeof(GQTEN_Double));
  } else {
    ldu = m;
    ldvt = n;
    s = (GQTEN_Double *) malloc(m * sizeof(GQTEN_Double));
    vt = (GQTEN_Double *) malloc((ldvt * m) * sizeof(GQTEN_Double));
  }
  u = (GQTEN_Double *) malloc((ldu * m) * sizeof(GQTEN_Double));
  auto info = LAPACKE_dgesdd(
                  LAPACK_ROW_MAJOR, 'S',
                  m, n,
                  mat, lda,
                  s,
                  u, ldu,
                  vt, ldvt
              );
  assert(info == 0);
}


inline void MatSVD(
    GQTEN_Complex *mat,
    const size_t m, const size_t n,
    GQTEN_Complex* &u,
    GQTEN_Double*  &s,
    GQTEN_Complex* &vt
) {
  auto lda = n;
  size_t ldu, ldvt;
  if (m >= n) {
    ldu = n;
    ldvt = n;
    s = (GQTEN_Double *) malloc(n * sizeof(GQTEN_Double));
    vt = (GQTEN_Complex *) malloc((ldvt * n) * sizeof(GQTEN_Complex));
  } else {
    ldu = m;
    ldvt = n;
    s = (GQTEN_Double *) malloc(m * sizeof(GQTEN_Double));
    vt = (GQTEN_Complex *) malloc((ldvt * m) * sizeof(GQTEN_Complex));
  }
  u = (GQTEN_Complex *) malloc((ldu * m) * sizeof(GQTEN_Complex));
  auto info = LAPACKE_zgesdd(
                  LAPACK_ROW_MAJOR, 'S',
                  m, n,
                  mat, lda,
                  s,
                  u, ldu,
                  vt, ldvt
              );
  assert(info == 0);
}


inline void MatQR(
    GQTEN_Double *mat,
    const size_t m, const size_t n,
    GQTEN_Double* &q,
    GQTEN_Double* &r
) {
  auto k = std::min(m, n);
  auto tau = (GQTEN_Double *) malloc(k * sizeof(GQTEN_Double));
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, mat, n, tau);

  // Create R matrix
  r = (GQTEN_Double *) malloc((k * n) * sizeof(GQTEN_Double));
  for (size_t i = 0; i < k; ++i) {
    memset(r + i*n, 0, i * sizeof(GQTEN_Double));
    memcpy(r + i*n + i, mat + i*n + i, (n - i) * sizeof(GQTEN_Double));
  }

  // Create Q matrix
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, k, k, mat, n, tau);
  free(tau);
  q = (GQTEN_Double *) malloc((m * k) * sizeof(GQTEN_Double));
  if (m == n) {
    memcpy(q, mat, (m*n) * sizeof(GQTEN_Double));
  } else {
    for (size_t i = 0; i < m; ++i) {
      memcpy(q + i*k, mat + i*n, k * sizeof(GQTEN_Double));
    }
  }
}
} /* hp_numeric */
} /* gqten */
#endif /* ifndef GQTEN_FRAMEWORK_HP_NUMERIC_LAPACK_H */


//void qr( double* const _Q, double* const _R, double* const _A, const size_t _m, const size_t _n) {
    //// Maximal rank is used by Lapacke
    //const size_t rank = std::min(_m, _n); 

    //// Tmp Array for Lapacke
    //const std::unique_ptr<double[]> tau(new double[rank]);

    //// Calculate QR factorisations
    //LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, (int) _m, (int) _n, _A, (int) _n, tau.get());

    //// Copy the upper triangular Matrix R (rank x _n) into position
    //for(size_t row =0; row < rank; ++row) {
        //memset(_R+row*_n, 0, row*sizeof(double)); // Set starting zeros
        //memcpy(_R+row*_n+row, _A+row*_n+row, (_n-row)*sizeof(double)); // Copy upper triangular part from Lapack result.
    //}

    //// Create orthogonal matrix Q (in tmpA)
    //LAPACKE_dorgqr(LAPACK_ROW_MAJOR, (int) _m, (int) rank, (int) rank, _A, (int) _n, tau.get());

    ////Copy Q (_m x rank) into position
    //if(_m == _n) {
        //memcpy(_Q, _A, sizeof(double)*(_m*_n));
    //} else {
        //for(size_t row =0; row < _m; ++row) {
            //memcpy(_Q+row*rank, _A+row*_n, sizeof(double)*(rank));
        //}
    //}
//}
