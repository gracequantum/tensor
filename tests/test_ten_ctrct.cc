// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-10 19:39
* 
* Description: GraceQ/tensor project. Unittests for tensor contraction functions.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"

#include <cmath>

#include "mkl.h"


using namespace gqten;


const double kEpsilon = 1.0E-12;


inline double drand(void) { return double(rand()) / RAND_MAX; }


struct TestContraction : public testing::Test {
  long d = 3;
  Index idx = Index({
      QNSector(QN({QNNameVal("Sz", -1)}), d),
      QNSector(QN({QNNameVal("Sz",  0)}), d),
      QNSector(QN({QNNameVal("Sz",  1)}), d)});
  long idx_size = 3 * d;
};


TEST_F(TestContraction, 1DCase) {
  auto ten = GQTensor({idx});
  srand(0);
  auto dense_ten = new double [idx_size];
  for (long i = 0; i < idx_size; ++i) {
    dense_ten[i] = drand();
    ten({i}) = dense_ten[i];
  }
  GQTensor res;
  Contract(&ten, &ten, {{0}, {0}}, &res);
  double res0 = 0;
  for (long i = 0; i < idx_size; ++i) { res0 += std::pow(dense_ten[i], 2.0); }
  EXPECT_NEAR(res.scalar, res0, kEpsilon);
  delete [] dense_ten;
}


TEST_F(TestContraction, 2DCase) {
  auto ten = GQTensor({idx, idx});
  srand(0);
  auto dense_ten = new double [idx_size * idx_size];
  for (long i = 0; i < idx_size; ++i) {
    for (long j = 0; j < idx_size; ++j) {
      dense_ten[i*idx_size + j] = drand();
      ten({i, j}) = dense_ten[i*idx_size + j];
    }
  }
  GQTensor res;
  Contract(&ten, &ten, {{1}, {0}}, &res);
  auto res0 = new double [idx_size * idx_size];
  cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      idx_size, idx_size, idx_size,
      1.0,
      dense_ten, idx_size,
      dense_ten, idx_size,
      0.0,
      res0, idx_size);
  for (long i = 0; i < idx_size; ++i) {
    for (long j = 0; j < idx_size; ++j) {
      EXPECT_NEAR(res.Elem({i, j}), res0[i*idx_size + j], kEpsilon);
    }
  }
  delete [] dense_ten;
  delete [] res0;
}


TEST_F(TestContraction, 3DCase) {
  auto ten = GQTensor({idx, idx, idx});
  srand(0);
  auto dense_ten = new double [idx_size * idx_size * idx_size];
  for (long i = 0; i < idx_size; ++i) {
    for (long j = 0; j < idx_size; ++j) {
      for (long k = 0; k < idx_size; ++k) {
        auto rand_elem = drand();
        ten({i, j, k}) = rand_elem;
        dense_ten[i*(idx_size*idx_size) + j*idx_size + k] = rand_elem;
      }
    }
  }
  GQTensor res;
  Contract(&ten, &ten, {{2}, {0}}, &res);
  auto res0 = new double [idx_size * idx_size * idx_size * idx_size];
  cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      idx_size*idx_size, idx_size*idx_size, idx_size,
      1.0,
      dense_ten, idx_size,
      dense_ten, idx_size*idx_size,
      0.0,
      res0, idx_size*idx_size);
  for (long i = 0; i < idx_size; ++i) {
    for (long j = 0; j < idx_size; ++j) {
      for (long k = 0; k < idx_size; ++k) {
        for (long l = 0; l < idx_size; ++l) {
          EXPECT_NEAR(
              res.Elem({i, j, k, l}),
              res0[(i*idx_size + j)*(idx_size*idx_size) + (k*idx_size + l)],
              kEpsilon);
        }
      }
    }
  }
  Contract(&ten, &ten, {{1, 2}, {0, 1}}, &res);
  delete [] res0;
  res0 = new double [idx_size * idx_size];
  cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      idx_size, idx_size, idx_size*idx_size,
      1.0,
      dense_ten, idx_size*idx_size,
      dense_ten, idx_size,
      0.0,
      res0, idx_size);
  for (long i = 0; i < idx_size; ++i) {
    for (long j = 0; j < idx_size; ++j) {
      EXPECT_NEAR(res.Elem({i, j}), res0[i*idx_size + j], kEpsilon);
    }
  }
  Contract(&ten, &ten, {{0, 1, 2}, {0, 1, 2}}, &res);
  delete [] res0;
  res0 = new double [1];
  cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      1, 1, idx_size*idx_size*idx_size,
      1.0,
      dense_ten, idx_size*idx_size*idx_size,
      dense_ten, 1,
      0.0,
      res0, 1);
  EXPECT_NEAR(res.scalar, *res0, kEpsilon);
  delete [] dense_ten;
  delete [] res0;
}
