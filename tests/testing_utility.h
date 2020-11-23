// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-10-30 14:40
*
* Description: GraceQ/tensor project. Utilities for unit tests.
*/
#ifndef TESTS_TESTING_UTILITY_H
#define TESTS_TESTING_UTILITY_H


#include "gqten/framework/value_t.h"    // CoorsT


#include <random>


template <typename IntT>
inline IntT RandomInteger(const IntT min, const IntT max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<IntT> distrib(min, max);
  return distrib(gen);
}


inline int RandInt(const int min, const int max) {
  return RandomInteger(min, max);
}


inline size_t RandUnsignedInt(const size_t min, const size_t max) {
  return RandomInteger(min, max);
}


inline size_t RandUnsignedInt(const size_t max) {
  return RandUnsignedInt(0, max);
}


inline gqten::CoorsT TransCoors(
    const gqten::CoorsT &old_coors, const std::vector<size_t> &axes_map) {
  gqten::CoorsT new_coors(old_coors.size());
  for (size_t i = 0; i < axes_map.size(); ++i) {
    new_coors[i] = old_coors[axes_map[i]];
  }
  return new_coors;
}
#endif /* ifndef TESTS_TESTING_UTILITY_H */
