// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-29 16:41
* 
* Description: GraceQ/tensor project. Testing utilities.
*/
#ifndef GQTEN_TESTING_UTILS_H
#define GQTEN_TESTING_UTILS_H


#include <vector>



inline std::vector<long> TransCoors(
    const std::vector<long> &old_coors, const std::vector<long> &axes_map) {
  std::vector<long> new_coors(old_coors.size());
  for (std::size_t i = 0; i < axes_map.size(); ++i) {
    new_coors[i] = old_coors[axes_map[i]];
  }
  return new_coors;
}


#endif /* ifndef GQTEN_TESTING_UTILS_H */
