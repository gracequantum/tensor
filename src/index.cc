// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-07 19:40
* 
* Description: GraceQ/tensor project. Implementation details about tensor index.
*/
#include "gqten/gqten.h"

#include <vector>
#include <iostream>
#include <fstream>


namespace gqten {


std::hash<std::string> str_hasher_;


size_t Index::Hash(void) const {
  return QNSectorSet::Hash() ^ str_hasher_(tag);
}


InterOffsetQnsct Index::CoorInterOffsetAndQnsct(const long coor) const {
  long inter_offset = 0;
  for (auto &qnsct : qnscts) {
    long temp_inter_offset = inter_offset + qnsct.dim;
    if (temp_inter_offset > coor) {
      return InterOffsetQnsct(inter_offset, qnsct);
    } else if (temp_inter_offset <= coor) {
      inter_offset = temp_inter_offset;
    }
  }
}


Index InverseIndex(const Index &idx) {
  Index inversed_idx = idx;
  inversed_idx.Dag();
  return inversed_idx;
}


std::ifstream &bfread(std::ifstream &ifs, Index &idx) {
  long qnscts_num;
  ifs >> qnscts_num;
  idx.qnscts = std::vector<QNSector>(qnscts_num);
  for (auto &qnsct : idx.qnscts) { bfread(ifs, qnsct); }
  ifs >> idx.dim >> idx.dir;
  // Deal with empty tag, where will be '\n\n'.
  char next1_ch, next2_ch;
  ifs.get(next1_ch);
  ifs.get(next2_ch);
  if (next2_ch != '\n') {
    ifs.putback(next2_ch);
    ifs.putback(next1_ch);
    ifs >> idx.tag;
  }
  return ifs; 
}


std::ofstream &bfwrite(std::ofstream &ofs, const Index &idx) {
  long qnscts_num = idx.qnscts.size();
  ofs << qnscts_num << std::endl;
  for (auto &qnsct : idx.qnscts) { bfwrite(ofs, qnsct); }
  ofs << idx.dim << std::endl;
  ofs << idx.dir << std::endl;
  ofs << idx.tag << std::endl;
  return ofs;
}
} /* gqten */ 
