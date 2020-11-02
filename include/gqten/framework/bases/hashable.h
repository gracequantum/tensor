// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-10-28 10:33
*
* Description: GraceQ/tensor project. Abstract base class for hashable object.
*/

/**
@file hashable.h
@brief Abstract base class for hashable object.
*/
#ifndef GQTEN_FRAMEWORK_BASES_HASHABLE_H
#define GQTEN_FRAMEWORK_BASES_HASHABLE_H


#include <stddef.h>     // size_t


namespace gqten {


/**
Abstract base class for hashable object.
*/
class Hashable {
public:
  Hashable(void) = default;
  virtual ~Hashable(void) = default;

  /// Return the hash value of the object.
  virtual size_t Hash(void) const = 0;

  /**
  Basic equal to operator overload.

  @param rhs Another Hashable object.

  @return Comparison result.
  */
  virtual bool operator==(const Hashable &rhs) const {
    return Hash() == rhs.Hash();
  }

  /**
  Basic not equal to operator overload.

  @param rhs Another Hashable object.

  @return Comparison result.
  */
  virtual bool operator!=(const Hashable &rhs) const { return !(*this == rhs); }
};


/**
Hash function for Hashable object.

@param obj Hashable object.
*/
inline size_t Hash(const Hashable &obj) { return obj.Hash(); }
} /* gqten */
#endif /* ifndef GQTEN_FRAMEWORK_BASES_HASHABLE_H */
