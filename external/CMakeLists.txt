#  SPDX-License-Identifier: LGPL-3.0-only
#
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-05-18 14:44
#
#  Description: GraceQ/tensor project. CMake file to control external dependencies.
#


# hptt
if(GQTEN_USE_EXTERNAL_HPTT_LIB)
  find_package(hptt REQUIRED)
else()
  add_subdirectory(hptt)
endif()
