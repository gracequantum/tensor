#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-09-07 16:58
#  
#  Description: GraceQ/tensor project. CMake module to find hptt library.
# 
find_path(hptt_INCLUDE_DIR hptt.h)

find_library(hptt_LIBRARY hptt)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(hptt DEFAULT_MSG hptt_LIBRARY hptt_INCLUDE_DIR)
mark_as_advanced(hptt_INCLUDE_DIR hptt_LIBRARY)


set(hptt_LIBRARIES ${hptt_LIBRARY})
set(hptt_INCLUDE_DIRS ${hptt_INCLUDE_DIR})
