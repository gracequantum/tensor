// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-08 16:58
* 
* Description: GraceQ/tensor project. Implementation details about timer.
*/
#include "gqten/gqten.h"

#include <iostream>
#include <string>
#include <iomanip>

#include <time.h>
#include <sys/time.h>


namespace gqten {


Timer::Timer(const std::string &notes) : notes_(notes), start_(0) {}


void Timer::Restart(void) { start_ = GetWallTime(); }


double Timer::Elapsed(void) { return GetWallTime() - start_; }


double Timer::PrintElapsed(std::size_t precision) {
  auto elapsed_time = Elapsed(); 
  std::cout << "[timing] "
            << notes_ << "\t"
            << std::setw(precision+3) << std::setprecision(precision) << std::fixed
            << elapsed_time << std::endl;
  return elapsed_time;
}


double Timer::GetWallTime(void) {
  struct timeval time;
  if (gettimeofday(&time, NULL)) { return 0; }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
} /* gqten */ 
