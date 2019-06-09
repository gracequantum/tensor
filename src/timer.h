// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-29 21:05
* 
* Description: GraceQ/tensor project. Timer.
*/
#ifndef GQTEN_TIMER_H
#define GQTEN_TIMER_H

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


void Timer::PrintElapsed(std::size_t precision) {
  auto elapsed_time = Elapsed(); 
  std::cout << "[timing] "
            << notes_ << "\t"
            << std::setw(precision+3) << std::setprecision(precision)
            << elapsed_time << std::endl;
}


double Timer::GetWallTime(void) {
  struct timeval time;
  if (gettimeofday(&time, NULL)) { return 0; }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
} /* gqten */ 
#endif /* ifndef GQTEN_TIMER_H */
