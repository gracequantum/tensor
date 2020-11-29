// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-27 10:46
*
* Description: GraceQ/tensor project. Abstract base class for asynchronous
* numerical calculations.
*/

/**
@file executor.h
@brief Abstract base class for asynchronous numerical calculations.
*/
#ifndef GQTEN_FRAMEWORK_BASES_EXECUTOR_H
#define GQTEN_FRAMEWORK_BASES_EXECUTOR_H


namespace gqten {


/// Possible status for an executor.
enum ExecutorStatus {
  UNINITED,     ///< Uninitialized.
  INITED,       ///< Initialized.
  EXEING,       ///< Executing.
  FINISH,       ///< Execution finished.
};


/**
Abstract base class for asynchronous numerical calculations.

@note The user to inherit and use this base class should also manipulate the
      status of the executor.
*/
class Executor {
public:
  Executor(void) = default;
  virtual ~Executor(void) = default;

  /// Execute the actual heavy numerical calculation.
  virtual void Execute(void) = 0;

  /// Get the status of the executor.
  ExecutorStatus GetStatus(void) const { return status_; }

  /// Set the status of the executor.
  void SetStatus(const ExecutorStatus new_status) { status_ = new_status; }

private:
  /// The status of the executor.
  ExecutorStatus status_ = ExecutorStatus::UNINITED;
};
} /* gqten */
#endif /* ifndef GQTEN_FRAMEWORK_BASES_EXECUTOR_H */
