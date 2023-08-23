#pragma once
void calculate_sum_x(double* x, long long N, double* sum_x_sq, double* sum_x);

double w(size_t j, size_t i,
  const double* sum_x,
  const double* sum_x_sq,
  double tau,
  size_t N);
