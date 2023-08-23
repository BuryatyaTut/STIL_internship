#pragma once
void run_SMAWK(double* F, 
	long long* J, 
	double* F_top, 
	long long x0, 
	long long x1, 
	long long y0, 
	long long y1, 
	double tau, 
	double* x_prefix_sum, 
	double* x_prefix_sum_sq, 
	size_t N);