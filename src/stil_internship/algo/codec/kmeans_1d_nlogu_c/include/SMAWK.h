#pragma once
void run_SMAWK(
	double* F, 
	long long* J, 
	double* F_top, 
	long long x0, 
	long long x1, 
	long long y0, 
	long long y1, 
	double tau, 
	double* x_prefix_sum, 
	double* x_prefix_sum_sq, 
	size_t N
);
void find_min_from_candidates
(
	long long imin, 
	long long imax, 
	long long istep,
	const std::vector<long long> & js,
	double* sum_x,
	double* sum_x_sq,
	double tau,
	double* F,
	long long* J,
	double* F_top,
	size_t N
);

void fill_even_positions
(
	long long imin, 
	long long imax, 
	long long istep,
	const std::vector<long long> & js,
	double* sum_x,
	double* sum_x_sq,
	double tau,
	double* F,
	long long* J,
	double* F_top,
	size_t N
);