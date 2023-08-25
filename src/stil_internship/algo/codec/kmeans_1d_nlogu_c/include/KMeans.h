#pragma once
#include <vector>
double bin_search( 
	long long N, 
	double max_rmse, 
	double taumin, 
	double taumax,
	double epsilon,
	std::vector<double> &F,
	std::vector<double> &H,
	std::vector<long long> &J,
	std::vector<long long> &JBottom,
    std::vector<double> &x_prefix_sum,
    std::vector<double> &x_prefix_sum_sq
);

long long KMeans(double* x, long long N, double max_rmse, double* borders, double* res_rmse, double* centers);

void try_kmeans(long long N,double* x_prefix_sum, double* x_prefix_sum_sq, double tau, double* F, double* H, long long* J, long long* Jbottom);

long long backtrack(long long N, long long* J, double* borders, double* x, double* centers);