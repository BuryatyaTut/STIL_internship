#include <algorithm>
#include "KMeans.h"

#include <iostream>
#include <numeric>
#include <vector>

#include "SMAWK.h"
#include "metrics.h"
long long backtrack(long long N, long long* J, double* borders, double* x, double* centers)
{
    long long cluster_idx = 0, j_idx = N;
    borders[0] = x[N - 1];
    while (j_idx > 0)
    {
        centers[cluster_idx] = std::accumulate(x + J[j_idx], x + j_idx, 0.0 ) / (j_idx - J[j_idx] );
		borders[++cluster_idx] = x[J[j_idx] - 1];
        j_idx = J[j_idx];
    }
    return cluster_idx;
}
long long backtrack_justK(long long N, long long* J)
{
	long long cluster_idx = 0, j_idx = N;
    while (j_idx > 0)
    {
        ++cluster_idx;
        j_idx = J[j_idx];
    }
    return cluster_idx;
}
void try_kmeans(long long N,double* x_prefix_sum, double* x_prefix_sum_sq, double tau, double* F, double* H, long long* J, long long* Jbottom)
{
    F[0] = 0;
    long long c = 0, r = 0, p, j0;
    while (c < N)
    {
        p = std::min(2 * c - r + 1, N);
        j0 = p + 1;
        run_SMAWK(F, J, F, r, c, c + 1, p, tau, x_prefix_sum, x_prefix_sum_sq, N + 1); //run two SMAWKs on both top and bottom squares
        run_SMAWK(H, Jbottom, F, c + 1, p - 1, c + 2, p, tau, x_prefix_sum, x_prefix_sum_sq, N + 1);
        for (long long j = c + 2; j <= p; ++j)
        {
            if (H[j] < F[j])
            {
                j0 = j;
                break;
            }
        }

        if (j0 == p + 1)
        {
            //no minima in the bottom square, continue as is
            c = p;
        }
        else
        {
            //at least one minimum is in the bottom square, apply magic to exclude some top rows from further consideration
            F[j0] = H[j0];
            J[j0] = Jbottom[j0];

            r = c + 1; //move the new smawk square further down 
            c = j0;    //(i.e. don't start it from the 0th row)
        }


    }
}
typedef struct dminmax
{
    double dmin, taumax;
} dminmax;
dminmax d_min(double* x, long long N, double* sum_x, double* sum_x_sq)
{
    double dmin = x[1] - x[0];
    double min_ = INFINITY;
    double max_ = -INFINITY;
    double taumax = INFINITY;
    for (int i = 2; i < N; ++i)
    {
        dmin = std::min(dmin, x[i] - x[i-1]);
    }
    double opt2 = INFINITY;
    for (int i = 1; i < N; ++i)
    {
	    opt2 = std::min(opt2, w(0, i, sum_x, sum_x_sq, 0, N) + w(i, N, sum_x, sum_x_sq, 0, N));
    }
    taumax = w(0, N, sum_x, sum_x_sq, 0, N) - opt2;
    
    return {dmin, taumax};
}
long long KMeans(double* x, long long N, double max_rmse, double* borders, double* res_rmse, double* centers)
{
	std::sort(x, x + N);

    std::vector<double> x_prefix_sum_sq(N+1);
    std::vector<double> x_prefix_sum(N+1);

    calculate_sum_x(x, N, x_prefix_sum_sq.data(), x_prefix_sum.data());
    dminmax bounds = d_min(x, N, x_prefix_sum.data(), x_prefix_sum_sq.data());
    //NOTE: the dmin/dmax bounds are really wrong.
    //The bounds depend __heavily__ on data and either
    //limit the potential values of K if the bounds are too tight,
    //or cause numerical instability when they're too loose.
    double taumin = bounds.dmin / 2 * bounds.dmin * 0.99;
    double taumax = bounds.taumax * 1.01;//this might be wrong, should check
        
    double max_se = max_rmse * max_rmse * N;
    
    //dminmax dminmax = d_min(x, N);

    std::vector<double> F(N + 1, INFINITY); //top min values
    std::vector<double> H(N + 1, INFINITY); //bottom min values
    std::vector<long long> J(N + 1); //top backtracking indices
    std::vector<long long> Jbottom(N + 1); //bottom backtracking indices

    long long cnt_iter = 0;
    size_t kOpt;
    double epsilon = 1e-10;
    double taures = bin_search(
        N, 
        max_se, 
        taumin, 
        taumax, 
        epsilon, 
        F, 
        H, 
        J, 
        Jbottom, 
        x_prefix_sum, 
        x_prefix_sum_sq);
    kOpt = backtrack(N, J.data(), borders, x, centers);
    *res_rmse = std::sqrt((F[N] - kOpt * taures) / N);
    return kOpt;
}
double bin_search(
    long long N, 
    double max_se, 
    double taumin, 
    double taumax,
    double epsilon,
    std::vector<double> &F,
    std::vector<double> &H,
    std::vector<long long> &J,
    std::vector<long long> &Jbottom,
    std::vector<double> &x_prefix_sum,
    std::vector<double> &x_prefix_sum_sq
)
{
    double taumin_l = taumin, taumax_l = taumax; //local copies
    long long cnt_iter = 0, kOpt;
    double taucheck, se;
    while (taumax_l - taumin_l >= (taumax_l + taumin_l) * epsilon)
    {
        cnt_iter++;
        taucheck = (taumax_l + taumin_l) / 2;
        try_kmeans(
            N, 
            x_prefix_sum.data(), 
            x_prefix_sum_sq.data(), 
            taucheck, 
            F.data(), 
            H.data(), 
            J.data(), 
            Jbottom.data()
        );

    	kOpt = backtrack_justK(N, J.data());
        se = F[N] - kOpt * taucheck;
        

        if (se > max_se)
        {
            taumax_l = taucheck;
        }
        else if (se < max_se)
        {
            taumin_l = taucheck;
        }
        else // (exact rmse match)
        {
        	return taucheck;
        }
    }
    
    return taucheck;

}
