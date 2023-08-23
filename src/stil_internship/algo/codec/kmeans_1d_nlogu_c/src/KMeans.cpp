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
    double dmin, dmax;
} dminmax;
dminmax d_min(double* x, long long N)
{
    double dmin = x[1] - x[0];
    double min_ = INFINITY;
    double max_ = -INFINITY;
    for (int i = 2; i < N; ++i)
    {
        dmin = std::min(dmin, x[i] - x[i-1]);
    }
    for (int i = 0; i < N; ++i)
    {
	    min_ = std::min(min_, x[i]); // i honestly don't know what dmax is supposed to be
        max_ = std::max(max_, x[i]);
    }
    return {dmin, max_ - min_};
}
long long bin_search(double* x, long long N, double max_rmse, double* borders, double* res_rmse, double* centers)
{
    dminmax dminmax = d_min(x, N);

    std::sort(x, x + N);

    std::vector<double> x_prefix_sum_sq(N+1);
    std::vector<double> x_prefix_sum(N+1);
    calculate_sum_x(x, N, x_prefix_sum_sq.data(), x_prefix_sum.data());

    //NOTE: the dmin/dmax bounds are really wrong.
    double taumin = 0;//dminmax.dmin / 2 * dminmax.dmin;
    double taumax = 100000;//dminmax.dmax / 2 * dminmax.dmax;//dminmax.dmax / 2 * dminmax.dmax; //this might be wrong, should check

    double se;
    double max_se = max_rmse * max_rmse * N;
    std::vector<double> F(N+1); //top min values
    std::vector<double> H(N + 1); //bottom min values
    std::vector<long long> J(N + 1); //top backtracking indices
    std::vector<long long> Jbottom(N+1); //bottom backtracking indices


    try_kmeans(N, x_prefix_sum.data(), x_prefix_sum_sq.data(), taumax, F.data(), H.data(), J.data(), Jbottom.data());
    size_t kMin = backtrack(N, J.data(), borders, x, centers);
    try_kmeans(N, x_prefix_sum.data(), x_prefix_sum_sq.data(), taumin, F.data(), H.data(), J.data(), Jbottom.data());
    size_t kMax = backtrack(N, J.data(), borders, x, centers);


    int cnt_iter = 0;

#ifdef _DEBUG
    std::cout<<"taumin: "<<taumin<<" taumax: "<<taumax<<std::endl;
#endif

    while ((taumax > taumin) && (taumax - taumin >= (taumax + taumin) / 2e10)) //should probably introduce epsilon?
    {
        cnt_iter++;
        try_kmeans(N, x_prefix_sum.data(), x_prefix_sum_sq.data(), (taumax + taumin) / 2, F.data(), H.data(), J.data(), Jbottom.data());


        size_t kOpt = backtrack(N, J.data(), borders, x, centers);
        se = F[N] - kOpt * (taumin + taumax) / 2;
        *res_rmse = std::sqrt(se / N);

        if (se > max_se)
        {
            kMin = kOpt;
            taumax = (taumin + taumax) / 2;
        }
        else if (se < max_se)
        {
            kMax = kOpt;
            taumin = (taumin + taumax) / 2;
        }
        //std::cout<<taumin<<" "<<taumax<<std::endl;
        else // (exact rmse match)
        {
        	
        	*res_rmse = std::sqrt((F[N] - kOpt * (taumin + taumax) / 2 )/ N);
        	return kOpt;
        }
    }
#ifdef _DEBUG
    std::cout<<"cnt_iter: "<<cnt_iter<<std::endl;
#endif
    
    size_t kOpt = backtrack(N, J.data(), borders, x, centers);
    *res_rmse = std::sqrt((F[N] - kOpt * (taumin + taumax) / 2 ) / N);
    return kOpt;

}
