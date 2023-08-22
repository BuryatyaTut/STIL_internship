#include <cmath>
void calculate_sum_x(double* x, long long N, double* sum_x_sq, double* sum_x)
{
	sum_x_sq[0] = x[0] * x[0];
	sum_x[0] = x[0];
	for (long long i = 1; i <= N; ++i)
	{
		sum_x_sq[i] = sum_x_sq[i-1] + x[i] * x[i];
		sum_x[i] = sum_x[i-1] + x[i];
	}
}
double w(
  const size_t j, const size_t i,
  const double* sum_x, // running sum of xi
  const double* sum_x_sq, // running sum of xi^2
  double tau,
  const size_t N
)
{
  double sji(0.0);

  if(j >= i 
	  || i > N) //this version of algorithm tries to access elements beyond the matrix quite frequently
  {
	sji = INFINITY;
  } else if(j > 0) {
    double muji = (sum_x[i - 1] - sum_x[j-1]) / (i - j); //not sure if correct application of tau
    sji = sum_x_sq[i - 1] - sum_x_sq[j-1] - (i - j) * muji * muji + tau;
  } else {
    sji = sum_x_sq[i - 1] - sum_x[i - 1] * sum_x[i - 1] / (i) + tau;
  }

  sji = (sji < 0) ? 0 : sji;
    
  return sji;
}
