#include <gtest/gtest.h>
#include "KMeans.h"
#include "metrics.h"

TEST(KMeansNLogU, FInTryKMeansWithTau) {
	std::vector<double> x = {1, 2, 3, 4, 5};
	std::vector<double> sum(x.size()), sum_sq(x.size());
	std::vector<double> F(x.size() + 1), H(x.size() + 1);
	std::vector<long long> J(x.size() + 1), Jbottom(x.size() + 1);
	calculate_sum_x(x.data(), x.size(), sum_sq.data(), sum.data());

	std::vector<double> Fres = {0, 0.8, 1.3, 2.1, 2.6, 3.4};

	try_kmeans(x.size(), sum.data(), sum_sq.data(), 0.8, F.data(), H.data(), J.data(), Jbottom.data());
	
	for (int i = 0; i <= x.size(); ++i)
	{
		EXPECT_NEAR(Fres[i], F[i], 1e-14);
	}
}

TEST(KMeansNLogU, JInTryKMeansWithTau) {
	std::vector<double> x = {1, 2, 3, 4, 5};
	std::vector<double> sum(x.size());
	std::vector<double> sum_sq(x.size());
	std::vector<double> F(x.size() + 1), H(x.size() + 1);
	std::vector<long long> J(x.size() + 1), Jbottom(x.size() + 1);
	calculate_sum_x(x.data(), x.size(), sum_sq.data(), sum.data());

	std::vector<long long> Jres = {0, 0, 0, 1, 2, 3};

	try_kmeans(x.size(), sum.data(), sum_sq.data(), 0.8, F.data(), H.data(), J.data(), Jbottom.data());
	
	EXPECT_EQ(J, Jres);
}

TEST(KMeansNLogU, Backtrack) {
	std::vector<double> x = {1, 2, 3, 4, 5};
	std::vector<double> sum(x.size());
	std::vector<double> sum_sq(x.size());
	std::vector<double> F(x.size() + 1), H(x.size() + 1), borders(x.size() + 1), centers(x.size() + 1);
	std::vector<long long> J(x.size() + 1), Jbottom(x.size() + 1);
	calculate_sum_x(x.data(), x.size(), sum_sq.data(), sum.data());

	std::vector<double> borders_res = {5, 3, 1};
	std::vector<double> centers_res = {4.5, 2.5, 1};

	try_kmeans(x.size(), sum.data(), sum_sq.data(), 0.8, F.data(), H.data(), J.data(), Jbottom.data());

	long long kOpt = backtrack(x.size(), J.data(), borders.data(), x.data(), centers.data());
	
	EXPECT_EQ(kOpt, borders_res.size());
	for (int i = 0; i < kOpt; ++i)
	{
		EXPECT_NEAR(borders_res[i], borders[i], 1e-14);
	}
	EXPECT_EQ(kOpt, centers_res.size());
	for (int i = 0; i < kOpt; ++i)
	{
		EXPECT_NEAR(centers_res[i], centers[i], 1e-14);
	}
}