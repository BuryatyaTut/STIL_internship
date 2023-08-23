#include <gtest/gtest.h>
#include "metrics.h"

TEST(KMeansNLogU, CalculateSum) {
	std::vector<double> x = {3, 5, 10, 26, 43, 90};

	std::vector<double> sum(x.size());
	std::vector<double> sum_sq(x.size());
	calculate_sum_x(x.data(), x.size(), sum_sq.data(), sum.data());

	std::vector<double> res_sq = {9, 34, 134, 810, 2659, 10759};
	std::vector<double> res = {3, 8, 18, 44, 87, 177};
	EXPECT_EQ(sum_sq, res_sq);
	EXPECT_EQ(sum, res);
}

TEST(KMeansNLogU, SSQ)
{
	std::vector<double> x = {3, 26, 43, 90};
	std::vector<double> sum(x.size());
	std::vector<double> sum_sq(x.size());
	calculate_sum_x(x.data(), x.size(), sum_sq.data(), sum.data());

	std::vector<std::vector<double>> res_expected = {
		{INFINITY, INFINITY, INFINITY, INFINITY, INFINITY},
		{0, INFINITY, INFINITY, INFINITY, INFINITY},
		{264.5, 0, INFINITY, INFINITY, INFINITY},
		{806, 144.5, 0, INFINITY, INFINITY},
		{4073, 2198, 1104.5, 0, INFINITY},
	};
	std::vector res(x.size() + 1, std::vector<double>(x.size() + 1));
	for (int i = 0; i <= x.size(); ++i)
	{
		for (int j = 0; j <= x.size(); ++j)
		{
			res[i][j] = w(j, i, sum.data(), sum_sq.data(), 0, x.size());
		}
	}
	EXPECT_EQ(res, res_expected);
}

TEST(KMeansNLogU, SSQWithTau)
{
	std::vector<double> x = {3, 26, 43, 90};
	std::vector<double> sum(x.size());
	std::vector<double> sum_sq(x.size());
	calculate_sum_x(x.data(), x.size(), sum_sq.data(), sum.data());
	double tau = 10;

	std::vector<std::vector<double>> res_expected = {
		{INFINITY, INFINITY, INFINITY, INFINITY, INFINITY},
		{10, INFINITY, INFINITY, INFINITY, INFINITY},
		{274.5, 10, INFINITY, INFINITY, INFINITY},
		{816, 154.5, 10, INFINITY, INFINITY},
		{4083, 2208, 1114.5, 10, INFINITY},
	};
	std::vector res(x.size() + 1, std::vector<double>(x.size() + 1));
	for (int i = 0; i <= x.size(); ++i)
	{
		for (int j = 0; j <= x.size(); ++j)
		{
			res[i][j] = w(j, i, sum.data(), sum_sq.data(), tau, x.size());
		}
	}
	EXPECT_EQ(res, res_expected);
}
TEST(KMeansNLogU, SSQUnbounded)
{
	std::vector<double> x = {3, 26, 43, 90};
	std::vector<double> sum(x.size());
	std::vector<double> sum_sq(x.size());
	calculate_sum_x(x.data(), x.size(), sum_sq.data(), sum.data());

	std::vector<std::vector<double>> res_expected = {
		{INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY},
		{0, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY},
		{264.5, 0, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY},
		{806, 144.5, 0, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY},
		{4073, 2198, 1104.5, 0, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY},
		{INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY},
		{INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY},
		{INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY},
		{INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY},
		{INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY},
	};
	std::vector res((x.size() + 1) * 2, std::vector<double>((x.size() + 1) * 2));
	for (int i = 0; i < (x.size() + 1) * 2; ++i)
	{
		for (int j = 0; j < (x.size() + 1) * 2; ++j)
		{
			res[i][j] = w(j, i, sum.data(), sum_sq.data(), 0, x.size());
		}
	}
	EXPECT_EQ(res, res_expected);
}