#include <random>
#include <gtest/gtest.h>

#include "metrics.h"
#include "SMAWK.h"

TEST(KMeansNLogU, WorkingSMAWK) {
	//setup random coordinates and generate x
	
	std::random_device dev;
	std::mt19937_64 gen(dev());
	std::uniform_real_distribution val_dis(-1e10, 1e10);
	std::uniform_int_distribution<size_t> x_size_dis(1, 10);
	size_t N = x_size_dis(gen);
	std::vector<double> x(N);
	std::uniform_int_distribution corner_idx_dis(0ULL, N - 1);
	size_t x1 = corner_idx_dis(gen);
	std::uniform_int_distribution top_idx_dis(0ULL, x1);
	size_t x0 = top_idx_dis(gen), y0 = std::min(x1 + 1, N), y1 = std::min(y0 + (x1 - x0), N);

	std::ranges::generate(x.begin(), x.end(), [&] {return val_dis(gen);});
	std::sort(x.begin(), x.end());

	std::vector<double> sum(x.size()), sum_sq(x.size());
	std::vector<double> F(x.size() + 1, INFINITY), H(x.size() + 1, INFINITY), F_expected(x.size() + 1, INFINITY);
	std::vector<long long> J(x.size() + 1);
	calculate_sum_x(x.data(), x.size(), sum_sq.data(), sum.data());
	
	F[0] = 0;
	F_expected[0] = 0;
	//calculate needed F's for proper calc of SMAWK
	for (size_t y = 1; y < y0; ++y)
	{
		double min_ = INFINITY;
		for (size_t x = 0; x < y; ++x)
		{
			min_ = std::min(min_, F[x] + w(x, y, sum.data(), sum_sq.data(), 0, N));
		}
		F_expected[y] = min_;
		F[y] = min_;
	}

	
	//run test
	run_SMAWK(F.data(), J.data(), F.data(), x0, x1, y0, y1, 0, sum.data(), sum_sq.data(), N);

	//calculate minima that should have been in the SMAWKed square
	for (size_t y = y0; y <= std::min(N, y1); ++y)
	{
		double min_ = INFINITY;
		for (size_t x = x0; x <= x1; ++x)
		{
			min_ = std::min(min_, F_expected[x] + w(x, y, sum.data(), sum_sq.data(), 0, N));
		}
		F_expected[y] = min_;
	}

	//output things
	std::stringstream ss;
	for (auto& i : x)
	{
		ss << i<<" ";
	}
	
	EXPECT_EQ(F_expected, F) << "x0 x1 y0 y1 " << x0 << " " << x1 << " " << y0 << " " << y1 << std::endl << "x: " << ss.str();
	
}

TEST(KMeansNLogU, FindMinFromCandidates)
{
	std::vector<double> x = {1, 2, 3, 4, 5};
	std::vector<double> sum(x.size()), sum_sq(x.size());
	std::vector<double> F(x.size() + 1, INFINITY), H(x.size() + 1, INFINITY);
	std::vector<long long> J(x.size() + 1, LLONG_MAX);
	calculate_sum_x(x.data(), x.size(), sum_sq.data(), sum.data());

	std::vector<long long> js = {0, 1, 2, 3};
	F[0] = 0;
	F[1] = 0.8;
	F[2] = 1.3;
	F[3] = 2.1;
	find_min_from_candidates(4, 4, 1, js, sum.data(), sum_sq.data(), 0.8, F.data(), J.data(), F.data(), x.size());
	EXPECT_DOUBLE_EQ(F[4], 2.6);
	EXPECT_EQ(J[4], 2);
}
// this test is broken (written incorrectly)
//
//TEST(KMeansNLogU, FillEvenPositions)
//{
//	std::vector<double> x = {1, 2, 3, 4, 5};
//	std::vector<double> sum(x.size()), sum_sq(x.size());
//	std::vector<double> F(x.size() + 1, INFINITY), H(x.size() + 1, INFINITY);
//	std::vector J(x.size() + 1, LLONG_MAX);
//	calculate_sum_x(x.data(), x.size(), sum_sq.data(), sum.data());
//	
//	std::vector<long long> js = {3, 4, 5};
//	F[0] = 0;
//	F[1] = 0.8;
//	F[2] = 1.3;
//	F[4] = 2.6;
//
//	J[0] = 0;
//	J[1] = 0;
//	J[2] = 0;
//	J[4] = 2;
//	fill_even_positions(0, 2, 1, js, sum.data(), sum_sq.data(), 0.8, F.data(), J.data(), F.data(), x.size());
//	EXPECT_DOUBLE_EQ(F[3], 2.1);
//	EXPECT_EQ(J[3], 1);
//	EXPECT_DOUBLE_EQ(F[5], 3.4);
//	EXPECT_EQ(J[5], 3);
//}