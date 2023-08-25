#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Python.h>
#include "KMeans.h"
namespace py = pybind11;
void kmeans_nlogu(py::array_t<double, py::array::c_style | py::array::forcecast> x, 
	py::array_t<double, py::array::c_style | py::array::forcecast> max_rmses, 
	py::array_t<double, py::array::c_style | py::array::forcecast> borders,
	py::array_t<long long, py::array::c_style | py::array::forcecast> kOpts,
	py::array_t<double, py::array::c_style | py::array::forcecast> res_rmses,
	py::array_t<double, py::array::c_style | py::array::forcecast> centers)
{
	auto* x_ptr = static_cast<double*>(x.request().ptr);
	auto* max_rmses_ptr = static_cast<double*>(max_rmses.request().ptr);
	auto* borders_ptr = static_cast<double*>(borders.request().ptr);
	auto* kOpts_ptr = static_cast<long long*>(kOpts.request().ptr);
	auto* res_rmses_ptr = static_cast<double*>(res_rmses.request().ptr);
	auto* centers_ptr = static_cast<double*>(centers.request().ptr);

	long long row_cnt = x.shape(0);
	long long col_cnt = x.shape(1);
	
	
	for (long long col = 0; col < col_cnt; ++col)
	{
		kOpts_ptr[col] = bin_search(x_ptr + col * row_cnt, row_cnt, max_rmses_ptr[col], borders_ptr + col * row_cnt, res_rmses_ptr + col, centers_ptr + col * row_cnt); 
	}
}
PYBIND11_MODULE(_kmeans_nlogu, m)
{
		m.def("kmeans_nlogu", &kmeans_nlogu, "The O(n lg U) implementation of 1d kmeans");
}
int main()
{
	double x[5] = {1.0, 2.0, 4.0, 8.0, 16.0};
	auto *borders = new double[11];
	auto *res_rmse = new double[11];
	auto *centers = new double[11];
	long long kopt = bin_search(x, 5,0.0000001, borders, res_rmse, centers);

	std::cout<<"Kopt: "<<kopt<<std::endl<<"Cluster borders:"<<std::endl;

	for (int i = 0; i < kopt + 1; ++i)
	{
		std::cout<<borders[i]<<std::endl;
	}

	delete[] borders;
	return 0; 
}