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
		//NOTE: n-dim numpy arrays are always treated in the c-side of pybind11 as 1-dim,
		//so we need to introduce an extremely cursed addressing
		//While yes, it does provide the .shape(dim) methods, there's no realistic way to reinterpret

		//(Thank god it's only two dimensions, I wouldn't want to know how numpy lays out n-dim arrays in memory)

		size_t col_offset = col * row_cnt;
		double* x_cur_col_ptr = x_ptr + col_offset;
		double* borders_cur_col_ptr = borders_ptr + col_offset;
		double* centers_cur_col_ptr = centers_ptr + col_offset;

		kOpts_ptr[col] = KMeans(
			x_cur_col_ptr, 
			row_cnt, 
			max_rmses_ptr[col], 
			borders_cur_col_ptr, 
			&res_rmses_ptr[col], 
			centers_cur_col_ptr
		); 
	}
}
PYBIND11_MODULE(_kmeans_nlogu, m)
{
		m.def("kmeans_nlogu", &kmeans_nlogu, "The O(n lg U) implementation of 1d kmeans");
}