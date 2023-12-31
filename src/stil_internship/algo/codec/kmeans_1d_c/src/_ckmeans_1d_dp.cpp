
#include <future>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <semaphore>
#include "Ckmeans.1d.dp.h"

namespace py = pybind11;

void Ckmeans_1d_dp(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                   py::array_t<double, py::array::c_style | py::array::forcecast> max_rmses,
                   py::array_t<double, py::array::c_style | py::array::forcecast> centers,
                   py::array_t<long long, py::array::c_style | py::array::forcecast> k_opts,
					const int nThreads,
					long long maxK)
{
    py::buffer_info bufx = x.request();
    
    auto *xp = static_cast<double *>(bufx.ptr);
    auto *center_p = static_cast<double *>(centers.request().ptr);
    auto *k_opts_p = static_cast<long long *>(k_opts.request().ptr);
    auto *max_rmses_p = static_cast<double *>(max_rmses.request().ptr);
    
    size_t length =  x.shape(1);
    size_t ncols = x.shape(0);
    std::counting_semaphore<> cs{nThreads};
    //NOTE: cluster and centers are modified and are to be treated as return values

    std::vector<std::future<long long>> jobs;

    for (size_t col = 0; col < ncols; col++)
    {
        jobs.emplace_back(std::async(
            std::launch::async | std::launch::deferred,

            kmeans_1d_dp, 
            
            xp + col * length,
            length, 
            center_p + col * length, 
            max_rmses_p[col], 
            maxK, 
            std::ref(cs)
			)
        );
       
    }
    for (size_t col = 0; col < ncols; col++)
    {
	    k_opts_p[col] = jobs[col].get();
        
    }
    
}


PYBIND11_MODULE(_ckmeans_1d_dp, m){
    m.doc() = "Python binding for Ckmeans.1d.dp";

    m.def("ckmeans", &Ckmeans_1d_dp, "the Ckmeans.1d.dp function");
}