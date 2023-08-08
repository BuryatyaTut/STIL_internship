#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "Ckmeans.1d.dp.h"

namespace py = pybind11;

long long Ckmeans_1d_dp(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                   const double max_rmse,
                   const size_t maxK,
                   py::array_t<long long> cluster,
                   py::array_t<double> centers)
{
    py::buffer_info bufx = x.request();

    auto *xp = static_cast<double *>(bufx.ptr);
    auto *cluster_p = static_cast<long long *>(cluster.request().ptr);
    auto *center_p = static_cast<double *>(centers.request().ptr);
    size_t length =  x.shape(x.ndim()-1);
    //NOTE: cluster and centers are modified and are to be treated as return values
    return kmeans_1d_dp(xp, length, NULL, maxK, maxK, cluster_p, center_p, NULL, NULL, NULL, max_rmse);
    
}


PYBIND11_MODULE(_ckmeans_1d_dp, m){
    m.doc() = "Python binding for Ckmeans.1d.dp";

    m.def("ckmeans", &Ckmeans_1d_dp, "the Ckmeans.1d.dp function");
}