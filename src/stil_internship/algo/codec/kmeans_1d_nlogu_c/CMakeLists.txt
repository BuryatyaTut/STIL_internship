set(INCLUDE_DIR include)
set(SRC src/KMeans.cpp src/SMAWK.cpp src/metrics.cpp)
set(SRC_BINDINGS src/_kmeans_nlogu.cpp)
include_directories(${INCLUDE_DIR})

pybind11_add_module(_kmeans_nlogu ${SRC} ${SRC_BINDINGS})