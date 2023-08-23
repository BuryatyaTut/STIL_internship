set(INCLUDE_DIR include)
set(SRC src/Ckmeans.1d.dp.cpp src/dynamic_prog.cpp src/EWL2_dynamic_prog.cpp src/EWL2_fill_SMAWK.cpp)
set(SRC_BINDINGS src/_ckmeans_1d_dp.cpp)
include_directories(${INCLUDE_DIR})

pybind11_add_module(_kmeans_1d_c ${SRC} ${SRC_BINDINGS})