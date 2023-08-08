from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext = Pybind11Extension(
        "_ckmeans_1d_dp",
        [
            "src/algo/codec/kmeans_1d_c/_ckmeans_1d_dp.cpp",
            "src/algo/codec/kmeans_1d_c/Ckmeans.1d.dp.cpp",
            "src/algo/codec/kmeans_1d_c/dynamic_prog.cpp",
            "src/algo/codec/kmeans_1d_c/EWL2_dynamic_prog.cpp",
            "src/algo/codec/kmeans_1d_c/EWL2_fill_SMAWK.cpp"
        ],
    )
ext._add_cflags(["/Zi"])
ext._add_ldflags(["/DEBUG"])
ext_modules = [
    ext
]

setup(
    package_dir = {"ckmeans_1d_dp": "src"},
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules,
)