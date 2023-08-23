import glob
import os
import subprocess
import sys
from pathlib import Path
from shutil import copymode, move

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={extdir}{os.sep}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG={extdir}{os.sep}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={extdir}{os.sep}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-T Intel C++ Compiler 2023"

        ]
        build_args = [
            #"--config Release"
        ]
        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake",  "--build", ".", *build_args], cwd=build_temp, check=True
        )
        test_bin = os.path.join(extdir, 'stil_internship_test.*')
        self.copy_test_file(test_bin)

    def copy_test_file(self, src_file):
        '''
        Copy ``src_file`` to `tests/bin` directory, ensuring parent directory 
        exists. Messages like `creating directory /path/to/package` and
        `copying directory /src/path/to/package -> path/to/package` are 
        displayed on standard output. Adapted from scikit-build.
        '''
        # Create directory if needed
        dest_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'test', 'py', 'bin')
        if dest_dir != "" and not os.path.exists(dest_dir):
            print("creating directory {}".format(dest_dir))
            os.makedirs(dest_dir)

        # Copy file
        for file in glob.glob(src_file):
            dest_file = os.path.join(dest_dir, os.path.basename(file))
            print("copying {} -> {}".format(file, dest_file))
            move(file, dest_file)




setup(
    name="stil-internship",
    version='0.0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    ext_modules=[CMakeExtension("stil-internship")],
    cmdclass={"build_ext": CMakeBuild},
)

