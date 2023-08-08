import setuptools
import cmake_build_extension
from pathlib import Path



setuptools.setup(
    name='dataframe',
    version='0.1',
	author="Eliot Heinrich",
	description="Provides access to C++ classes and methods for managing parallel computation and data management",
	install_requires=["cmake_build_extension"],
	packages=['dataframe'],
    ext_modules=[
        cmake_build_extension.CMakeExtension(
			name="DataFrameBindings",
			install_prefix="dataframe",
			cmake_depends_on=["nanobind"],
			source_dir=str(Path(__file__).parent.absolute()),
			cmake_configure_options=[
				"-DCMAKE_BUILD_TYPE=Release",
				"-DCALL_FROM_SETUP_PY:BOOL=ON",
			],
		),
	],
    cmdclass=dict(
        build_ext=cmake_build_extension.BuildExtension,
    ),
)