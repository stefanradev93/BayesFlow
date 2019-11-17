from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension


extensions = [
    Extension("diffusion", ["diffusion.pyx"], include_dirs=['./simulators', '.'])
]
		

setup (
	ext_modules = cythonize(extensions),
)