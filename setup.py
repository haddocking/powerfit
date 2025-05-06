#! env/bin/python
import os.path

import numpy
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


def main():
    ext_modules = [Extension("powerfit._powerfit",
                             [os.path.join("src", "_powerfit.pyx")],
                             include_dirs=[numpy.get_include()],
                             ),
                   Extension("powerfit._extensions",
                             [os.path.join("src", "_extensions.c")],
                             include_dirs=[numpy.get_include()],
                             extra_compile_args=['-ffast-math'],
                             ),
                   ]

    ext_modules = cythonize(ext_modules)

    setup(ext_modules=ext_modules,
         )


if __name__=='__main__':
    main()
