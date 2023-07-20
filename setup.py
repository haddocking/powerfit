#! env/bin/python
import os.path

import numpy
from setuptools import setup
from setuptools.extension import Extension
try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    CYTHON = True
except ImportError:
    CYTHON = False


def main():

    packages = ['powerfit']

    # the C or Cython extension
    ext = '.pyx' if CYTHON else '.c'
    ext_modules = [Extension("powerfit._powerfit",
                             [os.path.join("src", "_powerfit" + ext)],
                             include_dirs=[numpy.get_include()]),
                   Extension("powerfit._extensions",
                             [os.path.join("src", "_extensions.c")],
                             include_dirs=[numpy.get_include()],
                             extra_compile_args=['-ffast-math'],
                             ),
                   ]

    cmdclass = {}
    if CYTHON:
        ext_modules = cythonize(ext_modules)
        cmdclass = {'build_ext' : build_ext}

    package_data = {'powerfit': [os.path.join('data', '*.npy'), 'kernels.cl']}

    description = ("Rigid body fitting of high-resolution structures in "
        "low-resolution cryo-electron microscopy density maps. (Python 3.8)) "
        "Updated to be compatible with Python 3.8 for information about the "
        "previous version, please contact g.c.p.vanzundert@uu.nl.")

    setup(name="powerfit38",
          version='1.1.0',
          description=description,
          url="https://github.com/hllelli2/powerfit37",
          author='Gydo C.P. van Zundert, Luc Elliott, Adam Simpkin',
          author_email='g.c.p.vanzundert@uu.nl, hllelli2@liverpool.ac.uk, hlasimpk@liverpool.ac.uk',
          license="Apache",
          classifiers=[
              'Development Status :: 3 - Alpha',
              'Programming Language :: Python :: 3.8',
              'Intended Audience :: Science/Research',
              'Topic :: Scientific/Engineering :: Bio-Informatics',
              ],
          packages=packages,
          package_data = package_data,
          install_requires=['numpy>=1.23.0', 'scipy'],
          entry_points={
              'console_scripts': [
                  'powerfit = powerfit.powerfit:run',
                  'image-pyramid = powerfit.scripts:image_pyramid',
                  'em2em = powerfit.scripts:em2em',
                  ]
              },
          ext_modules=ext_modules,
          include_dirs=[numpy.get_include()],
          cmdclass=cmdclass,
         )


if __name__=='__main__':
    main()
