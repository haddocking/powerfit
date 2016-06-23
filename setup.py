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
                             include_dirs=[numpy.get_include()])]

    cmdclass = {}
    if CYTHON:
        ext_modules = cythonize(ext_modules)
        cmdclass = {'build_ext' : build_ext}

    package_data = {'powerfit': [os.path.join('data', '*.npy'), 'kernels.cl']}

    description = ("Rigid body fitting of high-resolution structures in "
        "low-resolution cryo-electron microscopy density maps")

    setup(name="powerfit",
          version='2.0.0',
          description=description,
          url="https://github.com/haddocking/powerfit",
          author='Gydo C.P. van Zundert',
          author_email='g.c.p.vanzundert@uu.nl',
          license="Apache",
          classifiers=[
              'Development Status :: 3 - Alpha',
              'Programming Language :: Python :: 2.7',
              'Intended Audience :: Science/Research',
              'Topic :: Scientific/Engineering :: Bio-Informatics',
              ],
          packages=packages,
          package_data = package_data,
          install_requires=['numpy>=1.8', 'scipy'],
          entry_points={
              'console_scripts': [
                  'powerfit = powerfit.powerfit:main',
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
