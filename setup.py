#! env/bin/python
import os.path
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

# test for numpy version
import numpy
np_major, np_minor, np_release = [int(x) for x in numpy.version.short_version.split('.')]
if np_major < 1 or (np_major == 1 and np_minor < 8):
    raise ImportError('PowerFit requires NumPy version 1.8 or '
        'higher. You have version {:s}'.format(numpy.version.short_version))


def main():

    packages = ['powerfit', 'powerfit.IO']

    ext_modules = [Extension("powerfit.libpowerfit",
                             [os.path.join("src", "libpowerfit.pyx")],
                             include_dirs=[numpy.get_include()],
                             )
                  ]

    package_data = {'powerfit': [os.path.join('data', '*.npy'), 
                                 os.path.join('kernels', '*.cl'), 
                                 ]
                   }

    scripts = [os.path.join('scripts', 'powerfit'),
               os.path.join('scripts', 'atom2dens'),
               os.path.join('scripts', 'generate_fits'),
               ]

    setup(name="powerfit",
          version='1.0.1',
          description='PDB fitting in cryoEM maps',
          author='Gydo C.P. van Zundert',
          author_email='g.c.p.vanzundert@uu.nl',
          packages=packages,
          cmdclass = {'build_ext': build_ext},
          ext_modules = cythonize(ext_modules),
          package_data = package_data,
          scripts=scripts,
          requires=['numpy', 'scipy', 'cython'],
          include_dirs=[numpy.get_include()],
        )

if __name__=='__main__':

    main()
