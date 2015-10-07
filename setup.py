#! env/bin/python
import os.path
from distutils.core import setup
from distutils.extension import Extension
try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    CYTHON = True
except ImportError:
    CYTHON = False

# test for numpy version
import numpy
np_major, np_minor, np_release = [int(x) for x in numpy.version.short_version.split('.')]
if np_major < 1 or (np_major == 1 and np_minor < 8):
    raise ImportError('PowerFit requires NumPy version 1.8 or '
        'higher. You have version {:s}'.format(numpy.version.short_version))


def main():

    packages = ['powerfit', 'powerfit.IO']
    requires = ['numpy', 'scipy']

    # the C or Cython extension
    ext = '.pyx' if CYTHON else '.c'
    ext_modules = [Extension("powerfit.libpowerfit",
                             [os.path.join("src", "libpowerfit" + ext)],
                             include_dirs=[numpy.get_include()])]

    cmdclass = {}
    if CYTHON:
        ext_modules = cythonize(ext_modules)
        cmdclass = {'build_ext' : build_ext}

    package_data = {'powerfit': [os.path.join('data', '*.npy'), 
                                 os.path.join('kernels', '*.cl'), 
                                 ]}

    scripts = [os.path.join('scripts', 'powerfit'),
               os.path.join('scripts', 'atom2dens'),
               os.path.join('scripts', 'generate_fits'),
               os.path.join('scripts', 'image-pyramid'),
               ]

    setup(name="powerfit",
          version='1.1.3',
          description='PDB fitting in cryoEM maps',
          author='Gydo C.P. van Zundert',
          author_email='g.c.p.vanzundert@uu.nl',
          packages=packages,
          cmdclass=cmdclass,
          ext_modules = ext_modules,
          package_data = package_data,
          scripts=scripts,
          requires=requires,
          include_dirs=[numpy.get_include()],
        )

if __name__=='__main__':
    main()
