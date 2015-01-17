from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

ext_modules = [Extension("powerfit/libpowerfit",
        ["src/libpowerfit.pyx"])]

#scripts = ['scripts/powerfit', 'scripts/pdb2volume']
package_data = {'powerfit': ['data/*.npy', 'kernels/*.cl']}

setup(name="powerfit",
      version='0.1.0',
      description='PDB fitting in cryoEM maps',
      author='Gydo C.P. van Zundert',
      author_email='g.c.p.vanzundert@uu.nl',
      packages=['powerfit'],
      cmdclass = {'build_ext': build_ext},
      ext_modules = cythonize(ext_modules),
      package_data = package_data,
      #scripts=scripts,
      requires=['numpy', 'scipy', 'cython'],
    )
