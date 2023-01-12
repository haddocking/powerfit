from __future__ import absolute_import
import sys

from powerfit.structure import Structure
from powerfit.volume import Volume, structure_to_shape_like

def main():
    target = Volume.fromfile(sys.argv[4])
    structure = Structure.fromfile(sys.argv[1])
    template = structure_to_shape_like(
          target, 
          structure.coor, resolution=float(sys.argv[2]),
          weights=structure.atomnumber, shape='vol'
          )
    template.tofile(sys.argv[3])

if __name__ == '__main__':
    main()
