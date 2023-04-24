from __future__ import absolute_import
import sys

from powerfit.structure import Structure
from powerfit.volume import Volume, structure_to_shape, structure_to_shape_like
import matplotlib.pyplot as plt

def main():
    target = Volume.fromfile(sys.argv[4])
    structure = Structure.fromfile(sys.argv[1])
    template = structure_to_shape_like(
          target, 
          structure.coor, resolution=float(sys.argv[3]),
          weights=0.4/structure.bfacs, shape='mask'
          )

    template.tofile(sys.argv[2])

    plt.imshow(template.grid[template.grid.shape[0]//2])
    plt.show()

    # target = Volume.fromfile(sys.argv[4])
    # structure = Structure.fromfile(sys.argv[1])

    # vol = _structure_to_shape_like(
    #     target,
    #     structure,
    #     resolution = float(sys.argv[2]),
    #     bfac=True,
    # )

    
    # vol.maskMap().tofile(sys.argv[3])
    # print(template)

if __name__ == '__main__':
    main()
