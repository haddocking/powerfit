import sys

from powerfit_em.structure import Structure
from powerfit_em.volume import structure_to_shape


def main():

    structure = Structure.fromfile(sys.argv[1])
    template = structure_to_shape(
        structure.coor,
        resolution=float(sys.argv[2]),
        weights=structure.atomnumber,
        shape="vol",
    )
    template.tofile(sys.argv[3])


if __name__ == "__main__":
    main()
