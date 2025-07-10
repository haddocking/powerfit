
from collections import defaultdict, OrderedDict
from collections.abc import Sequence
from io import TextIOWrapper
import operator
from string import capwords

import numpy as np

from .elements import ELEMENTS

# records
MODEL = 'MODEL '
ATOM = 'ATOM  '
HETATM = 'HETATM'
TER = 'TER   '

MODEL_LINE = 'MODEL ' + ' ' * 4 + '{:>4d}\n'
ENDMDL_LINE = 'ENDMDL\n'
TER_LINE = 'TER   ' + '{:>5d}' + ' ' * 6 + '{:3s}' + ' ' + '{:1s}' + \
        '{:>4d}' + '{:1s}' + ' ' * 53 + '\n'
ATOM_LINE = '{:6s}' + '{:>5d}' + ' ' + '{:4s}' + '{:1s}' + '{:3s}' + ' ' + \
        '{:1s}' + '{:>4d}' + '{:1s}' + ' ' * 3 + '{:8.3f}' * 3 + '{:6.2f}' * 2 + \
        ' ' * 10 + '{:<2s}' + '{:2s}\n'
END_LINE = 'END   \n'

ATOM_DATA = ('record id name alt resn chain resi i x y z q b ' \
        'e charge').split()
TER_DATA = 'id resn chain resi i'.split()


def parse_pdb(infile):

    if isinstance(infile, TextIOWrapper):
        f = infile
    elif isinstance(infile, str):
        f = open(infile)
    else:
        raise TypeError('Input should be either a file or string.')

    pdb = defaultdict(list)
    model_number = 1
    for line in f:
        record = line[:6]
        if record in (ATOM, HETATM):
            pdb['model'].append(model_number)
            pdb['record'].append(record)
            pdb['id'].append(int(line[6:11]))
            name = line[12:16].strip()
            pdb['name'].append(name)
            pdb['alt'].append(line[16])
            pdb['resn'].append(line[17:20].strip())
            pdb['chain'].append(line[21])
            pdb['resi'].append(int(line[22:26]))
            pdb['i'].append(line[26])
            pdb['x'].append(float(line[30:38]))
            pdb['y'].append(float(line[38:46]))
            pdb['z'].append(float(line[46:54]))
            pdb['q'].append(float(line[54:60]))
            pdb['b'].append(float(line[60:66]))
            # Be forgiving when determining the element
            e = line[76:78].strip()
            if not e:
                # If element is not given, take the first non-numeric letter of
                # the name as element.
                for e in name:
                    if e.isalpha():
                        break
            pdb['e'].append(e)
            pdb['charge'].append(line[78: 80].strip())
        elif record == MODEL:
            model_number = int(line[10: 14])
    f.close()
    return pdb


def tofile(pdb, out):

    f = open(out, 'w')

    nmodels = len(set(pdb['model']))
    natoms = len(pdb['id'])
    natoms_per_model = natoms // nmodels

    for nmodel in range(nmodels):
        offset = int(nmodel * natoms_per_model)
        # write MODEL record
        if nmodels > 1:
            f.write(MODEL_LINE.format(nmodel + 1))
        prev_chain = pdb['chain'][offset]
        for natom in range(natoms_per_model):
            index = int(offset + natom)

            # write TER record
            current_chain = pdb['chain'][index]
            if prev_chain != current_chain:
                prev_record = pdb['record'][index - 1]
                if prev_record == ATOM:
                    line_data = [pdb[data][index - 1] for data in TER_DATA]
                    line_data[0] += 1
                    f.write(TER_LINE.format(*line_data))
                prev_chain = current_chain

            # write ATOM/HETATM record
            line_data = [pdb[data][index] for data in ATOM_DATA]
            # take care of the rules for atom name position
            e = pdb['e'][index]
            name = pdb['name'][index]
            if len(e) == 1 and len(name) != 4:
                line_data[2] = ' ' + name
            f.write(ATOM_LINE.format(*line_data))

        # write ENDMDL record
        if nmodels > 1:
            f.write(ENDMDL_LINE)

    f.write(END_LINE)
    f.close()


def pdb_dict_to_array(pdb):
    dtype = [('record', np.str_, 6), ('id', np.int32),
             ('name', np.str_, 4), ('alt', np.str_, 1),
             ('resn', np.str_, 4), ('chain', np.str_, 2),
             ('resi', np.int32), ('i', np.str_, 1), ('x', np.float64),
             ('y', np.float64), ('z', np.float64),
             ('q', np.float64), ('b', np.float64),
             ('e', np.str_, 2), ('charge', np.str_, 2),
             ('model', np.int32)]

    natoms = len(pdb['id'])
    pdb_array = np.empty(natoms, dtype=dtype)
    for data in ATOM_DATA:
        pdb_array[data] = pdb[data]
    pdb_array['model'] = pdb['model']
    return pdb_array


def pdb_array_to_dict(pdb_array):
    pdb = defaultdict(list)
    for data in ATOM_DATA:
        pdb[data] = pdb_array[data].tolist()
    pdb['model'] = pdb_array['model'].tolist()
    return pdb


class Structure(object):

    @classmethod
    def fromfile(cls, fid):
        """Initialize Structure from PDB-file"""
        try:
            fname = fid.name
        except AttributeError:
            fname = fid

        if fname[-3:] in ('pdb', 'ent'):
            arr = pdb_dict_to_array(parse_pdb(fid))
        elif fname[-3:] == 'cif':
            arr = mmcif_dict_to_array(parse_mmcif(fid))
        else:
            raise IOError('Filetype not recognized.')
        return cls(arr)

    def __init__(self, pdb):
        self.data = pdb

    @property
    def atomnumber(self):
        """Return array of atom numbers"""
        return self._get_property('number')

    @property
    def chain_list(self):
        return np.unique(self.data['chain'])

    def combine(self, structure):
        return Structure(np.hstack((self.data, structure.data)))

    @property
    def coor(self):
        """Return the coordinates"""
        return np.asarray([self.data['x'], self.data[ 'y'], self.data['z']])

    def duplicate(self):
        """Duplicate the object"""
        return Structure(self.data.copy())

    def _get_property(self, ptype):
        elements, ind = np.unique(self.data['e'], return_inverse=True)
        return np.asarray([getattr(ELEMENTS[capwords(e)], ptype) 
            for e in elements], dtype=np.float64)[ind]

    @property
    def mass(self):
        return self._get_property('mass')

    def rmsd(self, structure):
        return np.sqrt(((self.coor - structure.coor) ** 2).mean() * 3)

    def rotate(self, rotmat):
        """Rotate atoms"""
        self.data['x'], self.data['y'], self.data['z'] = (
              np.asmatrix(rotmat) * np.asmatrix(self.coor)
              )

    def select(self, identifier, values, loperator='==', return_ind=False):
        """A simple way of selecting atoms"""
        if loperator == '==':
            oper = operator.eq
        elif loperator == '<':
            oper = operator.lt
        elif loperator == '>':
            oper = operator.gt
        elif loperator == '>=':
            oper = operator.ge
        elif loperator == '<=':
            oper = operator.le
        elif loperator == '!=':
            oper = operator.ne
        else:
            raise ValueError('Logic operator not recognized.')

        if not isinstance(values, Sequence) or isinstance(values, str):
            values = (values,)

        selection = oper(self.data[identifier], values[0])
        if len(values) > 1:
            for v in values[1:]:
                if loperator == '!=':
                    selection &= oper(self.data[identifier], v)
                else:
                    selection |= oper(self.data[identifier], v)

        if return_ind:
            return selection
        else:
            return Structure(self.data[selection])

    @property
    def sequence(self):
        resids, indices = np.unique(self.data['resi'], return_index=True)
        return self.data['resn'][indices]

    def translate(self, trans):
        """Translate atoms"""
        self.data['x'] += trans[0]
        self.data['y'] += trans[1]
        self.data['z'] += trans[2]

    def tofile(self, fid):
        """Write instance to PDB-file"""
        tofile(pdb_array_to_dict(self.data), fid)

    @property
    def rvdw(self):
        return self._get_property('vdwrad')


def parse_mmcif(infile):
    if isinstance(infile, TextIOWrapper):
        pass
    elif isinstance(infile, str):
        infile = open(infile)
    else:
        raise TypeError("Input should either be a file or string.")

    atom_site = OrderedDict()
    with infile as f:
        for line in f:

            if line.startswith('_atom_site.'):
                words = line.split('.')
                atom_site[words[1].strip()] = []

            if line.startswith('ATOM'):
                words = line.split()
                for key, word in zip(atom_site, words):
                    atom_site[key].append(word)
    return atom_site


def mmcif_dict_to_array(atom_site):

    natoms = len(atom_site['id'])
    dtype = [('record', np.str_, 6), ('id', np.int32),
             ('name', np.str_, 4), ('alt', np.str_, 1),
             ('resn', np.str_, 4), ('chain', np.str_, 2),
             ('resi', np.int32), ('i', np.str_, 1), ('x', np.float64),
             ('y', np.float64), ('z', np.float64),
             ('q', np.float64), ('b', np.float64),
             ('e', np.str_, 2), ('charge', np.str_, 2),
             ('model', np.int32)]

    cifdata = np.zeros(natoms, dtype=dtype)
    cifdata['record'] = 'ATOM  '
    cifdata['id'] = atom_site['id']
    cifdata['name'] = atom_site['label_atom_id']
    cifdata['resn'] = atom_site['label_comp_id']
    cifdata['chain'] = atom_site['label_asym_id']
    cifdata['resi'] = atom_site['label_seq_id']
    cifdata['x'] = atom_site['Cartn_x']
    cifdata['y'] = atom_site['Cartn_y']
    cifdata['z'] = atom_site['Cartn_z']
    cifdata['q'] = atom_site['occupancy']
    cifdata['b'] = atom_site['B_iso_or_equiv']
    cifdata['e'] = atom_site['type_symbol']
    cifdata['charge'] = atom_site['pdbx_formal_charge']
    cifdata['model'] = atom_site['pdbx_PDB_model_num']
    return cifdata
