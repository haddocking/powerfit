from __future__ import print_function
import sys
from collections import OrderedDict
import numpy as np

def parse_cif(infile):
    if isinstance(infile, file):
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

    natoms = len(atom_site['id'])
    dtype = [('atom_id', np.int64), ('name', np.str_, 4), 
             ('resn', np.str_, 4), ('chain', np.str_, 2), 
             ('resi', np.int64), ('x', np.float64),
             ('y', np.float64), ('z', np.float64), 
             ('occupancy', np.float64), ('bfactor', np.float64),
             ('element', np.str_, 2), ('charge', np.str_, 2),
             ('model', np.int64),
             ]

    cifdata = np.zeros(natoms, dtype=dtype)
    cifdata['atom_id'] = np.asarray(atom_site['id'], dtype=np.int64)
    cifdata['name'] = atom_site['label_atom_id']
    cifdata['resn'] = atom_site['label_comp_id']
    cifdata['chain'] = atom_site['label_asym_id']
    cifdata['resi'] = atom_site['label_seq_id']
    cifdata['x'] = atom_site['Cartn_x']
    cifdata['y'] = atom_site['Cartn_y']
    cifdata['z'] = atom_site['Cartn_z']
    cifdata['occupancy'] = atom_site['occupancy']
    cifdata['bfactor'] = atom_site['B_iso_or_equiv']
    cifdata['element'] = atom_site['type_symbol']
    cifdata['charge'] = atom_site['pdbx_formal_charge']
    cifdata['model'] = atom_site['pdbx_PDB_model_num']

    return cifdata

if __name__=='__main__':    
    import sys
    infile = sys.argv[1]
    data = parse_cif(infile)

