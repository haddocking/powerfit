def parse_pdb(pdbfile):
    ATOM = "ATOM "
    MODEL = "MODEL "

    model = []
    serial = []
    name = []
    alt_loc = []
    res_name = []
    chain_id = []
    res_seq = []
    i_code = []
    x = []
    y = []
    z = []
    occupancy = []
    temp_factor = []
    element = []
    charge = []

    model_number = 1
    for line in open(pdbfile):

        if line.startswith(MODEL):
            model_number = int(line[10:14])

        elif line.startswith(ATOM):

            model.append(model_number)
            serial.append(int(line[6:11].strip()))
            name.append(line[12:16].strip())
            alt_loc.append(line[16])
            res_name.append(line[17:20])
            chain_id.append(line[21])
            res_seq.append(int(line[22:26]))
            i_code.append(line[26])
            x.append(float(line[30:38]))
            y.append(float(line[38:46]))
            z.append(float(line[46:54]))
            occupancy.append(float(line[54:60]))
            temp_factor.append(float(line[60:66]))
            element.append(line[76:78].strip())
            charge.append(line[78:80])

            tmp = line[76:78].strip()
            if not tmp:
                tmp = line[12:16].strip()[0]

    natoms = len(name)
    dtype = [('atom_id', np.int32), ('name', np.str_, 4), 
             ('resn', np.str_, 4), ('chain', np.str_, 1), 
             ('resi', np.int32), ('x', np.float64),
             ('y', np.float64), ('z', np.float64), 
             ('occupancy', np.float64), ('bfactor', np.float64),
             ('element', np.str_, 2), ('charge', np.str_, 2),
             ('model', np.int32),
             ]
             
    pdbdata = np.zeros(natoms, dtype=dtype)
    pdbdata['atom_id'] = np.asarray(serial, dtype=np.int32)
    pdbdata['name'] = name
    pdbdata['resn'] = res_name
    pdbdata['chain'] = chain_id
    pdbdata['resi'] = res_seq
    pdbdata['x'] = x
    pdbdata['y'] = y
    pdbdata['z'] = z
    pdbdata['occupancy'] = occupancy
    pdbdata['bfactor'] = temp_factor
    pdbdata['element'] = element
    pdbdata['charge'] = charge
    pdbdata['model'] = model

    return pdbdata

def write_pdb(outfile, pdbdata):
    #HETATOM = "HETATM"
    #atom_line = ''.join(['{atom:6s}', '{serial:5d}', ' ', '{name:4s}', 
    #    '{altLoc:1s}', '{resName:3s}', ' ', '{chainID:1s}',
    #    '{resSeq:4d}', '{iCode:1s}', ' '*3, '{x:8.3f}', '{y:8.3f}', '{z:8.3f}', 
    #    '{occupancy:6.2f}', '{tempFactor:6.2f}', ' '*10, '{element:2s}',
    #    '{charge:2s}', '\n'])
    atom_line = ''.join(['{:6s}', '{:5d}', ' ', '{:4s}', 
        '{:1s}', '{:3s}', ' ', '{:1s}',
        '{:4d}', '{:1s}', ' '*3, '{:8.3f}', '{:8.3f}', '{:8.3f}', 
        '{:6.2f}', '{:6.2f}', ' '*10, '{:2s}',
        '{:2s}', '\n'])

    fhandle = open(outfile, 'w')
    for n in range(pdbdata.shape[0]):
        fhandle.write(atom_line.format('ATOM', 
                                       pdbdata['atom_id'][n], 
                                       pdbdata['name'][n],
                                       '',
                                       pdbdata['resn'][n],
                                       pdbdata['chain'][n],
                                       pdbdata['resi'][n],
                                       '',
                                       pdbdata['x'][n],
                                       pdbdata['y'][n],
                                       pdbdata['z'][n],
                                       pdbdata['occupancy'][n],
                                       pdbdata['bfactor'][n],
                                       pdbdata['element'][n],
                                       pdbdata['charge'][n],
                                       ))
    fhandle.write('END')
