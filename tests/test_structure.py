import gzip

import pytest

from powerfit_em.structure import parse_mmcif, parse_pdb


@pytest.fixture
def sample_cif():
    return """\
data_6J5VA2A    
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . MET A 1 1 ? 88.638 99.772 125.454 1 165.94 ? 1 A 1
ATOM 2 C CA . MET A 1 1 ? 88.663 98.475 124.794 1 165.94 ? 1 A 1
"""


def test_parse_mmcif_filename_cif(sample_cif, tmp_path):
    fn = tmp_path / "sample.cif"
    fn.write_text(sample_cif)

    result = parse_mmcif(str(fn))

    assert len(result) == 19
    assert result["type_symbol"] == ["N", "C"]


def test_parse_mmcif_fileobj_cif(sample_cif, tmp_path):
    fn = tmp_path / "sample.cif"
    fn.write_text(sample_cif)

    with open(fn, "rb") as f:
        result = parse_mmcif(f)

    assert len(result) == 19
    assert result["type_symbol"] == ["N", "C"]


def test_parse_mmcif_filename_cif_gz(sample_cif, tmp_path):
    fn = tmp_path / "sample.cif.gz"
    with gzip.open(fn, "wt", encoding="utf-8") as f:
        f.write(sample_cif)

    result = parse_mmcif(str(fn))

    assert len(result) == 19
    assert result["type_symbol"] == ["N", "C"]


def test_parse_mmcif_fileobj_cif_gz(sample_cif, tmp_path):
    fn = tmp_path / "sample.cif.gz"
    with gzip.open(fn, "wt", encoding="utf-8") as f:
        f.write(sample_cif)

    with open(fn, "rb") as f:
        result = parse_mmcif(f)

    assert len(result) == 19
    assert result["type_symbol"] == ["N", "C"]


@pytest.fixture
def sample_pdb():
    return """\
ATOM      1  N   MET A   1      88.638  99.772 125.454  1.00165.94           N  
ATOM      2  CA  MET A   1      88.663  98.475 124.794  1.00165.94           C  
"""


def test_parse_pdb_filename_pdb(sample_pdb, tmp_path):
    fn = tmp_path / "sample.pdb"
    fn.write_text(sample_pdb)

    result = parse_pdb(str(fn))

    assert len(result) == 16
    assert result["name"] == ["N", "CA"]


def test_parse_pdb_fileobj_pdb(sample_pdb, tmp_path):
    fn = tmp_path / "sample.pdb"
    fn.write_text(sample_pdb)

    with open(fn, "r") as f:
        result = parse_pdb(f)

    assert len(result) == 16
    assert result["name"] == ["N", "CA"]


def test_parse_pdb_filename_pdb_gz(sample_pdb, tmp_path):
    fn = tmp_path / "sample.pdb.gz"
    with gzip.open(fn, "wt", encoding="utf-8") as f:
        f.write(sample_pdb)

    result = parse_pdb(str(fn))

    assert len(result) == 16
    assert result["name"] == ["N", "CA"]


def test_parse_pdb_fileobj_pdb_gz(sample_pdb, tmp_path):
    fn = tmp_path / "sample.pdb.gz"
    with gzip.open(fn, "wt", encoding="utf-8") as f:
        f.write(sample_pdb)

    with open(fn, "rb") as f:
        result = parse_pdb(f)

    assert len(result) == 16
    assert result["name"] == ["N", "CA"]
