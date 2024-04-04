import os
import numpy as np
import torch
import h5py

from Bio.PDB import *
from Bio import PDB
from Bio.Data.PDBData import protein_letters_3to1

import collections
import dataclasses
import logging
from typing import Any, Mapping, Optional, Sequence, Tuple

from cerebra.config_datatransform import model_config
from cerebra.data.data_modules import OpenFoldSingleDatasetHU

import esm

atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37.

"""General-purpose errors used throughout the data pipeline"""
class Error(Exception):
    """Base class for exceptions."""


class MultipleChainsError(Error):
    """An error indicating that multiple chains were found for a given ID."""
# Type aliases:
ChainId = str
PdbHeader = Mapping[str, Any]
PdbStructure = PDB.Structure.Structure
SeqRes = str
MmCIFDict = Mapping[str, Sequence[str]]


@dataclasses.dataclass(frozen=True)
class Monomer:
    id: str
    num: int


# Note - mmCIF format provides no guarantees on the type of author-assigned
# sequence numbers. They need not be integers.
@dataclasses.dataclass(frozen=True)
class AtomSite:
    residue_name: str
    author_chain_id: str
    mmcif_chain_id: str
    author_seq_num: str
    mmcif_seq_num: int
    insertion_code: str
    hetatm_atom: str
    model_num: int


# Used to map SEQRES index to a residue in the structure.
@dataclasses.dataclass(frozen=True)
class ResiduePosition:
    chain_id: str
    residue_number: int
    insertion_code: str


@dataclasses.dataclass(frozen=True)
class ResidueAtPosition:
    position: Optional[ResiduePosition]
    name: str
    is_missing: bool
    hetflag: str


@dataclasses.dataclass(frozen=True)
class MmcifObject:
    """Representation of a parsed mmCIF file.

    Contains:
      file_id: A meaningful name, e.g. a pdb_id. Should be unique amongst all
        files being processed.
      header: Biopython header.
      structure: Biopython structure.
      chain_to_seqres: Dict mapping chain_id to 1 letter amino acid sequence. E.g.
        {'A': 'ABCDEFG'}
      seqres_to_structure: Dict; for each chain_id contains a mapping between
        SEQRES index and a ResidueAtPosition. e.g. {'A': {0: ResidueAtPosition,
                                                          1: ResidueAtPosition,
                                                          ...}}
      raw_string: The raw string used to construct the MmcifObject.
    """

    file_id: str
    header: PdbHeader
    structure: PdbStructure
    chain_to_seqres: Mapping[ChainId, SeqRes]
    seqres_to_structure: Mapping[ChainId, Mapping[int, ResidueAtPosition]]
    raw_string: Any


@dataclasses.dataclass(frozen=True)
class ParsingResult:
    """Returned by the parse function.

    Contains:
      mmcif_object: A MmcifObject, may be None if no chain could be successfully
        parsed.
      errors: A dict mapping (file_id, chain_id) to any exception generated.
    """

    mmcif_object: Optional[MmcifObject]
    errors: Mapping[Tuple[str, str], Any]


class ParseError(Exception):
    """An error indicating that an mmCIF file could not be parsed."""


def mmcif_loop_to_list(
    prefix: str, parsed_info: MmCIFDict
) -> Sequence[Mapping[str, str]]:
    """Extracts loop associated with a prefix from mmCIF data as a list.

    Reference for loop_ in mmCIF:
      http://mmcif.wwpdb.org/docs/tutorials/mechanics/pdbx-mmcif-syntax.html

    Args:
      prefix: Prefix shared by each of the data items in the loop.
        e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,
        _entity_poly_seq.mon_id. Should include the trailing period.
      parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython
        parser.

    Returns:
      Returns a list of dicts; each dict represents 1 entry from an mmCIF loop.
    """
    cols = []
    data = []
    for key, value in parsed_info.items():
        if key.startswith(prefix):
            cols.append(key)
            data.append(value)

    assert all([len(xs) == len(data[0]) for xs in data]), (
        "mmCIF error: Not all loops are the same length: %s" % cols
    )

    return [dict(zip(cols, xs)) for xs in zip(*data)]


def mmcif_loop_to_dict(
    prefix: str,
    index: str,
    parsed_info: MmCIFDict,
) -> Mapping[str, Mapping[str, str]]:
    """Extracts loop associated with a prefix from mmCIF data as a dictionary.

    Args:
      prefix: Prefix shared by each of the data items in the loop.
        e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,
        _entity_poly_seq.mon_id. Should include the trailing period.
      index: Which item of loop data should serve as the key.
      parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython
        parser.

    Returns:
      Returns a dict of dicts; each dict represents 1 entry from an mmCIF loop,
      indexed by the index column.
    """
    entries = mmcif_loop_to_list(prefix, parsed_info)
    return {entry[index]: entry for entry in entries}


def parse( file_id: str, path, catch_all_errors: bool = True
) -> ParsingResult:
    """Entry point, parses an mmcif_string.

    Args:
      file_id: A string identifier for this file. Should be unique within the
        collection of files being processed.
      mmcif_string: Contents of an mmCIF file.
      catch_all_errors: If True, all exceptions are caught and error messages are
        returned as part of the ParsingResult. If False exceptions will be allowed
        to propagate.

    Returns:
      A ParsingResult.
    """
    errors = {}
    try:
        parser = PDB.MMCIFParser(QUIET=True)
        full_structure = parser.get_structure("", path)
        first_model_structure = _get_first_model(full_structure)
        # Extract the _mmcif_dict from the parser, which contains useful fields not
        # reflected in the Biopython structure.
        parsed_info = parser._mmcif_dict  # pylint:disable=protected-access

        # Ensure all values are lists, even if singletons.
        for key, value in parsed_info.items():
            if not isinstance(value, list):
                parsed_info[key] = [value]

        header = _get_header(parsed_info)

        # Determine the protein chains, and their start numbers according to the
        # internal mmCIF numbering scheme (likely but not guaranteed to be 1).
        valid_chains = _get_protein_chains(parsed_info=parsed_info)
        if not valid_chains:
            return ParsingResult(
                None, {(file_id, ""): "No protein chains found in this file."}
            )
        seq_start_num = {
            chain_id: min([monomer.num for monomer in seq])
            for chain_id, seq in valid_chains.items()
        }

        # Loop over the atoms for which we have coordinates. Populate two mappings:
        # -mmcif_to_author_chain_id (maps internal mmCIF chain ids to chain ids used
        # the authors / Biopython).
        # -seq_to_structure_mappings (maps idx into sequence to ResidueAtPosition).
        mmcif_to_author_chain_id = {}
        seq_to_structure_mappings = {}
        for atom in _get_atom_site_list(parsed_info):
            if atom.model_num != "1":
                # We only process the first model at the moment.
                continue

            mmcif_to_author_chain_id[atom.mmcif_chain_id] = atom.author_chain_id

            if atom.mmcif_chain_id in valid_chains:
                hetflag = " "
                if atom.hetatm_atom == "HETATM":
                    # Water atoms are assigned a special hetflag of W in Biopython. We
                    # need to do the same, so that this hetflag can be used to fetch
                    # a residue from the Biopython structure by id.
                    if atom.residue_name in ("HOH", "WAT"):
                        hetflag = "W"
                    else:
                        hetflag = "H_" + atom.residue_name
                insertion_code = atom.insertion_code
                if not _is_set(atom.insertion_code):
                    insertion_code = " "
                position = ResiduePosition(
                    chain_id=atom.author_chain_id,
                    residue_number=int(atom.author_seq_num),
                    insertion_code=insertion_code,
                )
                seq_idx = (
                    int(atom.mmcif_seq_num) - seq_start_num[atom.mmcif_chain_id]
                )
                current = seq_to_structure_mappings.get(
                    atom.author_chain_id, {}
                )
                current[seq_idx] = ResidueAtPosition(
                    position=position,
                    name=atom.residue_name,
                    is_missing=False,
                    hetflag=hetflag,
                )
                seq_to_structure_mappings[atom.author_chain_id] = current

        # Add missing residue information to seq_to_structure_mappings.
        for chain_id, seq_info in valid_chains.items():
            author_chain = mmcif_to_author_chain_id[chain_id]
            current_mapping = seq_to_structure_mappings[author_chain]
            for idx, monomer in enumerate(seq_info):
                if idx not in current_mapping:
                    current_mapping[idx] = ResidueAtPosition(
                        position=None,
                        name=monomer.id,
                        is_missing=True,
                        hetflag=" ",
                    )

        author_chain_to_sequence = {}
        for chain_id, seq_info in valid_chains.items():
            author_chain = mmcif_to_author_chain_id[chain_id]
            seq = []
            for monomer in seq_info:
                code = protein_letters_3to1.get(monomer.id, "X")
                seq.append(code if len(code) == 1 else "X")
            seq = "".join(seq)
            author_chain_to_sequence[author_chain] = seq

        mmcif_object = MmcifObject(
            file_id=file_id,
            header=header,
            structure=first_model_structure,
            chain_to_seqres=author_chain_to_sequence,
            seqres_to_structure=seq_to_structure_mappings,
            raw_string=parsed_info,
        )

        return ParsingResult(mmcif_object=mmcif_object, errors=errors)
    except Exception as e:  # pylint:disable=broad-except
        errors[(file_id, "")] = e
        if not catch_all_errors:
            raise
        return ParsingResult(mmcif_object=None, errors=errors)


def _get_first_model(structure: PdbStructure) -> PdbStructure:
    """Returns the first model in a Biopython structure."""
    return next(structure.get_models())


_MIN_LENGTH_OF_CHAIN_TO_BE_COUNTED_AS_PEPTIDE = 21


def get_release_date(parsed_info: MmCIFDict) -> str:
    """Returns the oldest revision date."""
    revision_dates = parsed_info["_pdbx_audit_revision_history.revision_date"]
    return min(revision_dates)


def _get_header(parsed_info: MmCIFDict) -> PdbHeader:
    """Returns a basic header containing method, release date and resolution."""
    header = {}

    experiments = mmcif_loop_to_list("_exptl.", parsed_info)
    header["structure_method"] = ",".join(
        [experiment["_exptl.method"].lower() for experiment in experiments]
    )

    # Note: The release_date here corresponds to the oldest revision. We prefer to
    # use this for dataset filtering over the deposition_date.
    if "_pdbx_audit_revision_history.revision_date" in parsed_info:
        header["release_date"] = get_release_date(parsed_info)
    else:
        logging.warning(
            "Could not determine release_date: %s", parsed_info["_entry.id"]
        )

    header["resolution"] = 0.00
    for res_key in (
        "_refine.ls_d_res_high",
        "_em_3d_reconstruction.resolution",
        "_reflns.d_resolution_high",
    ):
        if res_key in parsed_info:
            try:
                raw_resolution = parsed_info[res_key][0]
                header["resolution"] = float(raw_resolution)
            except ValueError:
                logging.info(
                    "Invalid resolution format: %s", parsed_info[res_key]
                )

    return header


def _get_atom_site_list(parsed_info: MmCIFDict) -> Sequence[AtomSite]:
    """Returns list of atom sites; contains data not present in the structure."""
    return [
        AtomSite(*site)
        for site in zip(  # pylint:disable=g-complex-comprehension
            parsed_info["_atom_site.label_comp_id"],
            parsed_info["_atom_site.auth_asym_id"],
            parsed_info["_atom_site.label_asym_id"],
            parsed_info["_atom_site.auth_seq_id"],
            parsed_info["_atom_site.label_seq_id"],
            parsed_info["_atom_site.pdbx_PDB_ins_code"],
            parsed_info["_atom_site.group_PDB"],
            parsed_info["_atom_site.pdbx_PDB_model_num"],
        )
    ]


def _get_protein_chains(
    *, parsed_info: Mapping[str, Any]
) -> Mapping[ChainId, Sequence[Monomer]]:
    """Extracts polymer information for protein chains only.

    Args:
      parsed_info: _mmcif_dict produced by the Biopython parser.

    Returns:
      A dict mapping mmcif chain id to a list of Monomers.
    """
    # Get polymer information for each entity in the structure.
    entity_poly_seqs = mmcif_loop_to_list("_entity_poly_seq.", parsed_info)

    polymers = collections.defaultdict(list)
    for entity_poly_seq in entity_poly_seqs:
        polymers[entity_poly_seq["_entity_poly_seq.entity_id"]].append(
            Monomer(
                id=entity_poly_seq["_entity_poly_seq.mon_id"],
                num=int(entity_poly_seq["_entity_poly_seq.num"]),
            )
        )

    # Get chemical compositions. Will allow us to identify which of these polymers
    # are proteins.
    chem_comps = mmcif_loop_to_dict("_chem_comp.", "_chem_comp.id", parsed_info)

    # Get chains information for each entity. Necessary so that we can return a
    # dict keyed on chain id rather than entity.
    struct_asyms = mmcif_loop_to_list("_struct_asym.", parsed_info)

    entity_to_mmcif_chains = collections.defaultdict(list)
    for struct_asym in struct_asyms:
        chain_id = struct_asym["_struct_asym.id"]
        entity_id = struct_asym["_struct_asym.entity_id"]
        entity_to_mmcif_chains[entity_id].append(chain_id)

    # Identify and return the valid protein chains.
    valid_chains = {}
    for entity_id, seq_info in polymers.items():
        chain_ids = entity_to_mmcif_chains[entity_id]

        # Reject polymers without any peptide-like components, such as DNA/RNA.
        if any(
            [
                "peptide" in chem_comps[monomer.id]["_chem_comp.type"]
                for monomer in seq_info
            ]
        ):
            for chain_id in chain_ids:
                valid_chains[chain_id] = seq_info
    return valid_chains


def _is_set(data: str) -> bool:
    """Returns False if data is a special mmCIF character indicating 'unset'."""
    return data not in (".", "?")
def _atom_coord_transf_get(parsed_info: MmCIFDict):
    trans_m_x1 = np.float16(parsed_info['_pdbx_struct_oper_list.matrix[1][1]'])
    if len(trans_m_x1) !=2:
        print('matrix number not match!',parsed_info['data_'])
        return
    trans_m_y1 = np.float16(parsed_info['_pdbx_struct_oper_list.matrix[1][2]'])
    trans_m_z1 = np.float16(parsed_info['_pdbx_struct_oper_list.matrix[1][3]'])
    trans_m_x2 = np.float16(parsed_info['_pdbx_struct_oper_list.matrix[2][1]'])
    trans_m_y2 = np.float16(parsed_info['_pdbx_struct_oper_list.matrix[2][2]'])
    trans_m_z2 = np.float16(parsed_info['_pdbx_struct_oper_list.matrix[2][3]'])
    trans_m_x3 = np.float16(parsed_info['_pdbx_struct_oper_list.matrix[3][1]'])
    trans_m_y3 = np.float16(parsed_info['_pdbx_struct_oper_list.matrix[3][2]'])
    trans_m_z3 = np.float16(parsed_info['_pdbx_struct_oper_list.matrix[3][3]'])
    rot_matrix = np.zeros([2,3,3])
    rot_matrix[:,0,0] = trans_m_x1
    rot_matrix[:,0,1] = trans_m_y1
    rot_matrix[:,0,2] = trans_m_z1
    rot_matrix[:,1,0] = trans_m_x2
    rot_matrix[:,1,1] = trans_m_y2
    rot_matrix[:,1,2] = trans_m_z2
    rot_matrix[:,2,0] = trans_m_x3
    rot_matrix[:,2,1] = trans_m_y3
    rot_matrix[:,2,2] = trans_m_z3
    trans_v = np.zeros([2,3])
    trans_v[:,0] = np.float16(parsed_info['_pdbx_struct_oper_list.vector[1]'])
    trans_v[:,1] = np.float16(parsed_info['_pdbx_struct_oper_list.vector[2]'])
    trans_v[:,2] = np.float16(parsed_info['_pdbx_struct_oper_list.vector[3]'])
    if (rot_matrix[0] != np.eye(3)).all():
        print('first coord rot error',parsed_info['data_'])
        return
    if (trans_v[0] != np.zeros(3)).all():
        print('first coord tran error',parsed_info['data_'])
        return
    return rot_matrix[1],trans_v[1]
def get_atom_coords(
    mmcif_object: MmcifObject, 
    chain_id: str, 
    _zero_center_positions: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    # Locate the right chain
    chains = list(mmcif_object.structure.get_chains())
    relevant_chains = [c for c in chains if c.id == chain_id]
    if len(relevant_chains) != 1:
        raise MultipleChainsError(
            f"Expected exactly one chain in structure with id {chain_id}."
        )
    chain = relevant_chains[0]

    # Extract the coordinates
    num_res = len(mmcif_object.chain_to_seqres[chain_id])
    all_atom_positions = np.zeros(
        [num_res, atom_type_num, 3], dtype=np.float32    # residue_constants.atom_type_num:37
    )
    all_atom_mask = np.zeros(
        [num_res, atom_type_num], dtype=np.float32
    )
    for res_index in range(num_res):
        pos = np.zeros([atom_type_num, 3], dtype=np.float32)
        mask = np.zeros([atom_type_num], dtype=np.float32)
        res_at_position = mmcif_object.seqres_to_structure[chain_id][res_index]
        if not res_at_position.is_missing:
            res = chain[
                (
                    res_at_position.hetflag,
                    res_at_position.position.residue_number,
                    res_at_position.position.insertion_code,
                )
            ]
            for atom in res.get_atoms():
                atom_name = atom.get_name()
                x, y, z = atom.get_coord()
                if atom_name in atom_order.keys():
                    pos[atom_order[atom_name]] = [x, y, z]
                    mask[atom_order[atom_name]] = 1.0
                elif atom_name.upper() == "SE" and res.get_resname() == "MSE":
                    # Put the coords of the selenium atom in the sulphur column
                    pos[atom_order["SD"]] = [x, y, z]
                    mask[atom_order["SD"]] = 1.0

        all_atom_positions[res_index] = pos
        all_atom_mask[res_index] = mask

    if _zero_center_positions:
        binary_mask = all_atom_mask.astype(bool)
        translation_vec = all_atom_positions[binary_mask].mean(axis=0)
        all_atom_positions[binary_mask] -= translation_vec

    return all_atom_positions, all_atom_mask


def cif_data_load(name,path,id:list,one_chain_missing: bool = False):
    cif_data = parse(name,path)
    ori_seq = cif_data.mmcif_object.chain_to_seqres
    if one_chain_missing:
        all_atom_pos0 , all_atom_mask0 = get_atom_coords(cif_data.mmcif_object,id[0])
        parsed_info = cif_data.mmcif_object.raw_string
        rot_m, trans_v = _atom_coord_transf_get(parsed_info)
        all_atom_pos1 = np.matmul(all_atom_pos0,rot_m) + trans_v
        all_atom_mask1 = all_atom_mask0
        return ori_seq[id[0]],all_atom_pos0 , all_atom_mask0, all_atom_pos1 , all_atom_mask1
    if ori_seq[id[0]]!=ori_seq[id[1]]:
        print(name,'seq not match')
        print(ori_seq)
        return 
    all_atom_pos0 , all_atom_mask0 = get_atom_coords(cif_data.mmcif_object,id[0])
    all_atom_pos1 , all_atom_mask1 = get_atom_coords(cif_data.mmcif_object,id[1])
    return ori_seq[id[0]],all_atom_pos0 , all_atom_mask0, all_atom_pos1 , all_atom_mask1

def cif_data_load_normal(name,path,id:list):
    cif_data = parse(name,path)
    ori_seq = cif_data.mmcif_object.chain_to_seqres
    all_atom_pos0 , all_atom_mask0 = get_atom_coords(cif_data.mmcif_object,id[0])
    return ori_seq[id[0]],all_atom_pos0 , all_atom_mask0


gpu = torch.device('cpu')
esm_model_path = "/export/disk4/chenyinghui/database/Evolutionary_Scale_Modeling/esm2_t36_3B_UR50D.pt"
model, alphabet = esm.pretrained.load_model_and_alphabet(esm_model_path)
batch_converter = alphabet.get_batch_converter()
model.to(gpu)
model.eval()  

def esm2(seq_fasta):
    seqs = [['target', seq_fasta]]
    batch_size = 1
    seqs = [seqs[i:i+batch_size] for i in range(0, len(seqs), batch_size)]

    for data in seqs:
        with torch.no_grad():
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            results = model(batch_tokens.to(gpu), repr_layers=[36], return_contacts=True)
            
            token_embeds = results["representations"][36] # (batch=1, L+2, dim=2560) token_representations
            token_embeds = token_embeds[0, 1:-1, :].to(dtype=torch.float32).cpu().detach().numpy() # (batch=1, L, dim=2560)
            return token_embeds



low_prec = False
config = model_config(
    name="model_3",
    train=True, 
    low_prec=low_prec,
)

def dataloader(data_dir, alignment_dir, chain_data_cache_path, mode, batch_size, shuffle_mode):
    data = OpenFoldSingleDatasetHU(
        data_dir = data_dir,
        alignment_dir = alignment_dir,
        config = config.data,
        chain_data_cache_path = chain_data_cache_path,
        obsolete_pdbs_file_path = None,
        template_mmcif_dir = None,
        max_template_date = None,
        mode = mode,
    )
    return data 

cif_dir = 'rawdata/cif/'
msa_dir ='rawdata'
out_file = 'data'
chain_data_cache_json = 'list10.json'

train_dataloader = dataloader(
    data_dir = cif_dir,
    alignment_dir = msa_dir,
    chain_data_cache_path = chain_data_cache_json,
    mode='train', 
    batch_size=1, 
    shuffle_mode = True
)

fout_list = open('list.txt', 'w')

for batch in train_dataloader:
    mask = torch.tensor(batch['all_atom_mask'][:, [1, 2, 0]])
    mask = (mask.sum(-1) > 0.5) + (mask[:, 0] > 0.5)
    name = batch['name']
    msa  = torch.tensor(batch['msa'])
    seq = batch['sequence'][0].decode('utf-8')
    seq = np.array([seq.encode('utf-8')])
    
    xyz = torch.tensor(batch['all_atom_positions'][:, [1, 2, 0, 3]])
    CA = xyz[:, 0]
    C  = xyz[:, 1]
    N  = xyz[:, 2]
    CB = xyz[:, 3]

    if msa.shape[1] <= mask.shape[0]:

        CB[torch.where(msa[0] == 5)] = CA[torch.where(msa[0] == 5)]
        xyz = torch.stack([CA, C, N, CB], dim=1)
        _, all_atom_pos0, all_atom_mask0 = cif_data_load_normal(name, f'{cif_dir}{name[:4]}.cif', [name[5:]])

        seq_fasta = [x.strip() for x in os.popen(f'cat {msa_dir}/{name}/bfd_uniclust_hits.a3m').readlines()][1]
        

        X1D = esm2(seq_fasta)

        if mask.shape[0] == msa.shape[1]:
            fout_list.write('%s\t%d\n' % (name, mask.sum().item()))

            outfile = f'{out_file}/{name}.h5'
            fout = h5py.File(outfile, 'a')
            fout.create_dataset('xyz', data=xyz, dtype=np.float32)
            fout.create_dataset('toks', data=msa, dtype=np.int8)
            fout.create_dataset('mask', data=mask.long(), dtype=np.int8)
            fout.create_dataset('seq', data=seq, dtype='S100000')
            fout.create_dataset('all_atoms_pos', data=all_atom_pos0, dtype=np.float32)
            fout.create_dataset('all_atoms_mask', data=all_atom_mask0, dtype=np.int8)
            fout.create_dataset('X1D', data=X1D, dtype=np.float32)
            fout.close()
fout_list.close()