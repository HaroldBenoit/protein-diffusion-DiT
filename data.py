
import torch
import pandas as pd
import numpy as np
import dataclasses
import einops



def load_data():
    df = pd.read_json('chain_set.jsonl', lines=True)
    cath_splits = pd.read_json('chain_set_splits.json', lines=True)
    print('Read data.')

    def get_split(pdb_name):
        if pdb_name in cath_splits.train[0]:
            return 'train'
        elif pdb_name in cath_splits.validation[0]:
            return 'validation'
        elif pdb_name in cath_splits.test[0]:
            return 'test'
        else:
            return 'None'

    df['split'] = df.name.apply(lambda x: get_split(x))
    df['seq_len'] = df.seq.apply(lambda x: len(x))

    return df

PROTEIN_ATOMS = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]

def process_coordinates(coord_input):
    """Convert raw coordinate dictionary into structured numpy arrays"""
    backbone = ['N', 'CA', 'C', 'O']
    backbone_indices = {atom: idx for idx, atom in enumerate(PROTEIN_ATOMS) 
                       if atom in backbone}
    
    sequence_length = len(coord_input['N'])
    position_matrix = np.zeros((sequence_length, len(PROTEIN_ATOMS), 3))
    mask_matrix = np.zeros((sequence_length, len(PROTEIN_ATOMS)))
    
    for atom, idx in backbone_indices.items():
        position_matrix[:, idx] = coord_input[atom]
    
    invalid_coords = np.isnan(position_matrix[..., 0])
    position_matrix[invalid_coords] = 0
    
    mask_matrix[:, list(backbone_indices.values())] = 1
    mask_matrix[invalid_coords] = 0
    
    return {
        'coordinates': position_matrix,
        'atom_validity': mask_matrix,
        'res_indices': np.arange(sequence_length),
        'length': np.array([sequence_length])
    }

def standardize_length(features, target_length=500):
    """Standardize all features to a fixed sequence length"""
    for key, array in features.items():
        if key != 'length':
            current_len = array.shape[0]
            if current_len < target_length:
                pad_width = [(0, target_length - current_len)] + [(0, 0)] * (array.ndim - 1)
                features[key] = np.pad(array, pad_width)
            elif current_len > target_length:
                features[key] = array[:target_length]
        else:
            features[key] = np.minimum(array, target_length)

def normalize_coordinates(features):
    """Normalize coordinates relative to CA centroid"""
    coords = features['coordinates']
    validity = features['atom_validity']
    
    ca_coords = coords[:, 1]  # CA is always second atom
    ca_valid = validity[:, 1]
    
    centroid = np.sum(ca_valid[..., None] * ca_coords, axis=0) / (np.sum(ca_valid) + 1e-9)
    features['coordinates'] = (coords - centroid[None, None]) * validity[..., None]

class ProteinStructureDataset(torch.utils.data.Dataset):
    def __init__(self, structure_df, max_length=512):
        self.structures = structure_df
        self.max_length = max_length

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, index):
        raw_coords = self.structures.iloc[index].coords
        processed = process_coordinates(raw_coords)
        standardize_length(processed, self.max_length)
        normalize_coordinates(processed)
        
        return {
            'coordinates': torch.tensor(processed['coordinates'], dtype=torch.float32),
            'atom_validity': torch.tensor(processed['atom_validity'], dtype=torch.float32),
            'res_indices': torch.tensor(processed['res_indices']),
            'length': torch.tensor(processed['length'])
        }




def remove_useless_atoms(atom_positions, atom_mask):

  backbone_atoms = [0,1,2,4]

  ## make code agnostic to batched or not
  if len(atom_positions.shape) == 3:
    atom_positions = einops.rearrange(atom_positions, "ctx_len num_atom_types coord -> 1 ctx_len num_atom_types coord ")

  if len(atom_mask.shape) == 2:
    atom_mask = einops.rearrange(atom_mask, "ctx_len num_atom_types -> 1 ctx_len num_atom_types")

  ## only keep the backbone atoms

  return atom_positions[:,:,backbone_atoms,:], atom_mask[:,:,backbone_atoms]


def add_back_useless_atoms(atom_positions, atom_mask):

  # backbone_atoms = [0,1,2,4]

  ## make code agnostic to batched or not
  if len(atom_positions.shape) == 3:
    atom_positions = einops.rearrange(atom_positions, "ctx_len num_atom_types coord -> 1 ctx_len num_atom_types coord ")

  if len(atom_mask.shape) == 2:
    atom_mask = einops.rearrange(atom_mask, "ctx_len num_atom_types -> 1 ctx_len num_atom_types")

  if isinstance(atom_mask, torch.Tensor):
    zeros = torch.zeros
    concat = torch.cat
    concat_kwargs = {"dim":2}
  elif isinstance(atom_mask, np.ndarray):
    zeros = np.zeros
    concat = np.concatenate
    concat_kwargs = {"axis":2}
  else:
    raise NotImplementedError()


  batch, ctx_len, num_atom_types = atom_mask.shape

  #num_atoms_to_add_back = 37 - 5 = 32 (not counting the atom in-between the backbone atoms)

  atom_positions = concat([atom_positions[:,:,[0,1,2],:], zeros((batch, ctx_len, 1, 3)) , atom_positions[:,:,[3],:] , zeros((batch, ctx_len, 32, 3))], **concat_kwargs)

  atom_mask = concat([atom_mask[:,:,[0,1,2]], zeros((batch, ctx_len, 1)) , atom_mask[:,:,[3]] , zeros((batch, ctx_len, 32))], **concat_kwargs)


  return atom_positions, atom_mask

def create_attn_masks(batch, seq_len):

  num_residues = batch["num_res"]
  if len(num_residues.shape) == 1:
    num_residues = einops.rearrange(num_residues, "ctx_len  -> 1 ctx_len")

  masks = []
  for num_res in num_residues:
    mask = torch.ones((seq_len,seq_len))

    mask[:,num_res:] = 0
    mask = mask.bool()
    masks.append(mask)

  masks = torch.stack(masks)


  return masks


def transform_data(batch, data_cfg):
  atom_positions, atom_mask = remove_useless_atoms(batch["atom_positions"], batch["atom_mask"])
  attn_masks = create_attn_masks(batch, data_cfg.seq_len)

  #input = einops.rearrange(atom_positions, "batch seq_len num_atoms coord -> batch seq_len (num_atoms coord)")


  return {"atom_positions": atom_positions,  "attn_masks": attn_masks, "atom_mask": atom_mask, "residue_index": batch["residue_index"], "num_res": batch["num_res"]}

