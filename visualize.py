
## loading necessary modules

import numpy as np
import dataclasses
import torch
import einops
import py3Dmol
import itertools
import time

from diffusion import Diffuzz
from train import get_model, get_data_loader, transform_data
from data import add_back_useless_atoms, DataLoaderConfig, get_data_normalization


# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]

restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.

restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


@dataclasses.dataclass(frozen=True)
class Protein:
  """Protein structure representation."""

  # Cartesian coordinates of atoms in angstroms. The atom types correspond to
  # residue_constants.atom_types, i.e. the first three are N, CA, CB.
  atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

  # Amino-acid type for each residue represented as an integer between 0 and
  # 20, where 20 is 'X'.
  aatype: np.ndarray  # [num_res]

  # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
  # is present and 0.0 if not. This should be used for loss masking.
  atom_mask: np.ndarray  # [num_res, num_atom_type]

  # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
  residue_index: np.ndarray  # [num_res]

  # 0-indexed number corresponding to the chain in the protein that this residue
  # belongs to.
  chain_index: np.ndarray  # [num_res]

  # B-factors, or temperature factors, of each residue (in sq. angstroms units),
  # representing the displacement of the residue from its ground truth mean
  # value.
  b_factors: np.ndarray  # [num_res, num_atom_type]


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
  chain_end = 'TER'
  return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
          f'{chain_name:>1}{residue_index:>4}')


def to_pdb(prot: Protein) -> str:
  """Converts a `Protein` instance to a PDB string.

  Args:
    prot: The protein to convert to PDB.

  Returns:
    PDB string.
  """
  restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V', 'X']

  res_1to3 = lambda r: restype_1to3.get(restypes[r], 'UNK')

  pdb_lines = []

  atom_mask = prot.atom_mask
  aatype = prot.aatype
  atom_positions = prot.atom_positions
  residue_index = prot.residue_index.astype(np.int32)
  chain_index = prot.chain_index.astype(np.int32)
  b_factors = prot.b_factors

  if np.any(aatype > restype_num):
    raise ValueError('Invalid aatypes.')

  # Construct a mapping from chain integer indices to chain ID strings.
  chain_ids = {}
  for i in np.unique(chain_index):  # np.unique gives sorted output.
    if i >= PDB_MAX_CHAINS:
      raise ValueError(
          f'The PDB format supports at most {PDB_MAX_CHAINS} chains.')
    chain_ids[i] = PDB_CHAIN_IDS[i]

  pdb_lines.append('MODEL     1')
  atom_index = 1
  last_chain_index = chain_index[0]
  # Add all atom sites.
  for i in range(aatype.shape[0]):
    # Close the previous chain if in a multichain PDB.
    if last_chain_index != chain_index[i]:
      pdb_lines.append(_chain_end(
          atom_index, res_1to3(aatype[i - 1]), chain_ids[chain_index[i - 1]],
          residue_index[i - 1]))
      last_chain_index = chain_index[i]
      atom_index += 1  # Atom index increases at the TER symbol.

    res_name_3 = res_1to3(aatype[i])
    for atom_name, pos, mask, b_factor in zip(
        atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
      if mask < 0.5:
        continue

      record_type = 'ATOM'
      name = atom_name if len(atom_name) == 4 else f' {atom_name}'
      alt_loc = ''
      insertion_code = ''
      occupancy = 1.00
      element = atom_name[0]  # Protein supports only C, N, O, S, this works.
      charge = ''
      # PDB is a columnar format, every space matters here!
      atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                   f'{res_name_3:>3} {chain_ids[chain_index[i]]:>1}'
                   f'{residue_index[i]:>4}{insertion_code:>1}   '
                   f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                   f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                   f'{element:>2}{charge:>2}')
      pdb_lines.append(atom_line)
      atom_index += 1

  # Close the final chain.
  pdb_lines.append(_chain_end(atom_index, res_1to3(aatype[-1]),
                              chain_ids[chain_index[-1]], residue_index[-1]))
  pdb_lines.append('ENDMDL')
  pdb_lines.append('END')

  # Pad all lines to 80 characters.
  pdb_lines = [line.ljust(80) for line in pdb_lines]
  return '\n'.join(pdb_lines) + '\n'  # Add terminating newline.

def get_pdb_str(atom_positions, atom_mask, residue_index, max_seq_len):

    prot = Protein(
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        residue_index=residue_index,
        aatype=np.zeros([max_seq_len,], dtype=np.int32),
        chain_index=np.zeros([max_seq_len,], dtype=np.int32),
        b_factors=np.ones([max_seq_len, 37], dtype=np.int32)
    )

    pdb_str = to_pdb(prot)

    return pdb_str


def visualize_generated_samples(ckpt_path):

    device="cuda"

    ckpt = torch.load(ckpt_path)
    arch_config=ckpt["arch_config"]
    train_cfg=ckpt["train_config"]
    train_cfg.compile= False
    del ckpt


    model = get_model(train_cfg=train_cfg, arch_config=arch_config, device=device)
    model.eval()

    data_cfg = DataLoaderConfig()

    data_normalization = get_data_normalization(data_cfg)


    val_loader = get_data_loader(data_cfg, split="validation", shuffle=False)
    val_iter = iter(val_loader)
    val_iter = itertools.cycle(val_iter)

    diffuzz = Diffuzz(device=device)


    ## getting batch

    batch = next(val_iter)
    batch = transform_data(batch, data_cfg)
    attn_masks = batch["attn_masks"].to(device)
    batch_size=attn_masks.shape[0]
    atom_mask= batch["atom_mask"].to(device)

    cpu_atom_mask = batch["atom_mask"].to("cpu")
    cpu_residue_index =  batch["residue_index"].to("cpu")






    data_normalization = get_data_normalization(data_cfg)

    #for ix in range(0,64, 7):
    for ix in range(0,64,7):


        sampling_steps = 1000
        step_schedule="linear"
        sampler="ddpm"

        shape= (atom_mask.shape[0], atom_mask.shape[1], 12)
        ##fixing the start noise to compare between different options
        start_noise = torch.randn(*shape, device=diffuzz.device)

        max_seq_len=arch_config.seq_len

        true_atom_positions, atom_mask = add_back_useless_atoms(batch["atom_positions"].to("cpu"), cpu_atom_mask)

        view = py3Dmol.view(
            width=600, height=600, linked=True , viewergrid=(1, 1))
        view.setViewStyle({'style': 'outline', 'color': 'black', 'width': 0.1})
        style = {"cartoon": {'color': 'spectrum'}}


        pdb_str = get_pdb_str(true_atom_positions[ix].numpy(), atom_mask[ix].numpy(), cpu_residue_index[ix].numpy(), max_seq_len)


        view.addModelsAsFrames(pdb_str, viewer=(0, 0))
        view.setStyle({'model': -1}, style, viewer=(0, 0))
        view.zoomTo(viewer=(0, 0))


        print("TRUE PROTEIN")
        #view.render()
        view.show()


        print(f"SAMPLED PROTEIN {sampling_steps=}, {step_schedule=}")

        ## sampling
        with torch.no_grad():
                samples = diffuzz.sample(model=model, model_inputs={"attn_mask":attn_masks}, shape=shape, x_init=start_noise, timesteps=sampling_steps, sampler=sampler, steps=step_schedule)


        ## rearrange atom positions and get corresponding masks and indexes
        sampled_atom_positions = einops.rearrange(samples[sampling_steps-1], "batch seq_len (num_atoms coord) -> batch seq_len num_atoms coord", num_atoms=4, coord=3).to("cpu")

        ## denormalize

        sampled_atom_positions = (sampled_atom_positions*data_normalization["validation"]["std"]) + data_normalization["validation"]["mean"]

        sampled_atom_positions, atom_mask = add_back_useless_atoms(sampled_atom_positions, cpu_atom_mask)

        curr_atom_positions=sampled_atom_positions[ix].numpy()
        curr_atom_mask=atom_mask[ix].numpy()
        curr_residue_index=cpu_residue_index[ix].numpy()


        # Render.
        view = py3Dmol.view(
            width=600, height=600, linked=True , viewergrid=(1, 1))
        view.setViewStyle({'style': 'outline', 'color': 'black', 'width': 0.1})
        style = {"cartoon": {'color': 'spectrum'}}


        pdb_str = get_pdb_str(curr_atom_positions, curr_atom_mask, curr_residue_index, max_seq_len)


        view.addModelsAsFrames(pdb_str, viewer=(0, 0))
        view.setStyle({'model': -1}, style, viewer=(0, 0))
        view.zoomTo(viewer=(0, 0))

        #view.render()
        view.show()
        time.sleep(1.5)
