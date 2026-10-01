"""
Configuration loader for peptide systems.
Parses peptides.yaml into PeptideConfig objects.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import List
from dataclasses import dataclass


@dataclass
class PeptideConfig:
    """Configuration for a specific peptide system."""
    name: str
    sequence: str
    bound_pdb_path: str
    bound_chain: str                   # PDB chain letter (e.g. "B") or numeric chain index
    anchor_positions: List[int]        # 0-indexed residue indices of Phe/Trp/Leu anchors
    pocket_centers: np.ndarray         # Shape (3, 3), Å - pocket center coords per anchor
    groove_residues: List[int]         # Groove/binding site residue indices for alignment (informational)
    is_stapled: bool = False
    is_macrocycle: bool = False
    ref_residue_range: List[int] = None  # 1-indexed inclusive [start, end] for reference PDB
    ref_residue_slice: slice = None      # 0-indexed slice for Python (computed from range)


def load_peptide_configs(yaml_path: str) -> List[PeptideConfig]:
    """
    Load peptide configurations from YAML file.

    Args:
        yaml_path: Path to peptides.yaml

    Returns:
        List of PeptideConfig objects

    Notes:
        `bound_chain` may be given either as a PDB chain letter (e.g. "B")
        or as a numeric chain index (e.g. "1"); it's stored as-is here and
        resolved to an MDTraj chain index in BoundLikeClassifier.prepare()
        (see bound_like_classifier._resolve_chain_index).

        `ref_residue_range` (optional): 1-indexed inclusive [start, end] range
        of residues to use from the bound reference PDB for RMSD/superposition.
        Useful when crystal structure is missing terminal residues that are
        present in the simulation (e.g., disordered C-terminus). If not provided,
        all residues are used.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    configs = []
    for p in data['peptides']:
        # Convert anchor positions to 0-indexed
        anchor_positions = [x - 1 for x in p['anchor_positions']]

        # Pocket centers as numpy array (3, 3), Å
        pocket_centers = np.array(p['pocket_centers'], dtype=float)
        if pocket_centers.shape != (3, 3):
            raise ValueError(f"pocket_centers must be (3, 3) array, got {pocket_centers.shape}")

        # MDM2 groove residues - informational only; optional.
        groove_residues = p.get('groove_residues', [])

        # Parse ref_residue_range if provided: [start, end] 1-indexed inclusive
        ref_residue_range = p.get('ref_residue_range')
        ref_residue_slice = None
        if ref_residue_range is not None:
            if not isinstance(ref_residue_range, (list, tuple)) or len(ref_residue_range) != 2:
                raise ValueError(f"ref_residue_range must be a list of two integers [start, end], got {ref_residue_range}")
            start, end = ref_residue_range
            if not (isinstance(start, int) and isinstance(end, int)):
                raise ValueError(f"ref_residue_range values must be integers, got {ref_residue_range}")
            if start < 1 or end < 1:
                raise ValueError(f"ref_residue_range must be 1-indexed (start >= 1, end >= 1), got {ref_residue_range}")
            if start > end:
                raise ValueError(f"ref_residue_range start must be <= end, got {ref_residue_range}")
            # Convert to 0-indexed slice (end is inclusive in YAML, exclusive in Python slice)
            ref_residue_slice = slice(start - 1, end)

        configs.append(PeptideConfig(
            name=p['name'],
            sequence=p['sequence'],
            bound_pdb_path=p['bound_pdb'],
            bound_chain=str(p['bound_chain']),
            anchor_positions=anchor_positions,
            pocket_centers=pocket_centers,
            groove_residues=groove_residues,
            is_stapled=p.get('is_stapled', False),
            is_macrocycle=p.get('is_macrocycle', False),
            ref_residue_range=ref_residue_range,
            ref_residue_slice=ref_residue_slice,
        ))
    return configs


def load_peptide_configs_from_dict(peptides_cfg: dict) -> List[PeptideConfig]:
    """Load PeptideConfig objects from parsed YAML dict (already loaded)."""
    configs = []
    for p in peptides_cfg['peptides']:
        # Convert anchor positions to 0-indexed
        anchor_positions = [x - 1 for x in p['anchor_positions']]

        # Pocket centers as numpy array (3, 3), Å
        pocket_centers = np.array(p['pocket_centers'], dtype=float)
        if pocket_centers.shape != (3, 3):
            raise ValueError(f"pocket_centers must be (3, 3) array, got {pocket_centers.shape}")

        # MDM2 groove residues - informational only; optional.
        groove_residues = p.get('groove_residues', [])

        # Parse ref_residue_range if provided: [start, end] 1-indexed inclusive
        ref_residue_range = p.get('ref_residue_range')
        ref_residue_slice = None
        if ref_residue_range is not None:
            if not isinstance(ref_residue_range, (list, tuple)) or len(ref_residue_range) != 2:
                raise ValueError(f"ref_residue_range must be a list of two integers [start, end], got {ref_residue_range}")
            start, end = ref_residue_range
            if not (isinstance(start, int) and isinstance(end, int)):
                raise ValueError(f"ref_residue_range values must be integers, got {ref_residue_range}")
            if start < 1 or end < 1:
                raise ValueError(f"ref_residue_range must be 1-indexed (start >= 1, end >= 1), got {ref_residue_range}")
            if start > end:
                raise ValueError(f"ref_residue_range start must be <= end, got {ref_residue_range}")
            # Convert to 0-indexed slice (end is inclusive in YAML, exclusive in Python slice)
            ref_residue_slice = slice(start - 1, end)

        configs.append(PeptideConfig(
            name=p['name'],
            sequence=p['sequence'],
            bound_pdb_path=p['bound_pdb'],
            bound_chain=str(p['bound_chain']),
            anchor_positions=anchor_positions,
            pocket_centers=pocket_centers,
            groove_residues=groove_residues,
            is_stapled=p.get('is_stapled', False),
            is_macrocycle=p.get('is_macrocycle', False),
            ref_residue_range=ref_residue_range,
            ref_residue_slice=ref_residue_slice,
        ))
    return configs


def save_peptide_configs(configs: List[PeptideConfig], yaml_path: str):
    """Save peptide configurations to YAML file (for reference)."""
    data = {'peptides': []}
    for c in configs:
        data['peptides'].append({
            'name': c.name,
            'sequence': c.sequence,
            'bound_pdb': c.bound_pdb_path,
            'bound_chain': c.bound_chain,
            'anchor_positions': [x + 1 for x in c.anchor_positions],  # back to 1-indexed
            'pocket_centers': c.pocket_centers.tolist(),
            'groove_residues': c.groove_residues,
            'is_stapled': c.is_stapled,
            'is_macrocycle': c.is_macrocycle
        })
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    # Test loading
    import sys
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "peptides.yaml"
    configs = load_peptide_configs(yaml_path)
    for c in configs:
        print(f"{c.name}: {c.sequence}")
        print(f"  PDB: {c.bound_pdb_path} chain {c.bound_chain}")
        print(f"  Anchors (0-indexed): {c.anchor_positions}")
        print(f"  Pocket centers shape: {c.pocket_centers.shape}")
        print(f"  MDM2 groove residues: {c.groove_residues}")
        print(f"  Stapled: {c.is_stapled}, Macrocycle: {c.is_macrocycle}")