# pept-bioemu

Integrated BioEmu ensemble generation + conformational analysis pipeline for MDM2-binding peptides.

## Installation

```bash
# CPU-only
pip install -e .

# With CUDA support
pip install -e ".[cuda]"
```

## Quick Start

```bash
# 1. Copy and edit config
cp config/peptides.yaml.example peptides.yaml
# Edit project_dir, conda_env paths, and peptide list

# 2. Activate conda environment
conda activate bioemu_env

# 3. Run BioEmu for all peptides
pept-bioemu bioemu --config peptides.yaml

# 4. Run analysis on existing outputs
pept-bioemu analyze --config peptides.yaml

# 5. Or run both sequentially
pept-bioemu run-all --config peptides.yaml
```

## Commands

| Command | Description |
|---------|-------------|
| `pept-bioemu bioemu` | Generate BioEmu ensembles |
| `pept-bioemu analyze` | Run conformational analysis |
| `pept-bioemu run-all` | Run BioEmu + analysis sequentially |

### Options

```bash
# Run BioEmu for specific peptide
pept-bioemu bioemu --config peptides.yaml --peptide WT_p53

# Force overwrite existing BioEmu outputs
pept-bioemu bioemu --config peptides.yaml --force

# Dry run (print commands without executing)
pept-bioemu bioemu --config peptides.yaml --dry-run

# Run analysis for specific peptide
pept-bioemu analyze --config peptides.yaml --peptide PMI

# Custom output directory
pept-bioemu analyze --config peptides.yaml --output-dir ./my_results

# Skip BioEmu or analysis in run-all
pept-bioemu run-all --config peptides.yaml --skip-bioemu
pept-bioemu run-all --config peptides.yaml --skip-analysis
```

## Configuration

See `config/peptides.yaml.example` for full configuration options.

### Required Fields

- `project_dir`: Base output directory
- `conda_env`: Conda environment name or full path
- `peptides`: List of peptide configurations

### Peptide Configuration

Each peptide entry requires:
- `name`: Unique identifier
- `sequence`: Amino acid sequence
- `bound_pdb`: Path to bound structure PDB (relative to project_dir or absolute)
- `bound_chain`: PDB chain letter or numeric index
- `anchor_positions`: 1-indexed positions of Phe/Trp/Leu anchors
- `pocket_centers`: 3x3 array of pocket center coordinates (Å)
- `ref_residue_range`: Optional [start, end] 1-indexed for trimming reference
- `groove_residues`: Optional binding groove/binding site residue indices

## Output Structure

```
project_dir/
├── WT_p53/
│   └── bioemu/
│       ├── output/
│       │   ├── samples.xtc
│       │   └── topology.pdb
│       ├── bioemu_so3_cache/
│       └── bioemu_embeds_cache/
├── PDIQ/
│   └── ...
├── PMI/
│   └── ...
└── results/
    ├── summary_all.csv
    ├── summary_comparison.png
    ├── WT_p53/
    │   ├── rmsd_A.npy
    │   ├── helix_per_frame.npy
    │   ├── ...
    │   └── threshold_sweep.json
    └── ...
```

## Pocket Centers

The default pocket centers in the example config are derived from 1YCR (MDM2-p53 complex). For new peptides binding the same binding groove, these can be reused. For different binding modes, run `extract_pocket_centers.py` on your bound complex.

## Requirements

- Python >= 3.9
- Conda environment with BioEmu dependencies
- GPU recommended for BioEmu generation

## Citation

If you use this pipeline, please cite:
- BioEmu: [original BioEmu paper]
- This analysis pipeline: [your publication]