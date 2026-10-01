"""Core run functions for BioEmu generation and analysis."""
import sys
import yaml
import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def _get_peptide_dirs(project_dir: Path, pep: dict) -> tuple:
    pep_dir = project_dir / "systems" / pep['name']
    out_dir = pep_dir / "bioemu" / "output"
    so3_cache = pep_dir / "bioemu" / "bioemu_so3_cache"
    embeds_cache = pep_dir / "bioemu" / "bioemu_embeds_cache"
    return pep_dir, out_dir, so3_cache, embeds_cache

def _has_existing_output(out_dir: Path) -> bool:
    return (out_dir / "samples.xtc").exists() and (out_dir / "topology.pdb").exists()

def run_bioemu_for_peptide(
    pep: dict,
    cfg: dict,
    project_dir: Path,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    pep_dir, out_dir, so3_cache, embeds_cache = _get_peptide_dirs(project_dir, pep)
    
    if not force and _has_existing_output(out_dir):
        logger.info(f"[{pep['name']}] Skipping BioEmu - output exists (use --force to overwrite)")
        return False
    
    if dry_run:
        logger.info(f"[DRY RUN] Would create directories: {out_dir}, {so3_cache}, {embeds_cache}")
        logger.info(f"[DRY RUN] python -m pept_bioemu.bioemu.sample --sequence {pep['sequence']} --num_samples {cfg['bioemu']['num_samples']} --output_dir {out_dir} --cache_so3_dir {so3_cache} --cache_embeds_dir {embeds_cache}")
        return False
    
    out_dir.mkdir(parents=True, exist_ok=True)
    so3_cache.mkdir(parents=True, exist_ok=True)
    embeds_cache.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "pept_bioemu.bioemu.sample",
        "--sequence", pep['sequence'],
        "--num_samples", str(cfg['bioemu']['num_samples']),
        "--output_dir", str(out_dir),
        "--cache_so3_dir", str(so3_cache),
        "--cache_embeds_dir", str(embeds_cache),
    ]
    
    logger.info(f"[{pep['name']}] Running BioEmu ({cfg['bioemu']['num_samples']} samples)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"BioEmu failed for {pep['name']}: {result.stderr}")
    logger.info(f"[{pep['name']}] BioEmu completed")
    return True

def run_bioemu_all(
    config_path: Path,
    peptide: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
):
    cfg = load_config(config_path)
    project_dir = Path(cfg['project_dir']).expanduser().resolve()
    
    for pep in cfg['peptides']:
        if peptide and pep['name'] != peptide:
            continue
        logger.info(f"=== BioEmu: {pep['name']} ===")
        run_bioemu_for_peptide(pep, cfg, project_dir, force=force, dry_run=dry_run)

def run_analysis_all(
    config_path: Path,
    peptide: Optional[str] = None,
    output_dir: Optional[Path] = None,
):
    cfg = load_config(config_path)
    project_dir = Path(cfg['project_dir']).expanduser().resolve()
    out_dir = output_dir or (project_dir / "results")
    
    from pept_bioemu.pipeline.analyze import run_analysis_pipeline
    run_analysis_pipeline(cfg, project_dir, out_dir, peptide_filter=peptide)

def run_all(
    config_path: Path,
    peptide: Optional[str] = None,
    skip_bioemu: bool = False,
    skip_analysis: bool = False,
    force: bool = False,
):
    if not skip_bioemu:
        run_bioemu_all(config_path, peptide=peptide, force=force)
    if not skip_analysis:
        run_analysis_all(config_path, peptide=peptide)

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)