"""Main CLI entry point: `pept-bioemu`"""
import sys
import logging
import pkgutil
from pathlib import Path
from typing import Optional

# CRITICAL: Register bioemu as top-level package BEFORE any BioEmu imports
# This allows BioEmu's internal imports (from bioemu.chemgraph import ...) to work
from pept_bioemu import bioemu as _bioemu
sys.modules['bioemu'] = _bioemu

import typer
from rich.console import Console
from rich.logging import RichHandler

from pept_bioemu.run import (
    run_bioemu_all,
    run_analysis_all,
    run_all,
    load_config,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=Console(stderr=True))],
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="pept-bioemu",
    help="BioEmu ensemble generation + conformational analysis for peptides",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()

@app.command("bioemu")
def bioemu_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="Path to peptides.yaml"),
    peptide: Optional[str] = typer.Option(None, "--peptide", "-p", help="Run only specific peptide"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing samples.xtc"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands without executing"),
):
    """Generate BioEmu ensembles for peptides in config."""
    run_bioemu_all(config, peptide=peptide, force=force, dry_run=dry_run)

@app.command("analyze")
def analyze_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="Path to peptides.yaml"),
    peptide: Optional[str] = typer.Option(None, "--peptide", "-p", help="Analyze only specific peptide"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Override results directory"),
):
    """Run conformational analysis on existing BioEmu outputs."""
    run_analysis_all(config, peptide=peptide, output_dir=output_dir)

@app.command("run-all")
def run_all_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="Path to peptides.yaml"),
    peptide: Optional[str] = typer.Option(None, "--peptide", "-p"),
    skip_bioemu: bool = typer.Option(False, "--skip-bioemu", help="Only run analysis"),
    skip_analysis: bool = typer.Option(False, "--skip-analysis", help="Only run BioEmu"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing BioEmu outputs"),
):
    """Run BioEmu generation + analysis sequentially."""
    from pept_bioemu.run import run_all as run_all_impl
    run_all_impl(config, peptide=peptide, skip_bioemu=skip_bioemu, skip_analysis=skip_analysis, force=force)

if __name__ == "__main__":
    app()