"""Analysis pipeline - refactored from analyze_all.py"""
import json
import logging
import numpy as np
import pandas as pd
import mdtraj as md
from pathlib import Path
from typing import List, Dict, Optional

from pept_bioemu.pipeline.config_loader import load_peptide_configs_from_dict
from pept_bioemu.pipeline.bound_like_classifier import (
    BoundLikeClassifier,
    BoundLikeCriteria,
    threshold_sweep,
)

logger = logging.getLogger(__name__)

def run_analysis_pipeline(
    peptides_cfg: dict,
    project_dir: Path,
    output_dir: Path,
    peptide_filter: Optional[str] = None,
):
    """Run analysis for all peptides in config."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build criteria from config
    criteria = BoundLikeCriteria(
        rmsd_threshold=peptides_cfg['analysis']['rmsd_threshold'],
        helix_threshold=peptides_cfg['analysis']['helix_threshold'],
        anchor_dist_thresholds=tuple(peptides_cfg['analysis'].get(
            'anchor_dist_thresholds', BoundLikeCriteria().anchor_dist_thresholds)),
        phi_psi_tolerance=peptides_cfg['analysis']['phi_psi_tolerance'],
        rg_threshold=peptides_cfg['analysis']['rg_threshold'],
        # .get() with the BoundLikeCriteria default as fallback so existing
        # configs that predate the FlexE criterion / two-tier rule don't need
        # editing first.
        flexe_threshold=peptides_cfg['analysis'].get('flexe_threshold', BoundLikeCriteria().flexe_threshold),
        shape_min_pass=peptides_cfg['analysis'].get('shape_min_pass', BoundLikeCriteria().shape_min_pass),
        dock_min_pass=peptides_cfg['analysis'].get('dock_min_pass', BoundLikeCriteria().dock_min_pass),
    )
    
    # Load peptide configs
    configs = load_peptide_configs_from_dict(peptides_cfg)
    
    all_results = []
    for config in configs:
        if peptide_filter and config.name != peptide_filter:
            continue
        
        logger.info(f"=== Analyzing {config.name} ===")
        try:
            result = analyze_single_peptide(config, project_dir, criteria, output_dir)
            all_results.append(result)
        except Exception as e:
            logger.error(f"[{config.name}] Analysis failed: {e}")
            all_results.append({
                'peptide': config.name,
                'sequence': config.sequence,
                'n_samples': 0,
                'f_bound': 0.0,
                'dG_conf': np.nan,
                'dG_ci_lower': np.nan,
                'dG_ci_upper': np.nan,
                'error': str(e)
            })
    
    # Generate summary
    generate_summary_report(all_results, output_dir, criteria)

def load_peptide_configs_from_dict(peptides_cfg: dict):
    """Load PeptideConfig objects from parsed YAML dict."""
    from pept_bioemu.pipeline.config_loader import PeptideConfig
    import numpy as np
    
    configs = []
    for p in peptides_cfg['peptides']:
        anchor_positions = [x - 1 for x in p['anchor_positions']]
        pocket_centers = np.array(p['pocket_centers'], dtype=float)
        
        ref_residue_range = p.get('ref_residue_range')
        ref_residue_slice = None
        if ref_residue_range is not None:
            start, end = ref_residue_range
            ref_residue_slice = slice(start - 1, end)
        
        configs.append(PeptideConfig(
            name=p['name'],
            sequence=p['sequence'],
            bound_pdb_path=p['bound_pdb'],
            bound_chain=str(p['bound_chain']),
            anchor_positions=anchor_positions,
            pocket_centers=pocket_centers,
            groove_residues=p.get('groove_residues', []),
            is_stapled=p.get('is_stapled', False),
            is_macrocycle=p.get('is_macrocycle', False),
            ref_residue_range=ref_residue_range,
            ref_residue_slice=ref_residue_slice,
        ))
    return configs

def analyze_single_peptide(config, base_dir: Path, criteria: BoundLikeCriteria, output_dir: Path):
    """Analyze one peptide ensemble."""
    xtc_path = base_dir / "systems" / config.name / "bioemu" / "output" / "samples.xtc"
    top_path = base_dir / "systems" / config.name / "bioemu" / "output" / "topology.pdb"
    bound_ref_path = base_dir / config.bound_pdb_path
    
    if not xtc_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {xtc_path}")
    
    traj = md.load(str(xtc_path), top=str(top_path))
    logger.info(f"  Loaded {traj.n_frames} frames, {traj.n_residues} residues")
    
    classifier = BoundLikeClassifier(config, criteria)
    classifier.prepare(str(top_path), str(bound_ref_path))
    
    logger.info(f"  Classifying {traj.n_frames} frames...")
    result = classifier.classify_trajectory(traj)
    
    logger.info(f"  Bootstrap CI...")
    dg, ci_lo, ci_hi = classifier.bootstrap_dg_conf(result)
    result['dg_conf'] = dg
    result['dg_ci_lower'] = ci_lo
    result['dg_ci_upper'] = ci_hi
    
    logger.info(f"  Threshold sweep...")
    sweep = threshold_sweep(classifier, traj)
    result['threshold_sweep'] = sweep
    
    # Save arrays
    pep_out_dir = output_dir / config.name
    pep_out_dir.mkdir(parents=True, exist_ok=True)
    save_analysis_arrays(result, pep_out_dir, traj)
    
    return {
        'peptide': config.name,
        'sequence': config.sequence,
        'n_samples': traj.n_frames,
        'f_bound': result['f_bound'],
        'dG_conf': dg,
        'dG_ci_lower': ci_lo,
        'dG_ci_upper': ci_hi,
    }

def save_analysis_arrays(result, pep_out_dir: Path, traj):
    """Save .npy arrays for later use."""
    import json
    metrics = result['metrics']
    
    np.save(pep_out_dir / "rmsd_A.npy", np.array([m.rmsd for m in metrics]))
    np.save(pep_out_dir / "helix_per_frame.npy", np.array([m.helix_fraction for m in metrics]))
    np.save(pep_out_dir / "helix_per_residue.npy", 
            np.mean([m.helix_per_residue for m in metrics], axis=0))
    np.save(pep_out_dir / "anchor_distances.npy", 
            np.array([m.anchor_distances for m in metrics]))
    np.save(pep_out_dir / "phi_psi_anchors.npy", 
            np.array([m.phi_psi_anchors for m in metrics]))
    np.save(pep_out_dir / "rg.npy", np.array([m.rg for m in metrics]))
    np.save(pep_out_dir / "flexe_energy.npy", np.array([m.flexe_energy for m in metrics]))
    np.save(pep_out_dir / "criteria_passed.npy", result['criteria_passed_per_frame'])
    np.save(pep_out_dir / "is_bound_like.npy", result['is_bound_like'])
    
    # Save threshold sweep as JSON
    import json
    sweep_serializable = {}
    sweep = result.get('threshold_sweep', {})
    for k, v in sweep.items():
        sweep_serializable[k] = {str(thr): float(fb) for thr, fb in v.items()}
    with open(pep_out_dir / "threshold_sweep.json", 'w') as f:
        json.dump(sweep_serializable, f, indent=2)

def generate_summary_report(results: list, output_dir: Path, criteria: BoundLikeCriteria):
    """Generate summary CSV and comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / "summary_all.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved summary to {csv_path}")
    print(df.to_string(index=False))
    
    # Comparison plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        valid_results = [r for r in results if 'dG_conf' in r and not np.isnan(r.get('dG_conf', np.nan))]
        if not valid_results:
            return
            
        peptides = [r['peptide'] for r in valid_results]
        dg_vals = [r['dG_conf'] for r in valid_results]
        dg_err_lo = [r['dG_conf'] - r['dG_ci_lower'] for r in valid_results]
        dg_err_hi = [r['dG_ci_upper'] - r['dG_conf'] for r in valid_results]
        
        # ΔG_conf bar plot
        ax = axes[0, 0]
        colors = plt.cm.tab10(np.arange(len(peptides)))
        ax.bar(range(len(peptides)), dg_vals, yerr=[dg_err_lo, dg_err_hi], 
               capsize=5, color=colors, edgecolor='black', alpha=0.7)
        ax.set_xticks(range(len(peptides)))
        ax.set_xticklabels(peptides, rotation=45, ha='right')
        ax.set_ylabel('ΔG_conf (kcal/mol)')
        ax.set_title('Conformational Free Energy Penalty')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        
        # f_bound bar plot
        ax = axes[0, 1]
        f_bound_vals = [r['f_bound'] for r in valid_results]
        ax.bar(range(len(peptides)), f_bound_vals, color=colors, edgecolor='black', alpha=0.7)
        ax.set_xticks(range(len(peptides)))
        ax.set_xticklabels(peptides, rotation=45, ha='right')
        ax.set_ylabel('f_bound (bound-like fraction)')
        ax.set_title('Population of Bound-Like States')
        ax.set_ylim(0, 1)
        
        # Per-residue helical propensity heatmap
        ax = axes[1, 0]
        helix_data = []
        labels = []
        for r in valid_results:
            helix_file = output_dir / r['peptide'] / "helix_per_residue.npy"
            if helix_file.exists():
                helix_data.append(np.load(helix_file))
                labels.append(r['peptide'])
        
        if helix_data:
            max_len = max(len(h) for h in helix_data)
            helix_matrix = np.zeros((len(helix_data), max_len))
            for i, h in enumerate(helix_data):
                helix_matrix[i, :len(h)] = h
            
            im = ax.imshow(helix_matrix, aspect='auto', cmap='RdPu', vmin=0, vmax=1)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xlabel('Residue position')
            ax.set_title('Per-Residue Helical Propensity')
            plt.colorbar(im, ax=ax, label='Helix fraction')
        
        # Criteria contribution (first peptide)
        ax = axes[1, 1]
        if valid_results:
            first_pep = valid_results[0]['peptide']
            criteria_file = output_dir / first_pep / "criteria_passed.npy"
            if criteria_file.exists():
                criteria_passed = np.load(criteria_file)
                criteria_names = [
                    f'RMSD≤{criteria.rmsd_threshold:g}Å',
                    f'Helix≥{criteria.helix_threshold:.0%}',
                    f'Anchors≤{"/".join(f"{t:g}" for t in criteria.anchor_dist_thresholds)}Å (≥2/3)',
                    f'φ/ψ register (±{criteria.phi_psi_tolerance:g}°, ≥2/3)',
                    f'Rg≤{criteria.rg_threshold:g}Å',
                    f'FlexE≤{criteria.flexe_threshold:g} kcal/mol/res',
                ]
                n_criteria = criteria_passed.shape[1]
                pass_rates = criteria_passed.mean(axis=0)
                bars = ax.barh(range(n_criteria), pass_rates, color='purple', edgecolor='black', alpha=0.7)
                ax.set_yticks(range(n_criteria))
                ax.set_yticklabels(criteria_names[:n_criteria])
                ax.set_xlabel('Fraction of frames passing')
                ax.set_title(f'Criterion Contribution ({first_pep})')
                ax.set_xlim(0, 1)
                for i, (bar, rate) in enumerate(zip(bars, pass_rates)):
                    ax.text(rate + 0.02, bar.get_y() + bar.get_height()/2, 
                            f'{rate:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / "summary_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved comparison plot to {output_dir / 'summary_comparison.png'}")
        
    except ImportError:
        logger.warning("Matplotlib not available, skipping plots")