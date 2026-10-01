#!/usr/bin/env python
"""
TIER 0 -- reference-pose recovery.

The cheapest and strictest sanity check on the bound-like classifier:
score the EXPERIMENTAL bound conformation itself.

The crystal peptide is, by definition, bound-like. If it does not pass the
criteria, then f_bound is being measured against a target the true answer
cannot reach, and every dG_conf built on it is biased high.

This needs no BioEmu sampling -- it only reads the bound PDBs already in the
project, so it can be run before or after any ensemble is generated.

Usage:
    python benchmark/check_reference_pose.py -c config/peptides.yaml
    python benchmark/check_reference_pose.py -c config/peptides.yaml --sweep
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

try:
    import mdtraj as md
except ImportError:
    sys.exit("mdtraj is required: activate the pept-bioemu environment first.")

def _load_pipeline_modules():
    """Load the two pipeline modules directly from their files.

    `import pept_bioemu...` triggers pept_bioemu/__init__.py, which eagerly
    imports every BioEmu submodule (torch, JAX, TensorFlow) to set up the
    `bioemu` alias. That costs minutes on a shared filesystem and none of it
    is needed here: bound_like_classifier.py and config_loader.py import only
    numpy, mdtraj, yaml and the stdlib. Load them by path and skip all of it.
    Falls back to the normal import if the layout ever changes.
    """
    import importlib.util

    pkg = Path(__file__).resolve().parent.parent / "src" / "pept_bioemu" / "pipeline"
    mods = {}
    try:
        for name in ("bound_like_classifier", "config_loader"):
            spec = importlib.util.spec_from_file_location(
                f"_benchmark_{name}", pkg / f"{name}.py"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mods[name] = mod
        return (
            mods["bound_like_classifier"].BoundLikeClassifier,
            mods["bound_like_classifier"].BoundLikeCriteria,
            mods["bound_like_classifier"]._resolve_chain_index,
            mods["config_loader"].load_peptide_configs_from_dict,
        )
    except Exception:  # noqa: BLE001
        from pept_bioemu.pipeline.bound_like_classifier import (
            BoundLikeClassifier, BoundLikeCriteria, _resolve_chain_index,
        )
        from pept_bioemu.pipeline.config_loader import load_peptide_configs_from_dict
        return (BoundLikeClassifier, BoundLikeCriteria,
                _resolve_chain_index, load_peptide_configs_from_dict)


(BoundLikeClassifier, BoundLikeCriteria,
 _resolve_chain_index, load_peptide_configs_from_dict) = _load_pipeline_modules()

CRITERIA_NAMES = [
    "C1 RMSD to bound",
    "C2 helicity",
    "C3 anchors seated",
    "C4 phi/psi register",
    "C5 radius of gyration",
]


def extract_peptide_chain(pdb_path: Path, chain_spec: str, out_path: Path) -> md.Trajectory:
    """Write the bound peptide chain on its own so it can act as both
    the reference and the single-frame 'trajectory'."""
    full = md.load(str(pdb_path))
    idx = _resolve_chain_index(full.topology, chain_spec)
    sel = full.topology.select(f"chainid {idx}")
    if len(sel) == 0:
        raise ValueError(f"chain '{chain_spec}' selected 0 atoms in {pdb_path}")
    pep = full.atom_slice(sel)
    pep.save_pdb(str(out_path))
    return pep


def check_one(cfg, project_dir: Path, criteria: BoundLikeCriteria, verbose=True) -> dict:
    bound_pdb = project_dir / cfg.bound_pdb_path
    if not bound_pdb.exists():
        return {"peptide": cfg.name, "error": f"missing {bound_pdb}"}

    with tempfile.TemporaryDirectory() as td:
        pep_pdb = Path(td) / f"{cfg.name}_peptide.pdb"
        try:
            pep = extract_peptide_chain(bound_pdb, cfg.bound_chain, pep_pdb)
        except Exception as e:  # noqa: BLE001
            return {"peptide": cfg.name, "error": str(e)}

        # Guard: a Tier-0 check is only meaningful when the reference chain IS
        # this peptide. Control entries (Ala1/Ala2) borrow 1YCR chain B as a
        # shape reference, so scoring it would silently re-score p53's pose
        # under the control's name.
        ref_seq = "".join(r.code if r.code else "X" for r in pep.topology.residues)
        cfg_seq = cfg.sequence
        if cfg.ref_residue_slice is not None:
            cfg_seq = cfg_seq[cfg.ref_residue_slice]
        if ref_seq and not (ref_seq.startswith(cfg_seq[:len(ref_seq)])
                            or cfg_seq.startswith(ref_seq[:len(cfg_seq)])):
            return {
                "peptide": cfg.name,
                "error": (f"reference chain is a different peptide "
                          f"(PDB chain = {ref_seq}, config sequence = {cfg.sequence}). "
                          f"Tier 0 does not apply to borrowed references."),
            }

        clf = BoundLikeClassifier(cfg, criteria)
        # Reference and "trajectory" are the same crystal chain.
        clf.prepare(str(pep_pdb), str(bound_pdb))
        res = clf.classify_trajectory(pep)

    passed = res["criteria_passed_per_frame"][0]
    m = res["metrics"][0]
    out = {
        "peptide": cfg.name,
        "rmsd_A": m.rmsd,
        "helix_fraction": m.helix_fraction,
        "anchor_distances_A": np.round(m.anchor_distances, 2).tolist(),
        "rg_A": m.rg,
        "n_criteria_passed": int(passed.sum()),
        "is_bound_like": bool(res["is_bound_like"][0]),
        "criteria": {n: bool(p) for n, p in zip(CRITERIA_NAMES, passed)},
    }

    if verbose:
        ok = "PASS" if out["is_bound_like"] else "FAIL"
        print(f"\n--- {cfg.name}  [{ok}]  {out['n_criteria_passed']}/5 criteria")
        print(f"    RMSD {m.rmsd:.2f} A (self-reference, expect ~0)")
        print(f"    helix fraction {m.helix_fraction:.2f}")
        print(f"    Rg {m.rg:.2f} A")
        print(f"    anchor -> pocket distances (A): {out['anchor_distances_A']}"
              f"   threshold {criteria.anchor_dist_threshold}")
        for n, p in out["criteria"].items():
            print(f"      [{'x' if p else ' '}] {n}")
    return out


def anchor_sweep(results: list[dict]) -> None:
    """Smallest anchor_dist_threshold at which every reference pose would
    satisfy C3 (>=2 of 3 anchors within the cut-off)."""
    print("\n" + "=" * 68)
    print("ANCHOR THRESHOLD CALIBRATION")
    print("=" * 68)
    print("For each reference pose, the 2nd-smallest anchor distance is the")
    print("cut-off at which C3 (>=2 of 3) starts to pass.\n")
    needed = []
    for r in results:
        if "anchor_distances_A" not in r:
            continue
        d = sorted(x for x in r["anchor_distances_A"] if np.isfinite(x))
        if len(d) >= 2:
            needed.append((r["peptide"], d[1]))
            print(f"  {r['peptide']:<10s} sorted distances {d}  -> needs >= {d[1]:.2f} A")
    if needed:
        worst = max(n for _, n in needed)
        print(f"\n  Minimum threshold for C3 to pass on EVERY crystal pose: {worst:.2f} A")
        print("  (Current default in config/peptides.yaml: 4.5 A)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", type=Path, required=True)
    ap.add_argument("--sweep", action="store_true",
                    help="report the anchor cut-off each reference pose would need")
    args = ap.parse_args()

    cfg_dict = yaml.safe_load(args.config.read_text())
    project_dir = Path(cfg_dict["project_dir"]).expanduser().resolve()
    a = cfg_dict["analysis"]
    criteria = BoundLikeCriteria(
        rmsd_threshold=a["rmsd_threshold"],
        helix_threshold=a["helix_threshold"],
        anchor_dist_threshold=a["anchor_dist_threshold"],
        phi_psi_tolerance=a["phi_psi_tolerance"],
        rg_threshold=a["rg_threshold"],
        min_criteria_pass=a["min_criteria_pass"],
    )

    print("=" * 68)
    print("TIER 0: does the classifier recognise the experimental bound pose?")
    print("=" * 68)

    results = [check_one(c, project_dir, criteria)
               for c in load_peptide_configs_from_dict(cfg_dict)]

    scored = [r for r in results if "error" not in r]
    failed = [r for r in scored if not r["is_bound_like"]]
    for r in results:
        if "error" in r:
            print(f"\n--- {r['peptide']}  [SKIPPED] {r['error']}")

    print("\n" + "=" * 68)
    if not scored:
        print("VERDICT: nothing could be scored.")
        return 2
    if failed:
        print(f"VERDICT: FAIL -- {len(failed)}/{len(scored)} crystal poses are NOT "
              f"classified bound-like:")
        for r in failed:
            print(f"    {r['peptide']}: {r['n_criteria_passed']}/5")
        print("\nThe true bound conformation cannot reach the target. f_bound is")
        print("therefore an underestimate and dG_conf is biased high. Recalibrate")
        print("before treating any dG_conf value as physical.")
    else:
        print(f"VERDICT: PASS -- all {len(scored)} crystal poses classified bound-like.")
    print("=" * 68)

    if args.sweep:
        anchor_sweep(scored)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
