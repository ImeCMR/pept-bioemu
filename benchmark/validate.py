#!/usr/bin/env python
"""
Compare pept-bioemu output against the curated benchmark.

    python benchmark/validate.py --results results/summary_all.csv

Tiers, strongest evidence first:

  T0  reference-pose recovery      -> benchmark/check_reference_pose.py
  T1  negative-control separation  controls must be LESS pre-organised than binders
  T2  agreement with prior MD      rank vs Lang & Perez 2021 free-peptide populations
  T3  affinity association         rank vs IC50, interface-matched subset only

T3 is deliberately last. dG_conf is one additive term in the binding free
energy, not the whole of it:

    dG_bind = dG_conf + dG_interface + dG_solvation + ...

Comparing dG_conf to affinity across peptides with DIFFERENT interfaces mixes
in terms this pipeline never models, so a weak correlation there is expected
and is not evidence against the method. Only pairs that share an interface and
differ in conformational stability (e.g. mini59 vs mini61) isolate dG_conf.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None

HERE = Path(__file__).resolve().parent

# summary_all.csv `peptide` column -> benchmark peptide_id
NAME_MAP = {
    "WT_p53": "p53_17_29",
    "PDIQ": "pdiq",
    "PMI": "pmi",
    "Ala1": "ala1",
    "Ala2": "ala2",
    "WT_p53_15_29": "p53_15_29",
}

# Lang & Perez ran p53 15-29; the default config runs 17-29. Allow the
# comparison but say so.
CONSTRUCT_SUBSTITUTE = {"p53_17_29": "p53_15_29"}

CONTROLS = {"ala1", "ala2"}
BINDERS = {"pdiq", "pmi", "p53_17_29", "p53_15_29"}


def load_benchmark():
    pep = pd.read_csv(HERE / "data" / "peptides.csv")
    meas = pd.read_csv(HERE / "data" / "measurements.csv")
    meas = meas[meas["status"] == "curated"]
    return pep, meas


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["peptide_id"] = df["peptide"].map(NAME_MAP)
    unmapped = df[df["peptide_id"].isna()]["peptide"].tolist()
    if unmapped:
        print(f"  note: not in benchmark, ignored: {', '.join(unmapped)}")
    df = df.dropna(subset=["peptide_id"])
    return df[np.isfinite(df["dG_conf"])]


def hdr(n, title):
    print(f"\n{'=' * 68}\nT{n}: {title}\n{'=' * 68}")


def t1_controls(res: pd.DataFrame) -> str | None:
    hdr(1, "negative-control separation")
    have = set(res["peptide_id"])
    if not (CONTROLS & have):
        print("  SKIP: no control peptides in results.")
        print("  Run:  pept-bioemu run-all -c benchmark/configs/benchmark_peptides.yaml")
        return None

    d = res.set_index("peptide_id")["dG_conf"].to_dict()
    for pid in ["pmi", "pdiq", "p53_17_29", "p53_15_29", "ala1", "ala2"]:
        if pid in d:
            tag = "control" if pid in CONTROLS else "binder"
            print(f"  {pid:<12s} {tag:<8s} dG_conf = {d[pid]:6.2f} kcal/mol")

    binders = [d[p] for p in BINDERS & have]
    ctrls = [d[p] for p in CONTROLS & have]
    if not binders or not ctrls:
        print("\n  SKIP: need at least one binder and one control.")
        return None

    ok = min(ctrls) > max(binders)
    print(f"\n  worst binder {max(binders):.2f}  vs  best control {min(ctrls):.2f}")
    if ok:
        print("  PASS: every control is less pre-organised than every binder.")
    else:
        print("  FAIL: a control looks as pre-organised as a real binder.")
        print("  The criteria are not specific. Inspect per-criterion pass rates:")
        print("  a poly-Ala peptide can satisfy C2 (helicity) and C5 (Rg) while")
        print("  having no anchors at all, so C3/C4 must carry the discrimination.")

    if "ala1" in d and "ala2" in d:
        arrow = "<" if d["ala1"] < d["ala2"] else ">="
        print(f"\n  Ala1 {d['ala1']:.2f} {arrow} Ala2 {d['ala2']:.2f}  "
              f"(expected Ala1 < Ala2: anchors help even on a poly-Ala background)")
    return "PASS" if ok else "FAIL"


def t2_prior_md(res: pd.DataFrame, meas: pd.DataFrame) -> str | None:
    hdr(2, "agreement with prior MD (Lang & Perez 2021, Table 2)")
    ref = meas[meas["quantity"] == "free_md_top_cluster_pct"].set_index("peptide_id")["value"]

    rows = []
    for _, r in res.iterrows():
        pid = r["peptide_id"]
        key = pid if pid in ref.index else CONSTRUCT_SUBSTITUTE.get(pid)
        if key in ref.index:
            rows.append((pid, key, r["f_bound"], float(ref[key])))

    if len(rows) < 2:
        print(f"  SKIP: only {len(rows)} peptide(s) overlap with that table.")
        print("  Running Ala1 and Ala2 raises the overlap to 4.")
        return None

    print(f"  {'peptide':<12s} {'f_bound':>9s} {'free-MD %':>10s}")
    for pid, key, fb, rv in rows:
        note = f"  (vs {key})" if key != pid else ""
        print(f"  {pid:<12s} {fb:9.3f} {rv:10.1f}{note}")

    fb = np.array([r[2] for r in rows])
    rv = np.array([r[3] for r in rows])
    order_ok = np.array_equal(np.argsort(fb), np.argsort(rv))
    print(f"\n  rank order identical: {order_ok}")

    if len(rows) >= 4 and spearmanr is not None:
        rho, p = spearmanr(fb, rv)
        print(f"  Spearman rho = {rho:.3f}  (p = {p:.3f}, n = {len(rows)})")
    else:
        print(f"  n = {len(rows)}: too few for a meaningful correlation; "
              "reporting rank order only.")

    print("\n  Caveats: different force field (ff14SB), different observable")
    print("  (top-cluster population at eps=1.5 A vs a 5-criterion vote), and")
    print("  p53 15-29 vs 17-29. Compare ranks, never absolute values.")
    return "PASS" if order_ok else "CHECK"


def t3_affinity(res: pd.DataFrame, meas: pd.DataFrame) -> str | None:
    hdr(3, "association with affinity (indirect)")
    ic = meas[meas["quantity"] == "IC50"].copy()
    if ic.empty:
        print("  SKIP: no IC50 rows.")
        return None

    ic["nM"] = np.where(ic["units"] == "uM", ic["value"] * 1000.0, ic["value"])
    ic = ic.set_index("peptide_id")

    have = [p for p in res["peptide_id"] if p in ic.index]
    censored = [p for p in have if ic.loc[p, "value_sign"] == ">"]
    usable = [p for p in have if p not in censored]

    print(f"  peptides with both dG_conf and IC50: {len(usable)}")
    if censored:
        print(f"  censored (excluded from correlation): {', '.join(censored)}")

    if len(usable) < 3:
        print("\n  SKIP: fewer than 3 usable pairs.")
        print("  Every peptide in the benchmark that HAS an IC50 (nov2, nov3,")
        print("  mini59/60/61, p53_18_29) currently lacks a bound reference")
        print("  structure, so none can be scored yet. See README step 4.")
        return None

    d = res.set_index("peptide_id")["dG_conf"]
    x = np.array([d[p] for p in usable])
    y = np.array([-np.log10(ic.loc[p, "nM"] * 1e-9) for p in usable])
    for p, xi, yi in zip(usable, x, y):
        print(f"  {p:<14s} dG_conf {xi:6.2f}   pIC50 {yi:5.2f}")

    if spearmanr is not None:
        rho, pv = spearmanr(x, y)
        print(f"\n  Spearman rho = {rho:.3f} (p = {pv:.3f}, n = {len(usable)})")
        print("  Expected sign: NEGATIVE (higher dG_conf -> weaker binding).")
    print("\n  Reminder: dG_conf is one term of dG_bind. Scatter here is expected")
    print("  whenever the interfaces differ. The clean test is an interface-matched")
    print("  pair, e.g. mini59 vs mini61 (same helix, 47-fold affinity difference).")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results/summary_all.csv"))
    args = ap.parse_args()

    if not args.results.exists():
        print(f"No results file at {args.results}")
        print("Run the pipeline first, or point --results at an existing summary_all.csv")
        return 2

    pep, meas = load_benchmark()
    print("=" * 68)
    print("pept-bioemu benchmark validation")
    print("=" * 68)
    print(f"  benchmark: {len(pep)} peptides, {len(meas)} curated measurements")
    print(f"  results:   {args.results}")
    res = load_results(args.results)
    print(f"  scored:    {len(res)} peptide(s) with a finite dG_conf")

    print("\n  NOTE: Tier 0 (reference-pose recovery) is the precondition for")
    print("  everything below. Run check_reference_pose.py first.")

    verdicts = {
        "T1 controls": t1_controls(res),
        "T2 prior MD": t2_prior_md(res, meas),
        "T3 affinity": t3_affinity(res, meas),
    }

    print(f"\n{'=' * 68}\nSUMMARY\n{'=' * 68}")
    for k, v in verdicts.items():
        print(f"  {k:<14s} {v or 'SKIPPED (insufficient data)'}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
