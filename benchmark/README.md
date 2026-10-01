# pept-bioemu MDM2 benchmark

A curated dataset for validating the ΔG_conf pipeline, built from the papers in
`papers/` and from the crystal structures already in the project.

```
benchmark/
├── README.md                    this file
├── data/
│   ├── peptides.csv             15 peptides: identity, anchors, runnability
│   ├── measurements.csv         17 curated values + 5 flagged gaps, with provenance
│   └── sources.yaml             source registry (4 local papers + 3 to obtain)
├── configs/
│   └── benchmark_peptides.yaml  the entries that can be run today
├── check_reference_pose.py      Tier 0 — needs no sampling
└── validate.py                  Tiers 1–3 against results/summary_all.csv
```

---

## 1. What this benchmark has to validate

The thing being tested is **not** binding affinity. It is:

```
ΔG_conf = −RT · ln( f_bound / (1 − f_bound) )
```

the free energy cost of pre-organising the **free** peptide into its bound
shape. That is one additive term in the binding free energy:

```
ΔG_bind = ΔG_conf + ΔG_interface + ΔG_solvation + …
```

This has a direct consequence for dataset design. **Correlating ΔG_conf against
Kd or IC50 across peptides with different interfaces is not a valid test.** Such
a correlation mixes in terms the pipeline never models, so a weak result would
not be evidence against the method, and a strong result could be luck. Affinity
is therefore the *weakest* tier here, not the headline.

The strongest evidence is anything that constrains the **free-state ensemble
directly**. Hence the tier ordering below.

---

## 2. Validation tiers

### Tier 0 — Reference-pose recovery (no sampling needed, run this first)

Score the experimental bound conformation itself. It is bound-like by
definition, so it must be classified bound-like. If it is not, the true answer
cannot reach the target, `f_bound` is an underestimate, and every ΔG_conf built
on it is biased high.

```bash
python benchmark/check_reference_pose.py -c config/peptides.yaml --sweep
```

This is the cheapest and strictest test in the whole benchmark, and it is
currently expected to **fail** — see Finding 2.

### Tier 1 — Negative-control separation

Lang & Perez 2021 designed two poly-alanine controls for exactly this purpose:

| Peptide | Sequence | Has anchors? | Expected |
|---|---|---|---|
| Ala1 | `AAFAAAWAALAA` | F/W/L on poly-Ala | high ΔG_conf |
| Ala2 | `AAAAAAAAAAAA` | none | highest ΔG_conf |

Both are runnable today (`configs/benchmark_peptides.yaml`). Ala2 is a
particularly good adversarial case: poly-alanine is intrinsically helical, so it
can satisfy C2 (helicity) and C5 (Rg) while having no anchors whatsoever. All
discrimination must come from C3 and C4. If Ala2 scores as pre-organised as PMI,
the criteria are not specific.

### Tier 2 — Agreement with prior MD

Lang & Perez 2021 Table 2 reports free-peptide top-cluster populations from
unrestrained MD for five peptides — the same lab, the same system, an
independent method:

| Peptide | Free-MD top cluster (%) | pept-bioemu f_bound |
|---|---|---|
| p53 (15–29) | 0.6 | 0.113 *(17–29 construct)* |
| pDIQ | 24.0 | 0.388 |
| Ala1 | 1.4 | not yet run |
| Ala2 | 0.2 | not yet run |
| ATSP-7041 | 69.5 | not runnable (staple) |

Only two points overlap today, and they rank correctly. Running the two controls
raises this to four. **Compare ranks, not absolute values** — the observables are
different (a top-cluster population at ε = 1.5 Å vs a five-criterion vote) and
ff14SB has its own helical bias.

### Tier 3 — Association with affinity (indirect)

Use only where the interface is matched. The one clean pair available:

| Peptide | Sequence | IC50 |
|---|---|---|
| mini59 | `GPSQPTYP` GDDAPVEDLIRF KFLLQWYLLALTRHRYAAAC | 3.2 µM |
| mini61 | `GKSWMTVP` GDDAPVEDLIRF KFLLQWYLLALTRHRYAAAC | > 150 µM |

Identical helix and identical anchor triad; only the polyproline-II segment
differs, which is what holds the aPP scaffold folded. A ≥ 47-fold affinity
difference with a matched interface should be almost entirely conformational.
**If ΔG_conf cannot separate this pair, it is not measuring what it claims to.**

These are 40-residue miniproteins with no bound structure, so they need a model
before they can be scored — see step 4.

### Tier A — the gap worth closing (not yet populated)

The most direct validation is **experimental free-state helicity** (CD
spectroscopy, or NMR chemical shifts) compared against BioEmu's helix fraction.
That tests the ensemble itself, with no classifier and no thresholds in between.
No CD data appears in the four local PDFs. Rows `M020`–`M022` in
`measurements.csv` are placeholders marked `needs_source`.

---

## 3. Findings from building this

### Finding 1 — PDIQ anchor positions are wrong in `config/peptides.yaml`

`config/peptides.yaml` sets `anchor_positions: [3, 6, 10]` for PDIQ. Measured
directly from 3JZS (side-chain heavy atoms to pocket centres derived from that
file's own MDM2 chain):

| Residue | → pocket 1 | → pocket 2 | → pocket 3 |
|---|---|---|---|
| Phe3 | **3.88** | 4.80 | 13.56 |
| **Trp6** (configured) | 7.48 | **8.88** | 14.96 |
| **Trp7** (correct) | 8.21 | **2.35** | 8.27 |
| Leu10 | 11.57 | 5.83 | **6.94** |

PDIQ has two adjacent tryptophans; the one in the pocket is **W7**, not W6. The
correct triad is `[3, 7, 10]`, matching the i, i+4, i+7 spacing that WT p53 and
PMI both use (both verified — their W sits 2.25–2.30 Å from the pocket centre).

**Effect:** C3 for PDIQ is measured against a side chain ~8.9 Å from its pocket
instead of ~2.4 Å, so C3 fails far more often than it should. The published
PDIQ value of `f_bound = 0.388` is an **underestimate**, and ΔG_conf = 0.27
kcal/mol is correspondingly too high. Corrected in
`configs/benchmark_peptides.yaml`; `config/peptides.yaml` is unchanged — your call.

### Finding 2 — the Leu anchor never reaches pocket 3, in any crystal structure

Same measurement across all three bound structures:

| | Phe → P1 | Trp → P2 | Leu → P3 |
|---|---|---|---|
| WT p53 (1YCR) | 4.26 | 2.30 | **7.36** |
| PMI (3EQS) | 4.39 | 2.25 | **6.52** |
| PDIQ (3JZS) | 3.88 | 2.35 | **6.94** |

With `anchor_dist_threshold = 4.5 Å`, the Leu anchor fails in **every
experimental bound pose**, and Phe only barely passes. This is why
`bound_like_classifier.py` needs its "≥ 2 of 3" rule — the comment in the code
says as much, and these numbers are the reason.

The likely cause is the pocket-3 definition in `extract_pocket_centers.py`:

```python
pocket3_res = mdm2.topology.select("resSeq 100 101 53 ...")   # Leu26 pocket
```

Three residues, versus eight for pocket 1 and six for pocket 2. A centroid over
three residues sits off to one side of the real cleft. Two things to try, and
`check_reference_pose.py --sweep` reports the cut-off each pose would need:

1. **Widen the pocket-3 residue list** so its centroid lands in the cleft.
   Candidates from the p53–MDM2 literature: Leu54, Val93, His96, Ile99, Ile103.
2. **Raise the threshold** to whatever makes all three crystal poses pass.

Option 1 is preferable — it fixes the geometry rather than loosening the test.
Note that the existing `results/summary_all.csv` was produced with a 6.0 Å
cut-off, which is roughly what option 2 would land on.

**Why this matters for the whole project:** calibrating against the crystal pose
is what makes ΔG_conf an absolute number rather than an arbitrary score. Until
Tier 0 passes, ΔG_conf values are usable for *ranking* peptides but should not
be quoted as physical free energies.

### Finding 3 — construct mismatch with the prior study

Lang & Perez used p53 **15–29** (`SQETFSDLWKLLPEN`, anchors at 5/9/12);
`config/peptides.yaml` uses **17–29** (`ETFSDLWKLLPEN`, anchors at 3/7/10).
1YCR only resolves 17–29, so the two extra N-terminal residues have no reference
to align to. Running both would quantify the construct sensitivity; a commented
entry is in `configs/benchmark_peptides.yaml` with the alignment decision left
open rather than silently guessed.

### Finding 4 — chain indexing is correct, by a narrow margin

`bound_chain: "1"` for PDIQ resolves through `_resolve_chain_index` to MDTraj
chain index 1. In 3JZS the chain order is `A` (MDM2) then `P` (peptide), so
index 1 is indeed the peptide. This works, but it depends on file ordering
rather than on the chain ID `P`. Worth a comment in the config.

---

## 4. Suggested order of work

1. **Run Tier 0.** `python benchmark/check_reference_pose.py -c config/peptides.yaml --sweep`.
   No sampling needed. Fix the pocket-3 definition until all three poses pass.
2. **Fix the PDIQ anchors** (Finding 1) and re-run the analysis stage only —
   the ensembles do not need regenerating:
   `pept-bioemu analyze -c config/peptides.yaml`
3. **Run the two controls.** `pept-bioemu run-all -c benchmark/configs/benchmark_peptides.yaml`.
   This is the highest-value new sampling: it turns Tier 1 on and takes Tier 2
   from n = 2 to n = 4.
4. **Decide on the miniproteins.** mini59/mini61 is the best interface-matched
   test available, but needs a bound model (graft the helix onto 1YCR chain B,
   or use a docking/folding model) and 40-residue BioEmu runs. Worth it — this
   pair is the closest thing to a direct experimental test of ΔG_conf.
5. **Close the Tier A gap.** Find CD helicity for p53, pDI/pDIQ and PMI free
   peptides. This is the only tier that tests the ensemble without the
   classifier in the way.

---

## 5. Data provenance and honesty notes

Every row in `measurements.csv` carries `source_id`, `locator`, `extraction`
and `confidence`.

- `extraction = text` — taken from the PDF text layer.
- `extraction = figure_read` — **read from a rendered page image.** Murray &
  Gellman Figure 3 draws its peptides as chemical structures, so the sequences
  and IC50 values were read visually. The three Figure 3 values pass an internal
  cross-check (the text states peptide 2 is 28-fold better than peptide 1;
  8673 / 313 = 27.7), which is why they are marked high confidence. `nov3` is
  marked medium — verify its sequence against the primary reference.
- `status = needs_source` — **deliberately empty.** PMI and pDIQ affinities are
  well known but do not appear in the local PDFs, and are not filled in from
  memory. Cite `pazgier2009` / `phan2010` once you have them.

Murray & Gellman is a **review**; its values are secondary. Trace them to the
primary references (its ref 46 for Figure 3, ref 130 for Table I) before
anything goes into a paper.

IC50 is assay-dependent. Figure 3 values are competition ELISA, Table I values
are fluorescence polarisation. They are not directly comparable to each other,
and `validate.py` does not pool them.
