"""
Multi-criteria bound-like conformation classifier for MDM2-binding peptides.

Fixes applied (see accompanying review):
  1. Frames are now superposed onto the bound reference peptide before anchor
     distances are computed, so distances are measured in the same coordinate
     frame the pocket centers were extracted in (extract_pocket_centers.py).
  2. phi/psi are now read from the correct return value of md.compute_phi/psi
     (previously the atom-index array was used where the angle array should
     have been), and residue -> column mapping is derived from the CA atom in
     each dihedral quartet instead of assumed positional offsets.
  3. bound_chain now accepts either a numeric chain index or a PDB chain
     letter (e.g. "B"), resolved to the integer index MDTraj's `chainid`
     selection requires.
  4. groove_residues is documented as informational only; the alignment
     already places the peptide in the correct 1YCR-derived coordinate frame
     via its own backbone, so no separate groove-based alignment is needed.
  5. classify_trajectory is vectorized (RMSD/DSSP/phi-psi/Rg computed once
     for the whole trajectory instead of frame-by-frame), anchor-distance
     units are corrected (nm -> Å before comparing to pocket_centers, which
     are in Å), threshold_sweep reuses those arrays instead of recomputing,
     and the composite min-criteria sweep reuses the already-computed
     per-frame criteria matrix instead of rerunning the whole classifier.
"""

import logging
import numpy as np
import mdtraj as md
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from pept_bioemu.pipeline.flexe_score import EDENMHamiltonian

# Module-level logger
logger = logging.getLogger(__name__)


@dataclass
class PeptideConfig:
    """Configuration for a specific peptide system."""
    name: str
    sequence: str
    bound_pdb_path: str
    bound_chain: str                   # PDB chain letter (e.g. "B") or numeric chain index
    anchor_positions: List[int]        # 0-indexed residue indices of Phe/Trp/Leu anchors
    pocket_centers: np.ndarray         # Shape (3, 3), Å - MDM2 pocket center coords per anchor
    groove_residues: List[int]         # Groove/binding site residue indices (informational)
    is_stapled: bool = False
    is_macrocycle: bool = False
    ref_residue_range: Optional[List[int]] = None  # 1-indexed inclusive [start, end] for reference PDB
    ref_residue_slice: Optional[slice] = None      # 0-indexed slice for Python (computed from range)


@dataclass
class FrameMetrics:
    """Per-frame computed metrics."""
    rmsd: float                        # Å
    helix_fraction: float
    helix_per_residue: np.ndarray      # Shape (n_residues,)
    anchor_distances: np.ndarray       # Shape (3,), Å - min dist per anchor to pocket
    phi_psi_anchors: np.ndarray        # Shape (3, 2), degrees - (phi, psi) per anchor, NaN if undefined
    rg: float                          # Å
    flexe_energy: float                # kcal/mol/residue - see flexe_score.py


@dataclass
class BoundLikeCriteria:
    """Thresholds for each criterion (adjustable)."""
    rmsd_threshold: float = 2.0        # Å
    helix_threshold: float = 0.50      # fraction
    # Per-anchor thresholds (Phe-pocket, Trp-pocket, Leu-pocket), Å - see the
    # root bound_like_classifier.py for the calibration (12 native bound
    # reference structures, CB-atom distances, ceil(max observed)+1A each).
    anchor_dist_thresholds: Tuple[float, float, float] = (6.0, 8.0, 10.0)  # Å
    phi_psi_tolerance: float = 30.0    # degrees from ideal (-60, -45)
    rg_threshold: float = 7.5          # Å
    # ~ThermE-per-residue (0.7-0.8 kcal/mol/res for these peptide sizes, see
    # EDENMHamiltonian.thermal_energy_per_residue) plus one kBT of headroom -
    # a starting point, meant to be refined via threshold_sweep() like the
    # other four distance/fraction thresholds, not a settled value.
    flexe_threshold: float = 1.5       # kcal/mol/residue
    # Two-tier composite rule (replaces a flat "N of 6" vote): RMSD, helix,
    # Rg and FlexE all measure overlapping "generically native-shaped"
    # information, while only anchor-distance and phi/psi directly test the
    # binding-specific side-chain-in-pocket geometry. A flat vote lets a
    # frame pass purely on shape criteria without ever showing docking
    # evidence; requiring >=1 of the 2 docking criteria in addition to a
    # majority of the 4 shape criteria was validated against the actual
    # WT_p53/PDIQ/PMI data (see threshold-sweep discussion) to preserve the
    # same peptide ranking while barely disturbing f_bound/ΔG_conf, unlike
    # requiring both docking criteria at once (collapses f_bound to <1%).
    shape_min_pass: int = 2            # >= N of 4 shape criteria (RMSD, helix, Rg, FlexE)
    dock_min_pass: int = 1             # >= N of 2 docking criteria (anchor distance, phi/psi)


# Column indices into the (n_frames, 6) criteria-passed matrix, grouped by
# what they actually test (see BoundLikeCriteria docstring above). Column
# order: 0=RMSD, 1=Helix, 2=Anchor, 3=phi/psi, 4=Rg, 5=FlexE.
_SHAPE_CRITERIA_COLS = (0, 1, 4, 5)   # RMSD, Helix, Rg, FlexE
_DOCK_CRITERIA_COLS = (2, 3)          # Anchor distance, phi/psi


def _resolve_chain_index(topology, chain_spec) -> int:
    """
    Resolve a bound_chain spec from peptides.yaml (e.g. "B" or "1") to the
    0-indexed chain index MDTraj's `chainid` selection expects.

    Numeric strings/ints are used directly. Single-letter chain IDs are
    mapped assuming chains appear in the file in alphabetical order
    (A -> 0, B -> 1, ...), which holds for 1YCR (MDM2=A=0, p53 peptide=B=1)
    but should be spot-checked for structures with non-standard chain
    ordering (print topology.chains and compare against the PDB header).
    """
    s = str(chain_spec).strip()
    if s.lstrip('-').isdigit():
        idx = int(s)
    else:
        idx = ord(s.upper()) - ord('A')
    if not (0 <= idx < topology.n_chains):
        raise ValueError(
            f"Cannot resolve bound_chain '{chain_spec}' to a valid chain index "
            f"(resolved to {idx}, but topology has {topology.n_chains} chains). "
            f"Print topology.chains to check the actual ordering in this PDB."
        )
    return idx


def _phi_psi_column_map(topology, atom_index_quartets) -> Dict[int, int]:
    """
    Map residue.index -> column index into the (n_frames, n_dihedrals) angle
    array returned alongside `atom_index_quartets` by md.compute_phi/compute_psi.

    Every phi/psi dihedral quartet contains exactly one CA atom, and that CA
    always belongs to the residue the dihedral "belongs to" - this is more
    robust than assuming a fixed positional offset between residue index and
    column index (compute_phi skips the first residue, compute_psi skips the
    last, so a naive `column == res_idx` mapping is wrong).
    """
    mapping: Dict[int, int] = {}
    for col, atom_idxs in enumerate(atom_index_quartets):
        for a in atom_idxs:
            atom = topology.atom(a)
            if atom.name == 'CA':
                mapping[atom.residue.index] = col
                break
    return mapping


class BoundLikeClassifier:
    """
    Classifies frames as bound-like using composite criteria.
    """

    def __init__(self, config: PeptideConfig, criteria: Optional[BoundLikeCriteria] = None):
        self.config = config
        self.criteria = criteria or BoundLikeCriteria()
        self._ref_peptide = None
        self._mdm2_groove_indices = None
        self._anchor_atom_indices = None  # Cβ (or Cγ for aromatics) per anchor
        self._pocket_centers = None
        self._peptide_ca_indices = None
        self._ref_ca_indices = None
        self._traj_topology = None
        self._bb_indices = None           # backbone atom indices, cached on first use
        self._flexe_hamiltonian = None    # built once from the reference peptide in prepare()

    def prepare(self, topology_path: str, bound_ref_path: Optional[str] = None):
        """Load reference and compute fixed indices."""
        # Load bound reference peptide (for RMSD reference coordinates and
        # for anchor-distance alignment). This is loaded from the original
        # bound-complex PDB, so its coordinates are already in the same
        # frame that extract_pocket_centers.py used to compute pocket_centers -
        # atom_slice() below only subsets atoms, it never moves coordinates.
        ref_path = bound_ref_path or self.config.bound_pdb_path
        self._ref_peptide = md.load(ref_path)

        # Select bound chain if specified. bound_chain may be a PDB letter
        # ("B") or a numeric chain index ("1") - resolved to an index here.
        if self.config.bound_chain not in (None, ""):
            chain_idx = _resolve_chain_index(self._ref_peptide.topology, self.config.bound_chain)
            chain_mask = self._ref_peptide.topology.select(f"chainid {chain_idx}")
            if len(chain_mask) == 0:
                raise ValueError(
                    f"bound_chain '{self.config.bound_chain}' (chain index {chain_idx}) "
                    f"selected 0 atoms in {ref_path}"
                )
            self._ref_peptide = self._ref_peptide.atom_slice(chain_mask)

        # Apply reference residue range selection if specified.
        # This allows trimming the reference (e.g., to exclude disordered termini
        # that are missing in the crystal structure but present in the simulation).
        if self.config.ref_residue_slice is not None:
            ref_n_res = self._ref_peptide.n_residues
            # Get the actual indices that will be selected
            ref_res_indices = list(range(*self.config.ref_residue_slice.indices(ref_n_res)))
            if not ref_res_indices:
                raise ValueError(
                    f"ref_residue_range {self.config.ref_residue_range} resulted in empty selection "
                    f"for reference with {ref_n_res} residues"
                )
            # Select atoms belonging to these residues
            atom_mask = self._ref_peptide.topology.select(f"resid {' '.join(map(str, ref_res_indices))}")
            if len(atom_mask) == 0:
                raise ValueError(
                    f"ref_residue_range {self.config.ref_residue_range} selected 0 atoms "
                    f"from reference chain"
                )
            self._ref_peptide = self._ref_peptide.atom_slice(atom_mask)
            logger.info(
                f"[{self.config.name}] Applied ref_residue_range {self.config.ref_residue_range} "
                f"(1-indexed inclusive) -> using {len(ref_res_indices)} residues "
                f"(indices {ref_res_indices[0]}-{ref_res_indices[-1]} 0-indexed) "
                f"for RMSD/superposition"
            )
            # Warn if any anchor positions fall outside the trimmed reference
            ref_len = len(ref_res_indices)
            for ap in self.config.anchor_positions:
                if ap >= ref_len:
                    logger.warning(
                        f"[{self.config.name}] Anchor position {ap+1} (0-indexed {ap}) "
                        f"falls outside trimmed reference length ({ref_len} residues). "
                        f"Anchor distances/phi-psi for this anchor may fail or be inaccurate."
                    )

        # Load trajectory topology (for atom indexing during frame processing)
        self._traj_topology = md.load(topology_path).topology

        # Peptide Cα indices for RMSD/superposition.
        # atom_indices (used on the moving `traj`) must come from the SAMPLED
        # trajectory's own topology, NOT the reference — the two structures have
        # different atom counts (BioEmu output lacks full side-chain atoms that
        # the 1YCR crystal structure has), so borrowing indices from one topology
        # and using them on the other silently points at the wrong (or nonexistent) atoms.
        # If ref_residue_slice is specified, we only use the corresponding first N CA atoms
        # from the trajectory to match the trimmed reference.
        traj_ca_indices = self._traj_topology.select("name CA")
        ref_ca_indices = self._ref_peptide.topology.select("name CA")
        
        if self.config.ref_residue_slice is not None:
            # Trim trajectory CA indices to match reference length
            n_ref_ca = len(ref_ca_indices)
            if len(traj_ca_indices) < n_ref_ca:
                logger.warning(
                    f"[{self.config.name}] Trajectory has fewer CA atoms ({len(traj_ca_indices)}) "
                    f"than reference ({n_ref_ca}) after trimming. Using all trajectory CA atoms."
                )
                self._peptide_ca_indices = traj_ca_indices
                self._ref_ca_indices = ref_ca_indices[:len(traj_ca_indices)]
            else:
                self._peptide_ca_indices = traj_ca_indices[:n_ref_ca]
                self._ref_ca_indices = ref_ca_indices
            logger.info(
                f"[{self.config.name}] Using {len(self._peptide_ca_indices)} CA atoms "
                f"from trajectory (first {len(self._peptide_ca_indices)} residues) "
                f"and {len(self._ref_ca_indices)} from reference for superposition/RMSD"
            )
        else:
            self._peptide_ca_indices = traj_ca_indices
            self._ref_ca_indices = ref_ca_indices

        # Informational only - see module docstring fix #4. Not currently
        # used for alignment: aligning the sampled peptide's own backbone
        # onto the bound reference peptide (which is already in the correct
        # 1YCR coordinate frame) is sufficient to make anchor_distances
        # meaningful, without needing MDM2 atoms present at analysis time.
        self._groove_indices = self.config.groove_residues

        # Anchor atom indices in TRAJECTORY topology (Cβ for aliphatic, Cγ for Phe/Trp)
        self._anchor_atom_indices = self._get_anchor_atom_indices_from_traj()

        # Pocket centers, Å, in the same coordinate frame as self._ref_peptide
        self._pocket_centers = self.config.pocket_centers

        self._bb_indices = None  # reset cache; resolved lazily against the working trajectory

        # FlexE elastic-network Hamiltonian, built ONCE from the reference
        # peptide's own Cα coordinates (Å) - the spring network never changes
        # between frames, only the candidate frame's distances do (see
        # flexe_score.py). Internal pairwise distances are rotation/
        # translation invariant, so this doesn't need (but isn't harmed by)
        # the superposition classify_trajectory does for the other criteria.
        if len(self._ref_ca_indices) > 0:
            ref_coords_A = self._ref_peptide.xyz[0, self._ref_ca_indices, :] * 10.0
            self._flexe_hamiltonian = EDENMHamiltonian(ref_coords_A)
        else:
            self._flexe_hamiltonian = None

    def _get_anchor_atom_indices_from_traj(self) -> List[List[int]]:
        """Get atom indices for anchor sidechains from trajectory topology (Cβ or Cγ)."""
        indices = []
        for res_idx in self.config.anchor_positions:
            if res_idx >= self._traj_topology.n_residues:
                indices.append([])
                continue
            res = list(self._traj_topology.residues)[res_idx]
            resname = res.name
            if resname in ('PHE', 'TRP', 'TYR'):
                # Use Cγ (first aromatic carbon) if available, else CB
                atoms = [a.index for a in res.atoms if a.name in ('CG', 'CD1', 'CD2', 'CB')]
            else:
                # Cβ for LEU, etc.
                atoms = [a.index for a in res.atoms if a.name == 'CB']
            indices.append(atoms)
        return indices

    # ------------------------------------------------------------------
    # Single-frame API (kept for compatibility / ad-hoc use). Not used by
    # classify_trajectory anymore, which is fully vectorized below.
    # ------------------------------------------------------------------

    def compute_frame_metrics(self, frame: md.Trajectory) -> FrameMetrics:
        """Compute all metrics for a single frame. Prefer classify_trajectory
        for anything bigger than a handful of frames - it's vectorized."""
        result = self.classify_trajectory(frame)
        return result['metrics'][0]

    def evaluate_criteria(self, metrics: FrameMetrics) -> Tuple[np.ndarray, bool]:
        """
        Evaluate all 6 criteria for a single frame's metrics.
        Returns: (criteria_passed[6], is_bound_like)
        """
        passed = np.zeros(6, dtype=bool)

        passed[0] = metrics.rmsd <= self.criteria.rmsd_threshold
        passed[1] = metrics.helix_fraction >= self.criteria.helix_threshold
        # >=2 of 3 anchors (not all 3): see bound_like_classifier.py at the
        # project root for the rationale (even the true bound-state reference
        # conformation fails all-3 at this threshold for every peptide tested).
        thresholds = np.array(self.criteria.anchor_dist_thresholds)
        passed[2] = np.sum(metrics.anchor_distances <= thresholds) >= 2

        phi, psi = metrics.phi_psi_anchors[:, 0], metrics.phi_psi_anchors[:, 1]
        # NaN (undefined phi/psi, e.g. terminal residue) safely evaluates to
        # False in these comparisons, so such anchors simply don't count
        # toward the alpha-region tally.
        in_alpha = (np.abs(phi + 60) <= self.criteria.phi_psi_tolerance) & \
                   (np.abs(psi + 45) <= self.criteria.phi_psi_tolerance)
        passed[3] = np.nansum(in_alpha) >= 2

        passed[4] = metrics.rg <= self.criteria.rg_threshold

        passed[5] = metrics.flexe_energy <= self.criteria.flexe_threshold

        n_shape_passed = passed[list(_SHAPE_CRITERIA_COLS)].sum()
        n_dock_passed = passed[list(_DOCK_CRITERIA_COLS)].sum()
        is_bound_like = (n_shape_passed >= self.criteria.shape_min_pass) and \
                         (n_dock_passed >= self.criteria.dock_min_pass)
        return passed, is_bound_like

    def _evaluate_criteria_arrays(
        self,
        rmsd_arr: np.ndarray,
        helix_frac_arr: np.ndarray,
        anchor_dists_arr: np.ndarray,
        phi_psi_arr: np.ndarray,
        rg_arr: np.ndarray,
        flexe_arr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized version of evaluate_criteria over a whole trajectory."""
        n_frames = len(rmsd_arr)
        passed = np.zeros((n_frames, 6), dtype=bool)

        passed[:, 0] = rmsd_arr <= self.criteria.rmsd_threshold
        passed[:, 1] = helix_frac_arr >= self.criteria.helix_threshold
        # >=2 of 3 anchors, not all 3 - see evaluate_criteria() for rationale.
        thresholds = np.array(self.criteria.anchor_dist_thresholds)
        passed[:, 2] = np.sum(anchor_dists_arr <= thresholds[np.newaxis, :], axis=1) >= 2

        phi = phi_psi_arr[:, :, 0]
        psi = phi_psi_arr[:, :, 1]
        with np.errstate(invalid='ignore'):
            in_alpha = (np.abs(phi + 60) <= self.criteria.phi_psi_tolerance) & \
                       (np.abs(psi + 45) <= self.criteria.phi_psi_tolerance)
        passed[:, 3] = np.nansum(in_alpha, axis=1) >= 2

        passed[:, 4] = rg_arr <= self.criteria.rg_threshold

        passed[:, 5] = flexe_arr <= self.criteria.flexe_threshold

        n_shape_passed = passed[:, list(_SHAPE_CRITERIA_COLS)].sum(axis=1)
        n_dock_passed = passed[:, list(_DOCK_CRITERIA_COLS)].sum(axis=1)
        is_bound_like = (n_shape_passed >= self.criteria.shape_min_pass) & \
                         (n_dock_passed >= self.criteria.dock_min_pass)
        return passed, is_bound_like

    # ------------------------------------------------------------------
    # Vectorized whole-trajectory API - use this one.
    # ------------------------------------------------------------------

    def classify_trajectory(self, traj: md.Trajectory) -> Dict[str, Any]:
        """
        Classify an entire trajectory in a vectorized way.
        Returns dict with arrays and summary statistics.
        """
        n_frames = traj.n_frames

        # Work on a copy so we never mutate the caller's trajectory in place.
        traj = traj[:]

        # --- Superpose onto the bound reference peptide FIRST. ---
        # Without this, anchor distances below are meaningless: BioEmu
        # samples an isolated peptide with an arbitrary global position and
        # orientation, while pocket_centers were measured in 1YCR's own
        # coordinate frame. Superposing puts every frame into that same frame.
        have_ca = len(self._peptide_ca_indices) > 0 and len(self._ref_ca_indices) > 0
        if have_ca:
            traj.superpose(
                self._ref_peptide, frame=0,
                atom_indices=self._peptide_ca_indices,
                ref_atom_indices=self._ref_ca_indices,
            )

        # --- RMSD (Å) ---
        if not have_ca:
            rmsd_arr = np.full(n_frames, np.inf)
        else:
            try:
                rmsd_arr = md.rmsd(
                    traj, self._ref_peptide,
                    atom_indices=self._peptide_ca_indices,
                    ref_atom_indices=self._ref_ca_indices,
                ) * 10.0  # nm -> Å
            except Exception:
                rmsd_arr = np.full(n_frames, np.inf)

        # --- Helicity (DSSP) ---
        try:
            dssp_arr = md.compute_dssp(traj, simplified=True)   # (n_frames, n_residues)
            helix_bool = (dssp_arr == 'H')
            helix_frac_arr = helix_bool.mean(axis=1)
            helix_per_res_arr = helix_bool.astype(float)
        except Exception:
            helix_frac_arr = np.zeros(n_frames)
            helix_per_res_arr = np.zeros((n_frames, traj.n_residues))

        # --- Anchor distances to pocket centers (Å) ---
        # traj.xyz is in nm (MDTraj convention); pocket_centers are in Å
        # (see extract_pocket_centers.py, which multiplies by 10 on the way
        # out) - convert coordinates to Å before comparing.
        n_anchors = len(self._anchor_atom_indices)
        anchor_dists_arr = np.full((n_frames, n_anchors), np.inf)
        for i, atom_idxs in enumerate(self._anchor_atom_indices):
            if len(atom_idxs) == 0 or i >= len(self._pocket_centers):
                continue
            coords_A = traj.xyz[:, atom_idxs, :] * 10.0             # (n_frames, n_atoms_i, 3), Å
            center = self._pocket_centers[i]                         # (3,), Å
            dists = np.linalg.norm(coords_A - center, axis=-1)       # (n_frames, n_atoms_i)
            anchor_dists_arr[:, i] = dists.min(axis=1)

        # --- phi/psi of anchor residues (degrees) ---
        phi_psi_arr = np.full((n_frames, n_anchors, 2), np.nan)
        try:
            phi_atom_idx, phi_angles = md.compute_phi(traj)   # angles: (n_frames, n_phi), radians
            psi_atom_idx, psi_angles = md.compute_psi(traj)
            phi_map = _phi_psi_column_map(traj.topology, phi_atom_idx)
            psi_map = _phi_psi_column_map(traj.topology, psi_atom_idx)
            for a_i, res_idx in enumerate(self.config.anchor_positions):
                phi_col = phi_map.get(res_idx)
                psi_col = psi_map.get(res_idx)
                if phi_col is not None:
                    phi_psi_arr[:, a_i, 0] = np.degrees(phi_angles[:, phi_col])
                if psi_col is not None:
                    phi_psi_arr[:, a_i, 1] = np.degrees(psi_angles[:, psi_col])
        except Exception:
            pass  # left as NaN; NaN safely fails the alpha-region check below

        # --- Radius of gyration (Å) ---
        try:
            if self._bb_indices is None:
                self._bb_indices = traj.topology.select("backbone")
            if len(self._bb_indices) > 0:
                rg_arr = md.compute_rg(traj.atom_slice(self._bb_indices)) * 10.0
            else:
                rg_arr = np.full(n_frames, np.inf)
        except Exception:
            rg_arr = np.full(n_frames, np.inf)

        # --- FlexE deformation energy (kcal/mol/residue) ---
        # Internal-coordinate-only, so it doesn't need the superposition
        # above - but reusing the already-superposed traj is harmless and
        # keeps this loop symmetric with the other criteria.
        if self._flexe_hamiltonian is not None and have_ca:
            try:
                cand_coords_A = traj.xyz[:, self._peptide_ca_indices, :] * 10.0
                flexe_arr = self._flexe_hamiltonian.flexe_per_residue(cand_coords_A)
            except Exception:
                flexe_arr = np.full(n_frames, np.inf)
        else:
            flexe_arr = np.full(n_frames, np.inf)

        # --- Criteria (cheap numpy ops - no more mdtraj calls in a loop) ---
        all_passed, all_bound = self._evaluate_criteria_arrays(
            rmsd_arr, helix_frac_arr, anchor_dists_arr, phi_psi_arr, rg_arr, flexe_arr
        )
        f_bound = all_bound.mean()

        # --- Package per-frame FrameMetrics for compatibility with analyze_all.py ---
        all_metrics = [
            FrameMetrics(
                rmsd=float(rmsd_arr[i]),
                helix_fraction=float(helix_frac_arr[i]),
                helix_per_residue=helix_per_res_arr[i],
                anchor_distances=anchor_dists_arr[i],
                phi_psi_anchors=phi_psi_arr[i],
                rg=float(rg_arr[i]),
                flexe_energy=float(flexe_arr[i]),
            )
            for i in range(n_frames)
        ]

        return {
            'n_frames': n_frames,
            'f_bound': f_bound,
            'criteria_passed_per_frame': all_passed,
            'is_bound_like': all_bound,
            'metrics': all_metrics,
            'criteria_thresholds': self.criteria,
            # Raw arrays, kept so threshold_sweep() can reuse this work
            # instead of recomputing everything from scratch.
            '_rmsd_arr': rmsd_arr,
            '_helix_frac_arr': helix_frac_arr,
            '_anchor_dists_arr': anchor_dists_arr,
            '_rg_arr': rg_arr,
            '_flexe_arr': flexe_arr,
        }

    def bootstrap_dg_conf(self, result: Dict[str, Any], n_bootstrap: int = 1000,
                          rt: float = 0.5962) -> Tuple[float, float, float]:
        """Bootstrap 95% CI for ΔG_conf."""
        f_bound = result['f_bound']
        n = result['n_frames']

        # Edge cases: f_bound = 0 or 1 -> ΔG_conf = ±inf, CI undefined
        if f_bound <= 0.0 or f_bound >= 1.0:
            return np.inf, np.inf, np.inf

        boot_f = np.random.binomial(n, f_bound, n_bootstrap) / n
        boot_f = np.clip(boot_f, 1e-6, 1 - 1e-6)
        boot_dg = -rt * np.log(boot_f / (1 - boot_f))

        dg_point = -rt * np.log(f_bound / (1 - f_bound))
        ci_lower = np.percentile(boot_dg, 2.5)
        ci_upper = np.percentile(boot_dg, 97.5)

        return dg_point, ci_lower, ci_upper


def threshold_sweep(
    classifier: BoundLikeClassifier,
    traj: md.Trajectory,
    precomputed: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[float, float]]:
    """
    Sweep thresholds for sensitivity analysis.

    Pass `precomputed` (the dict returned by classify_trajectory) to reuse
    its RMSD/helix/anchor/Rg arrays and per-frame criteria matrix instead of
    recomputing everything from scratch. If omitted, this will call
    classify_trajectory(traj) itself.
    """
    result = precomputed if precomputed is not None else classifier.classify_trajectory(traj)

    rmsd_vals = result['_rmsd_arr']
    helix_vals = result['_helix_frac_arr']
    anchor_dists = result['_anchor_dists_arr'].max(axis=1)  # worst anchor per frame
    rg_vals = result['_rg_arr']
    flexe_vals = result['_flexe_arr']

    # Sweep ranges
    rmsd_threshs = np.arange(1.0, 4.5, 0.5)
    helix_threshs = np.arange(0.3, 0.8, 0.1)
    anchor_threshs = np.arange(3.0, 7.0, 0.5)
    rg_threshs = np.arange(6.0, 10.0, 0.5)
    flexe_threshs = np.arange(0.5, 3.25, 0.25)

    results: Dict[str, Dict[float, float]] = {}

    results['rmsd'] = {thr: float((rmsd_vals < thr).mean()) for thr in rmsd_threshs}
    results['helix'] = {thr: float((helix_vals >= thr).mean()) for thr in helix_threshs}
    results['anchor'] = {thr: float((anchor_dists <= thr).mean()) for thr in anchor_threshs}
    results['rg'] = {thr: float((rg_vals <= thr).mean()) for thr in rg_threshs}
    results['flexe'] = {thr: float((flexe_vals <= thr).mean()) for thr in flexe_threshs}

    # Composite sweep (vary shape_min_pass / dock_min_pass) - reuse the
    # already-computed per-frame criteria matrix instead of rerunning
    # classify_trajectory for every combination.
    passed = result['criteria_passed_per_frame']   # (n_frames, 6) bool
    n_shape_passed = passed[:, list(_SHAPE_CRITERIA_COLS)].sum(axis=1)  # 0-4
    n_dock_passed = passed[:, list(_DOCK_CRITERIA_COLS)].sum(axis=1)    # 0-2
    for shape_min in [1, 2, 3, 4]:
        for dock_min in [0, 1, 2]:
            f_bound = float(((n_shape_passed >= shape_min) & (n_dock_passed >= dock_min)).mean())
            results[f'composite_shape{shape_min}_dock{dock_min}'] = {float(shape_min): f_bound}

    return results