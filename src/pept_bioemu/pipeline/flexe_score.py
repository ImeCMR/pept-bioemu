"""
FlexE: elastic-network-based deformation energy between two conformations.

Ported from the original FlexE implementation (Perez, Yang, Bahar, Dill,
MacCallum, "FlexE: Using Elastic Network Models to Compare Models of Protein
Structure", J. Chem. Theory Comput. 2012, 8, 3985-3991;
code: https://github.com/laufercenter/FlexE, hamiltonian.py/EDENMHamiltonian).

This port replaces the original's ProDy-based coordinate loading with
mdtraj (already used throughout this project - see bound_like_classifier.py)
and its numpy.matlib-based distance-matrix helper (deprecated) with a direct
broadcasting implementation. The physical model itself - the ED-ENM spring
network of Orellana, Rueda, Ferrer-Costa, Lopez-Blanco, Chacon, Orozco,
J. Chem. Theory Comput. 2010, 6, 2910-2923 - is reproduced exactly, including
two details that are in the original code but NOT spelled out in the FlexE
paper's own text (confirmed by reading the actual published hamiltonian.py):

  1. The spatial cutoff is NOT a flat 12 A. It's size-adaptive:
     cutoff = max(8.0, 2.9*ln(n_res) - 2.9), which floors at 8.0 A for any
     protein/peptide under ~43 residues - i.e. always exactly 8.0 A for the
     12-13 residue peptides in this project, not the 12 A the paper's prose
     mentions (that value applies only to the paper's larger CASP/protein-
     pair test systems).
  2. The spatial (non-sequence-local) spring term clamps the reference
     distance at a 3.8 A floor before raising it to the (Ccart/d)^Ex power,
     to avoid a numerically unstable/enormous spring constant if two
     non-adjacent residues happen to sit unusually close together in the
     reference structure.

All distances handled by this module are in Angstrom. mdtraj's own
coordinates are in nm, so callers must convert with `* 10.0` before use here
- exactly as the rest of this codebase already does for RMSD/anchor
distances in bound_like_classifier.py.
"""

import numpy as np

# ED-ENM spring-constant parameters (Orellana et al. 2010; identical to the
# published FlexE tool's hamiltonian.py). Not intended to be tuned per-peptide.
_SEQ_LOCAL_LIMIT = 3       # sequence separation |i-j| <= this uses the sequence-local term
_SEQ_LOCAL_CONST = 60.0    # Cseq
_SPATIAL_CONST = 6.0       # Ccart
_SPATIAL_EXPONENT = 6      # Ex
_SPATIAL_DIST_FLOOR = 3.8  # Angstrom; prevents (Ccart/d)^Ex from blowing up

# Empirical rescaling of all spring constants so absolute FlexE energies
# match explicit-solvent MD deformation energies (Perez et al. 2012, Sec 2.3).
_FLEXE_MD_SCALING = 0.40

# Size-adaptive spatial cutoff: cutoff = max(_CUTOFF_FLOOR, _CUTOFF_SCALE*ln(N) - _CUTOFF_OFFSET)
_CUTOFF_SCALE = 2.9
_CUTOFF_OFFSET = 2.9
_CUTOFF_FLOOR = 8.0

_KB_KCAL_PER_MOL_K = 0.0019872041  # Boltzmann constant, kcal/(mol*K)


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """
    Pairwise Euclidean distances, batched over any number of leading dims.

    Args:
        coords: (..., N, 3) coordinates, Angstrom.
    Returns:
        (..., N, N) distance matrix.
    """
    diff = coords[..., :, None, :] - coords[..., None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _calc_cutoff(n_res: int) -> float:
    """Size-adaptive spatial cutoff (Angstrom), floored at 8.0 A."""
    cutoff = _CUTOFF_SCALE * np.log(n_res) - _CUTOFF_OFFSET
    return max(cutoff, _CUTOFF_FLOOR)


class EDENMHamiltonian:
    """
    Elastic network model (ED-ENM) spring network built from ONE reference
    conformation.

    Build once from the bound-reference peptide's Ca coordinates, then call
    `evaluate_energy`/`flexe_per_residue` for any number of candidate
    conformations (e.g. every frame of a BioEmu trajectory) - the spring
    network (which pairs are connected, at what strength, at what rest
    length) never changes; only the candidate's own current distances do.
    """

    def __init__(self, reference_coords_A: np.ndarray):
        """
        Args:
            reference_coords_A: (N, 3) Ca coordinates of the reference
                (bound-state) conformation, Angstrom.
        """
        reference_coords_A = np.asarray(reference_coords_A, dtype=float)
        self.n_res = reference_coords_A.shape[0]
        self.cutoff = _calc_cutoff(self.n_res)
        # Rest lengths d_ij^0 (Å) - the reference structure's own pairwise distances.
        self.dist_matrix_ref = _pairwise_distances(reference_coords_A)
        self.k_matrix = self._build_k_matrix()

    def _build_k_matrix(self) -> np.ndarray:
        n = self.n_res
        idx = np.arange(n)
        seq_sep = np.abs(idx[:, None] - idx[None, :])  # (N, N), |i-j|

        k = np.zeros((n, n))

        seq_local = (seq_sep >= 1) & (seq_sep <= _SEQ_LOCAL_LIMIT)
        k[seq_local] = _SEQ_LOCAL_CONST / (seq_sep[seq_local].astype(float) ** 2)

        spatial_candidate = seq_sep > _SEQ_LOCAL_LIMIT
        within_cutoff = spatial_candidate & (self.dist_matrix_ref <= self.cutoff)
        d_floored = np.maximum(self.dist_matrix_ref, _SPATIAL_DIST_FLOOR)
        k[within_cutoff] = (_SPATIAL_CONST / d_floored[within_cutoff]) ** _SPATIAL_EXPONENT
        # Pairs beyond cutoff, and the diagonal (seq_sep == 0), stay at 0.0 -
        # i.e. no spring, matching the original loop which never assigns them.

        return k

    def evaluate_energy(self, coords_A: np.ndarray) -> np.ndarray:
        """
        Raw (not-yet-per-residue) deformation energy of one or more candidate
        conformations against this Hamiltonian's fixed reference spring network.

        Args:
            coords_A: (N, 3) or (n_frames, N, 3) Ca coordinates, Angstrom, in
                the SAME residue order as the reference used to build this
                Hamiltonian. No superposition is required or used - only
                internal pairwise distances enter the calculation, which are
                already rotation/translation invariant.
        Returns:
            Total deformation energy in kcal/mol: scalar if coords_A was
            (N,3), else shape (n_frames,). Divide by self.n_res for the
            per-residue score the original paper/tool reports (or just call
            `flexe_per_residue` below).
        """
        coords_A = np.asarray(coords_A, dtype=float)
        dists = _pairwise_distances(coords_A)
        e_mat = _FLEXE_MD_SCALING * self.k_matrix * (self.dist_matrix_ref - dists) ** 2
        # 1/4 (not 1/2) because k_matrix/dist_matrix_ref/e_mat are all
        # symmetric N x N matrices, so summing the full matrix double-counts
        # every (i,j)/(j,i) pair; 1/4 = 1/2 (Hookean prefactor) * 1/2
        # (double-counting correction) - equivalent to summing 0.5*k*(d-d0)^2
        # once per unique pair, matching eq. 1-2 of the FlexE paper exactly.
        return 0.25 * np.sum(e_mat, axis=(-2, -1))

    def flexe_per_residue(self, coords_A: np.ndarray) -> np.ndarray:
        """evaluate_energy(...) / n_res - the FlexE score as reported in the paper (kcal/mol/residue)."""
        return self.evaluate_energy(coords_A) / self.n_res

    def thermal_energy_per_residue(self, temperature_K: float = 300.0) -> float:
        """
        Equipartition thermal energy, kcal/mol/residue, for comparison against
        `flexe_per_residue`.

        The paper defines ThermE = (3N-6)*kB*T/2 as a TOTAL energy (not
        per-residue), while FlexE itself is already per-residue (eq. 2's own
        1/N prefactor) - the paper's Figure 3 plots FlexE per residue against
        a "thermal noise" line that is consistent with also dividing ThermE
        by N (that line sits at ~1 kcal/mol/residue, matching (3N-6)/N *
        kB*T/2 in the large-N limit -> 3*kB*T/2 ~= 0.89 kcal/mol at 300K).
        This per-residue division is our own derivation for comparing
        directly against `flexe_per_residue`, not a value given explicitly
        in either the paper's equations or the published hamiltonian.py
        (which doesn't compute ThermE at all - it only reports FlexE and RMSD).
        This is a diagnostic/interpretive aid only; it is NOT used as the
        classifier's pass/fail threshold (see BoundLikeCriteria.flexe_threshold).
        """
        n_modes = max(3 * self.n_res - 6, 0)
        total = n_modes * _KB_KCAL_PER_MOL_K * temperature_K / 2.0
        return total / self.n_res
