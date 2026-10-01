import mdtraj as md
import numpy as np

traj = md.load("/orange/alberto.perezant/imesh.ranaweera/projects/7_bioEMU_peptide/WT_p53/1YCR.pdb")

# Confirm chain 0 is really MDM2 before trusting it
for chain in traj.topology.chains:
    residues = list(chain.residues)
    print(f"chain {chain.index}: {len(residues)} residues, "
          f"{residues[0]}–{residues[-1]}")

mdm2 = traj.atom_slice(traj.topology.select("chainid 0"))  # confirm this is MDM2 from the printout above

# Use resSeq (actual PDB numbering) — matches mdm2_groove_residues directly, no offset guessing
pocket1_res = mdm2.topology.select("resSeq 65 67 69 73 74 75 62 93 and not element H")   # Phe19 pocket
pocket2_res = mdm2.topology.select("resSeq 92 93 54 58 60 91 and not element H")        # Trp23 pocket
pocket3_res = mdm2.topology.select("resSeq 100 101 53 and not element H")     # Leu26 pocket

centers = []
for sel in [pocket1_res, pocket2_res, pocket3_res]:
    if len(sel) == 0:
        raise ValueError("Empty selection — check resSeq numbers exist in this chain")
    com = md.compute_center_of_mass(mdm2.atom_slice(sel))
    centers.append((com[0] * 10).tolist())  # nm → Å

print(centers)



# comment out the lines below if you are not using the code for p53
#If you still need to use the code change the this according to your system.
# p53 is chain 1
p53 = traj.atom_slice(
    traj.topology.select("chainid 1")
)

# Anchor residues: full-length p53 numbering
anchor_residues = {
    "F19": 19,
    "W23": 23,
    "L26": 26
}

anchor_centers = {}

for name, resseq in anchor_residues.items():

    sel = p53.topology.select(
        f"resSeq {resseq}"
    )

    com = md.compute_center_of_mass(
        p53.atom_slice(sel)
    )

    anchor_centers[name] = com[0] * 10  # Å


for name, pocket_center in zip(
    ["F19", "W23", "L26"],
    centers
):

    distance = np.linalg.norm(
        anchor_centers[name] - np.array(pocket_center)
    )

    print(
        f"{name}: "
        f"{distance:.2f} Å from pocket center"
    )