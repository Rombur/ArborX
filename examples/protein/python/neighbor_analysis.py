#!/usr/bin/env python3

import mdtraj as md
t = md.load('trajectory.xtc', top='coordinates.gro')
# The trajectory has 1251 frames
print("n total frames", t.n_frames)
# the trajectory is quite large, so for testing we can just use the first 10 frames:
t = t[:10]
print("reduces n frames", t.n_frames)

# t is the trajectory object, and t.xyz stores the xyz coordinates.
# The shape is: (time, points, coordinates)
print ("t_shape", t.xyz.shape)
print (t.xyz[0, :2, :]) # xyz coordinates of the first two points of the first frame

# We get the indices for the query and database points
qi = t.topology.select('protein') # query points for which we need to find nearest neighbors
bi = t.topology.select('not protein') # database points

# If we wanted to get the xyz coordinates for the above indices, we can do:
# this is not needed for our purposes here.
q_coor = t.xyz[:, qi]
b_coor = t.xyz[:, bi]

# The radius we will use to search for nearest neighbors
radius = 0.7

## Query the entire protein
# The output contains the indices of found neighbors for each frame in the trajectory.
# len(protein_nn) is t.n_frames
# Each element of protein_nn contains the indices of found neighbors.
protein_nn = md.compute_neighbors(t, radius, qi, bi)
for p in protein_nn:
    print("n proteins", p.shape)

## Query each amino acid
residues = t.topology.subset(qi).residues
residue_nn = {}
for r in residues:
    idx = [ri.index for ri in r.atoms] # We get the indices of atoms making up the residue/aminoacid
    residue_nn[r.index] = md.compute_neighbors(t, radius, idx, bi)
#    for s in residue_nn[r.index]:
#        print("amino acidd", s.shape)

## Query each atom
atoms = t.topology.subset(qi).atoms
atoms_nn = {}
for a in atoms:
    atoms_nn[a.index] = md.compute_neighbors(t, radius, [a.index], bi)
#    for b in atoms_nn[a.index]:
#        print("atorms", b.shape)
