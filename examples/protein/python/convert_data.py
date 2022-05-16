#!/usr/bin/env python3

import numpy as np
import pickle
import mdtraj as md

t = md.load('trajectory.xtc', top='coordinates.gro')
qi = t.topology.select('protein') # query points for which we need to find nearest neighbors
bi = t.topology.select('not protein') # database points
np.savetxt("protein_indices.txt", qi)
np.savetxt("not_protein_indices.txt", bi)
