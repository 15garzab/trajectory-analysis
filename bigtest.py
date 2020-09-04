import pickle
import numpy as np
from pymatgen.optimization.neighbors import find_points_in_spheres

with open('fulldata.p', 'rb') as f:
    data = pickle.load(f)

vactrajs = data['vactrajs']

tracktrajs = {}
tracktrajs[0] = vactrajs[0]
for frame in range(len(vactrajs)-1):
    second = vactrajs[frame+1][:,1:].copy(order='C')
    first = vactrajs[frame][:,1:].copy(order='C')
    result = find_points_in_spheres(center_coords = first, all_coords = second, r = 3, pbc = np.array([1,1,1]), lattice = data['cell'])
    tracktrajs[frame+1] = second[result[1][:len(second)]]

print(tracktrajs[frame])
