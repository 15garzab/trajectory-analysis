import pickle
from pymatgen.optimization.neighbors import find_points_in_spheres
import numpy as np


with open('fulldata.p', 'rb') as f:
    data = pickle.load(f)

vactrajs = data['vactrajs']

tracktrajs = {}
tracktrajs[0] = vactrajs[0][:,1:]
for frame in range(len(vactrajs)-1):
    second = vactrajs[frame+1][:,1:].copy(order='C')
    # shapes of first and second must be identical otherwise malloc will freak out because of invalid chunk sizing, hence we slice 'second'
    if len(tracktrajs[frame]) != len(second):
        second = second[:len(tracktrajs[frame]),:]
    result = find_points_in_spheres(center_coords = tracktrajs[frame].copy(order='C'), all_coords = second, r = 3, pbc = np.array([1,1,1]).copy(order='C'), lattice = data['cell'].copy(order='C').astype('double'))
    tracktrajs[frame+1] = second[result[1][:len(second)]]
    print(np.shape(tracktrajs[frame+1]))


arr = np.array(list(tracktrajs.values()), dtype=float)
print(np.shape(arr))
