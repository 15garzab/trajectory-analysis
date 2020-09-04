import pickle
import numpy as np
from pymatgen.optimization.neighbors import find_points_in_spheres
with open('data.p', 'rb') as f:
    data = pickle.load(f)

result = find_points_in_spheres(center_coords = data['first'], all_coords = data['second'], r = 3, pbc = np.array([1,1,1]), lattice = data['cell'])
print(result)
