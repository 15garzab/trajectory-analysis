import pickle
from ovito.pipeline import *
from ovito.modifiers import *
from ovito.io import *
import numpy as np

pipeline = import_file('new900.lmp')
data = pipeline.compute(0)
cell = np.array(data.cell).copy(order = 'C').astype('double')

trajs = {}
vactrajs = {}
nframes = pipeline.source.num_frames

for frame_index in range(0, pipeline.source.num_frames):
    data = pipeline.compute(frame_index)
    pos = np.array(data.particles['Position'])
    types = np.array(data.particles['Particle Type'])
    # must be 2D for np.append
    types = np.reshape(types, (len(types), 1))
    trajs[frame_index] = np.append(types, pos, axis = 1)
    vactrajs[frame_index] = trajs[frame_index][np.ma.where(trajs[frame_index][:,0] == 3)]

with open('fulldata.p', 'wb') as f:
    pickle.dump({"vactrajs": vactrajs,"cell": cell,}, f)
