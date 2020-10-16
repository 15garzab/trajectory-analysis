import os
import sys
import numpy as np
from ovito.pipeline import ReferenceConfigurationModifier, FileSource
from ovito.modifiers import WignerSeitzAnalysisModifier, ComputePropertyModifier, InvertSelectionModifier, DeleteSelectedModifier, CombineDatasetsModifier
from ovito.io import import_file, export_file
    """ Vacancy Labeler and File I/O
    This code is mean for use on a single .xyz trajectory file with a pristine,
    crystalline reference configuration (.lmp) to locate vacancies in all frames
    of the trajectory based on Voronoi cell binning (Wigner-Seitz Analysis).

    Input (requested): Directory path and file name ____.xyz
    * FCCreference.lmp must also be present in the working directory*
    Output: Two files named '___vacancies.xyz' (1) and 'new___.xyz' (2)
        (1): contains only the labeled vacancies - good for visualization
        (2): contains all particles including vacancies - good for analysis
     """
# custom modifier to select vacancies after Wigner-Seitz analysis
def modify(frame,data):
    occupancies = data.particles['Occupancy']
    total_occupancy = np.sum(occupancies, axis = 1)
    selection = data.particles_.create_property('Selection')
    selection[...] = (total_occupancy == 0)

# User input for directory and filename
print('OVITO is ready to find your vacancies and make some new files (this could take a while!)')
dir = input('Please state the working directory for the OVITO files: ')
try:
    os.chdir(dir)
except:
    print('Couldn''t find that directory, check syntax (/mnt/d/UserName/etc.)')
    sys.exit(1)
file = input('Please provide the name of your file: ')
try:
    pipeline = import_file(file)
except:
    print('File not found (check the name and that you''re in the right directory)')
    sys.exit(1)
wigmod = WignerSeitzAnalysisModifier(per_type_occupancies = True, affine_mapping = ReferenceConfigurationModifier.AffineMapping.ToReference)
wigmod.reference = FileSource()
try:
    wigmod.reference.load('FCCreference.lmp')
except:
    print('FCCreference.lmp was not found!')
    sys.exit(1)
pipeline.modifiers.append(wigmod)
pipeline.modifiers.append(ComputePropertyModifier(output_property='Particle Type', expressions = '3'))
# see custom modifier above
pipeline.modifiers.append(modify)
pipeline.modifiers.append(InvertSelectionModifier())
pipeline.modifiers.append(DeleteSelectedModifier())
# export a file with the vacancies only (type = 3)
export_file(pipeline, file.split('.')[0] + "vacancies.xyz", "xyz", columns = ['Particle Identifier', 'Particle Type', 'Position.X', 'Position.Y', 'Position.Z'], multiple_frames = True)
# new pipeline to combine the vacancies with the original data (this will append the vacancies at the end of the Particle IDs and reassign new IDs accordingly)
newpipeline = import_file(file)
mod = CombineDatasetsModifier()
mod.source.load(file.split('.')[0] + 'vacancies.xyz')
newpipeline.modifiers.append(mod)
# make a new file with the vacancies and all atoms included for visualization
export_file(newpipeline, "new" + file.split('.')[0] + ".xyz", "xyz", columns = ['Particle Identifier', 'Particle Type', 'Position.X', 'Position.Y', 'Position.Z'], multiple_frames = True )
