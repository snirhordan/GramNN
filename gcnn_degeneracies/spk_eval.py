import torch
import schnetpack as spk
from ase.io import read
import numpy as np 

import argparse
"""
evaluates energies and RMSE for a model built using schnetpack. 
models must have been built using the default schnetpack scripts,
spk_run.py train schnet custom /dev/shm/nmers.db model/ --property energy_b3lyp_ha --aggregation_mode sum --split 15000 2000 --cuda

analysis can be run with 
$ python3 spk_eval.py -m model/ -d selected_structures.xyz
"""

parser = argparse.ArgumentParser()
parser.add_argument("-m", help="model")
parser.add_argument("-d", help="dataset")
args = parser.parse_args()

best_model = torch.load(args.m)

spk_calculator = spk.interfaces.SpkCalculator(model=best_model, energy='energy_b3lyp_ha')

frames = read(args.d, ':')

mse = 0
for f in frames[2:]:
    f.set_calculator(spk_calculator)
    ref = f.info['energy_ha']
    pred = f.get_total_energy()
    print((ref-pred)*27.211386)
    mse += (ref-pred)**2
print("RMSE ", np.sqrt(mse/len(frames[2:]))*27.211386 )
