import numpy as np
import h5py

import errno
import os
from pathlib import Path

import argparse

############
# Constants
############
clite = 2.99792458e10 #cgs
MeVc2_to_grams = 1.78266232885782e-27
baryonic_mass = 939.56542052 # Mev / c^2

#####################
# Read table
#####################

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help = "Path to the EOS file. It must be a .h5 format that can be found at https://stellarcollapse.org/equationofstate.html", type = str)
parser.add_argument('-N', '--name', help = "Name of the table that is saved. By deafult it is equal to the name of the H5 file.", type = str)
args = parser.parse_args()

if not os.path.isfile(args.file):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.file)
if args.name == None:
    args.name = args.file.split('/')[-1][:-3]
    
data = {}
with h5py.File(args.file) as f:
    keys = list(f.keys())
    for key in keys:
        data[key] = np.array(f[key])


######################
# Baryon mass density
baryon_mass_density = 10**data['logrho']   # gr / cm^3
# Baryon number density
baryon_number_density =  baryon_mass_density / baryonic_mass / MeVc2_to_grams  # 1 / cm^3

# Find indexes of electronic fractions that minimize energy at every density
ye_0_idx = np.argmin(data['logenergy'][:, 0, :], axis = 0)
entry = np.array(range(0, len(ye_0_idx)))

# Find specific internal energy and shift it back
eps = 10**data['logenergy'][ye_0_idx, 0, entry] - data['energy_shift'][0]   # erg / gr
# Find pressure
pressure = 10**data['logpress'][ye_0_idx, 0, entry]   # dyn / cm^2
# Find speed of sound squared
cs2 = data['cs2'][ye_0_idx, 0, entry]
# Find energy density
energy_density = baryon_mass_density * (1 + eps/clite**2)
# Find log-enthalpy
logh              = cs2 * np.log((energy_density + pressure/clite**2) / baryon_mass_density)
logh[logh < 0]    = 1


######################
# Save table
######################

Path("./RNSID_Tables/").mkdir(parents=True, exist_ok=True)
output_path = "RNSID_Tables/" + args.name + ".rnsidtable"
if os.path.isfile(output_path):
    os.system(f'rm {output_path}')

with open(output_path, "a") as f:
    f.write(f"{len(baryon_mass_density)}\n")
    np.savetxt(f, np.c_[energy_density, pressure, logh, baryon_number_density])