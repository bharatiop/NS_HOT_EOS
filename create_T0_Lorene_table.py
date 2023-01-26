import os
import sys
import errno

import h5py

import numpy as np
import pycactus3.UtilityEOS as EOS

path = sys.argv[1]
with h5py.File(path, "r") as f:
    logrho = f['logrho'][()]

EOS.ReadOttTable(path, nrows=5000, rhomin=10**logrho[0], rhomax=10**logrho[-1], out_format="lorene")