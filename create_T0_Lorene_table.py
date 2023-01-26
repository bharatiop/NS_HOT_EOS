import os
import sys
import errno

import argparse
import h5py

import numpy as np
import matplotlib.pyplot as plt
import pycactus3.UtilityEOS as EOS

def file_path(string):
    """
    This function checks if the file path passed as an argument is a valid file.
    If the file path is not a valid file, it raises a FileNotFoundError with additional information such as the error number, error message and the string passed as an argument.

    Parameters:
    string: path of file to check
    
    Returns:
    return the path if it is valid directory else raise FileNotFoundError
    """
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), string)

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Define the 'folder' argument
parser.add_argument('-f', '--file', 
    help = "Path to the H5 EOS file.\
            Note that if the path contains spaces or special characters the path must be provided with the following syntax: \
            -f/--folder='special/path/to/folder/'.", 
    type = file_path, # use the dir_path function as a type checker
    required = True)

# Define the 'nrows' argument
parser.add_argument('-n', '--nrows', 
    help = "Number of rows in the output table. Default is 5000",
    default = 5000,
    type = int)

# Parse the command-line arguments
args = parser.parse_args()

name = os.path.split(args.file)[1]
name = os.path.splitext(name)[0]


with h5py.File(args.file, "r") as f:
    logrho = f['logrho'][()]

EOS.ReadOttTable(args.file, nrows=args.nrows, rhomin=10**logrho[0], rhomax=10**logrho[-1], out_format="lorene")

data = np.loadtxt(name+'_lorene.d', skiprows=9)

fig = plt.figure()
plt.plot(data[:,2], data[:,3], color = 'k')

plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'$\rho$ [g/$cm^3$]')
plt.ylabel(r'P [dyn/$cm^2$]')

plt.savefig(name + '.png')