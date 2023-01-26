import os
import h5py

import argparse
import errno

import numpy as np
from scipy.interpolate import CubicSpline 

from itertools import chain

def read_table_varying_columns(FILE_PATH):
    """
    Read a file and return the contents as a 2D NumPy array.
    
    Parameters:
    - FILE_PATH (str): The path to the file to be read.
    
    Returns:
    - out_arr (NumPy array): A 2D NumPy array with the contents of the file, with missing values
        represented as NaN.
    """
    
    # Open the file and read the contents as a list of lines
    with open(FILE_PATH, 'r') as f:
        all_data=[x.split() for x in f.readlines()]

    # Get the number of rows and columns in the data
    n_cols = max(len(line) for line in all_data)
    n_rows = len(all_data)
    
    # Initialize a 2D NumPy array with default values of NaN
    out_arr = np.full((n_rows, n_cols), np.NaN)

    # Iterate through the lines of data
    for i, line in enumerate(all_data):
        # Convert the line to a NumPy array and assign it to the corresponding row in the output array
        out_arr[i, :len(line)] = np.array(line, dtype = np.float64)
    
    # Return the output array
    return out_arr

# Map from (i_T, i_nb, i_Ye) to 1D index
def maps(i, j, k, pointsrho, pointsyq):
    """
    Defines a function that maps indices from the original matrix to the reshaped matrix calculating the index in the original matrix corresponding to the given indices in the reshaped matrix.
    """
    return i + pointsrho * pointsyq * j + pointsyq * k

 
def reshape_array(original_matrix, pointsrho, pointstemp, pointsyq):
    """
    Reshapes the original matrix into the desired shape using the maps function.
    """
    indices = maps(np.arange(pointsyq)[:, None, None], np.arange(pointstemp)[None, :, None], np.arange(pointsrho)[None, None, :], pointsrho, pointsyq)
    # Returns the values from the original matrix at the calculated indices
    return np.take(original_matrix, indices)

def spline_derivative(y, x, nu=1):
    """
    Computes the 'nu'-th order derivative of the input array 'y' with respect to 'x'.

    Parameters:
    y (1D array): The input array.
    x (1D array): The independent variable. Must be a strictly increasing sequence.
    nu (int): The order of the derivative. Default is 1 (first-order derivative).

    Returns:
    1D array: The 'nu'-th order derivative of 'y' with respect to 'x' computed at 'x'.
    """
    return CubicSpline(x, y).derivative(nu = nu)(x)


def dir_path(string):
    """
    This function checks if the file path passed as an argument is a valid directory.
    If the file path is not a valid directory, it raises a NotADirectoryError with additional information such as the error number, error message and the string passed as an argument.

    Parameters:
    string: path of file to check
    
    Returns:
    return the path if it is valid directory else raise NotADirectoryError
    """
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), string)

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Define the 'folder' argument
parser.add_argument('-f', '--folder', 
    help = "Path to the EOS files. It must be a folder containing files from the CompOSE website.\
            Note that if the path contains spaces or special characters the path must be provided with the following syntax: \
            -f/--folder='special/path/to/folder/'.", 
    type = dir_path, # use the dir_path function as a type checker
    required = True)

# Define the 'name' argument
parser.add_argument('-N', '--name', 
    help = "Name of the h5 table that is saved. By deafult it is equal to the name of the given folder.", 
    type = str)

# Parse the command-line arguments
args = parser.parse_args()

# If the folder path ends with a '/', remove it
if args.folder[-1] == '/':
    args.folder = args.folder[:-1]

# If the 'name' argument is not provided 
if args.name == None:
    # set the name to the name of the folder
    args.name = args.folder.split('/')[-1] + '.h5'
# or if it does not have the '.h5' extension
elif args.name[-3:] != '.h5':
    # and append '.h5'
    args.name += '.h5'

# Define the OUTPUT constant
OUTPUT = os.path.join(args.folder, args.name)

# Files used to create the H5 file. 
files = [file for file in os.listdir(args.folder) if 'eos.' in file and not '.pdf' in file and not '.init' in file and not '.ns' in file]
# Check if all the mandatory files are present
mandatory_files = ['eos.nb', 'eos.t', 'eos.yq', 'eos.thermo']
missing_files = [file for file in mandatory_files if file not in files]
if len(missing_files) > 0:
    raise FileNotFoundError(f'The following mandatory files are missing: {missing_files}')

# Set the number of lines to skip for each file
skip_rows = {
                'eos.nb' : 2,
                'eos.thermo' : 1,
                'eos.t' : 2,
                'eos.yq' : 2,
                'eos.compo' : 0,
                'eos.micro' : 0,
                'eos.mr'    : 1
            }

# Initialize an empty dictionary to store the data tables
data = {}

print('Starting conversion.')
# Iterate through the list of file names
for file in files:
    print(f'Reading the file {file}...', end='\r')

    # Construct the file path
    FILE_PATH = os.path.join(args.folder, file)
    
    # Try to read the file as a table with a fixed number of columns
    # If this fails, handle the exception and try reading the file as a table with a variable number of columns
    try:
        data[file] = np.loadtxt(FILE_PATH, skiprows=skip_rows[file], dtype=np.float64)
        
        # If the file is the 'eos.thermo' file, read the first line and extract the m_n and m_p values
        if file == 'eos.thermo':
            with open(FILE_PATH) as f:
                m_n, m_p, _ = np.fromstring(f.readline().strip('\n'), dtype = float, sep='\t')
    except ValueError:
        data[file] = read_table_varying_columns(FILE_PATH)
    except KeyError:
        # If the file is not found, print an error message
        print(f'{file} not found, but it is optional!')
        

print('All files read correctly!', end = '     \n')
print('EOS informations saved:')

# Indices go from 1 to N in Fortran, from 0 to N-1 in Python
index_T    = data['eos.thermo'][:, 0].astype(int) - 1
index_nb   = data['eos.thermo'][:, 1].astype(int) - 1
index_yq   = data['eos.thermo'][:, 2].astype(int) - 1

# Baryon matter density
logrho     = np.log10(data['eos.nb'] * m_n * 1.78e12)  # log_10( nb * m_n  MeV / fm^3 --> g / cm^3 )
pointsrho  = len(logrho)

print('\t* log_10(rho = m_n * nb [g / cm^3])')

# Charge fraction
y_q        = data['eos.yq']                  # LINEARE                 Adimensionale
pointsyq   = len(y_q)

print('\t* Y_q [Adim]')

# Temperature
logtemp    = np.log10(data['eos.t'])        # log_10( T )         1 MeV --> 11604512367.3632 K
pointstemp = len(logtemp)

print('\t* log_10(T [MeV])')

# Find quantities and reshape them to match the H5 files in stellarcollapse.org
# To reshape the matrices the method used is faster and equivalent to
# for i in range(pointsyq):
#     for j in range(pointstemp):
#         for k in range(pointsrho):
#             reshaped_matrix[i, j, k] = original_matrix[maps(i, j, k)]
energy = data['eos.thermo'][:, 9] * 2.99792458e10 * 2.99792458e10     # c^2 == erg / g
energy = reshape_array(energy, pointsrho, pointstemp, pointsyq)
# Check if the energy needs to be shifted to compute the log:
# if the minimum energy is greater than zero set the shift to zero, else it is 1% bigger than the minimum value (to avoid div by zero)
energy_shift = np.abs(np.min([np.min(energy), 0])) * 1.01
energy   = energy + energy_shift
logenergy = np.log10(energy)

print('\t* log_10(Specific internal energy [erg / g])')

pressure = data['eos.thermo'][:, 3] * data['eos.nb'][index_nb] * 1.602e33  # MeV / fm^3 --> dyne / cm^2
pressure = reshape_array(pressure, pointsrho, pointstemp, pointsyq)
logpress = np.log10(pressure)

print('\t* log_10(Pressure [dyne / cm^2])')

entropy  = data['eos.thermo'][:, 4] * data['eos.nb'][index_nb]   # Adimensional
entropy = reshape_array(entropy, pointsrho, pointstemp, pointsyq)

print('\t* Entropy [Adim]')

#### CS2 and GAMMA #############
# Compute and reshape the energy e = rho * (1 + eps)
e = m_n * data['eos.nb'][index_nb] * ( 1 + data['eos.thermo'][:, 9] ) # MeV / fm^3
e = reshape_array(e, pointsrho, pointstemp, pointsyq)

# Compute the square of the speed of sound from
# cs2 = dp/de = dp/dn_b * dn_b/de + dp/dT * dT/de + dp/dy_q * dy_q/de
dpdnb = np.apply_along_axis(spline_derivative, 2, pressure, data['eos.nb'])
dednb = np.apply_along_axis(spline_derivative, 2, e       , data['eos.nb'])
dpdt  = np.apply_along_axis(spline_derivative, 1, pressure, data['eos.t'] )
dedt  = np.apply_along_axis(spline_derivative, 1, e       , data['eos.t'] )
dpdy  = np.apply_along_axis(spline_derivative, 0, pressure, data['eos.yq'])
dedy  = np.apply_along_axis(spline_derivative, 0, e       , data['eos.yq'])

cs2 = dpdnb/dednb + dpdt/dedt + dpdy/dedy
print('\t* Squared speed of sound [c^2]')

# The adiabatic index is gamma = dln(p)/dln(e) = e * cs2 / p
gamma = e * cs2 / pressure

print('\t* Adiabatic index [Adim]')
################################


if 'eos.compo' in files:
    # Thede dictionaries might need to be generalized
    particle_index = {  0    : 'e',    10   : 'n',     11   : 'p',
                        100  : 'Λ',    110  : 'Σ−',    111  : 'Σ0',  112  : 'Σ+',  120  : 'Ξ−', 121  : 'Ξ0', 
                        4002 : '24He', 3002 : '23He',  3001 : '13H', 2001 : '12H', 
                        999  : 'other'}
    particle_index_inv = {v: k for k, v in particle_index.items()}

    baryonic_number = { 0    : 1, 10   : 1, 11   : 1,
                        100  : 1, 110  : 1, 111  : 1, 112  : 1, 120 : 1, 121 : 1,
                        4002 : 4, 3002 : 3, 3001 : 3, 2001 : 2,}

    #### MASS FRACTIONS ############

    mass_fractions = {}
    for i, row in enumerate(data['eos.compo']):
        # Extract the number of pairs and quads from the current row
        n_pairs = int(row[4])
        n_quads = int(row[5 + n_pairs * 2])

        conc_iter = chain(range(5, 5 + n_pairs * 2, 2), range(6 + n_pairs * 2, 6 + n_pairs * 2 + n_quads * 4, 4))
        # Iterate through the pairs and quads
        for j in conc_iter:
            # Get the particle
            particle = int(row[j])
            lbl = f'X{particle_index[particle]}'
            # If j is within the range of the pairs, get its mass fraction
            if j < 5 + n_pairs * 2:
                mass_fraction = row[j+1] * baryonic_number[particle]
            # If j is within the range of the quads, get its mass fraction
            else:
                mass_fraction = row[j+1] * row[j+3]

            # Try to add the mass fraction to the mass_fractions dictionary
            # If the particle is not already in the dictionary, create a new entry
            # with a default value of NaN for all indices and add the value
            try:
                mass_fractions[lbl][i] = mass_fraction
            except KeyError:
                mass_fractions[lbl] = np.full(len(data['eos.compo']), np.NaN)
                mass_fractions[lbl][i] = mass_fraction

    # Reshape arrays
    for key in mass_fractions.keys():
        mass_fractions[key] = reshape_array(mass_fractions[key], pointsrho, pointstemp, pointsyq)

    print('\t* Mass fractions [Adim]')
    
    ################################
    #### CHEMICAL POTENTIALS #######
    
    free_energy = (1 + data['eos.thermo'][:, 9]) * m_n * data['eos.nb'][index_nb]
    free_energy = reshape_array(free_energy, pointsrho, pointstemp, pointsyq)
    
    mus = {}
    
    for key in mass_fractions.keys():
        key = particle_index_inv[key[1:]]
        if key == 999:
            continue

        mu_lbl = f'mu_{particle_index[key]}'
        X_lbl  = f'X{particle_index[key]}'
        mus[mu_lbl] = reshape_array(np.full(len(data['eos.compo']), np.NaN), pointsrho, pointstemp, pointsyq)

        if key == 0 or key == 11:
            for i in range(pointstemp):
                for j in range(pointsrho):
                    for k in range(pointsyq-1):
                        mus[mu_lbl][k, i, j] = ( free_energy[k+1, i, j] - free_energy[k, i, j] ) / (np.abs(data['eos.yq'][k+1] - data['eos.yq'][k])) / data['eos.nb'][j]
        else:
            for i in range(pointstemp):
                for j in range(pointsrho):
                    for k in range(pointsyq-1):
                        mus[mu_lbl][k, i, j] = baryonic_number[key] * ( free_energy[k+1, i, j] - free_energy[k, i, j] ) / (np.abs(mass_fractions[X_lbl][k+1, i, j] - mass_fractions[X_lbl][k, i, j] )) / data['eos.nb'][j]
        
    print('\t* Chemical potentials [MeV]')
    ################################

# Create a dictionary to automatically generate the H5 file
ds = {  
        'logrho'   : logrho   , 'pointsrho'   : pointsrho, 
        'ye'       : y_q      , 'pointsyq'    : pointsyq, 
        'logtemp'  : logtemp  , 'pointstemp'  : pointstemp, 
        'logpress' : logpress , 
        'entropy'  : entropy  , 
        'logenergy': logenergy, 'energy_shift': [energy_shift], 'free_energy' : free_energy,
        'cs2'      : cs2      , 'gamma'       : gamma
    }

if 'eos.compo' in files:
    for key in mass_fractions.keys():
       ds[key] = mass_fractions[key]

    for key in mus.keys():
        ds[key] = mus[key]

    ds['munu'] = ds['mu_e'] - ds['mu_n'] + ds['mu_p']

with h5py.File(OUTPUT, "w") as f:
    for d in ds:
        dset = f.create_dataset(d, data=np.nan_to_num(ds[d]), dtype=np.float64)

print(f'\nHDF5 file is')
print(f'{OUTPUT}')