import os
import h5py
import numpy as np
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


PATH = '/home/lorenzo/phd/NS_HOT_EOS/EOS/compOSE/FOP(SFHoY)'
h5_name = 'TEST_' + PATH.split('/')[-1] + '.h5'
OUTPUT = os.path.join(PATH, h5_name)

# Files used to create the H5 file. To be generalized to all files
files = ['eos.nb', 'eos.thermo', 'eos.t', 'eos.yq', 'eos.compo']
# Set the number of lines to skip for each file
skip_rows = {
                'eos.nb' : 2,
                'eos.thermo' : 1,
                'eos.t' : 2,
                'eos.yq' : 2,
                'eos.compo' : 0
            }

# Initialize an empty dictionary to store the data tables
data = {}

# Iterate through the list of file names
for file in files:
    print(f'Reading the file {file}...', end='\r      ')

    # Construct the file path
    FILE_PATH = os.path.join(PATH, file)
    
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
        print(f'{file} not found!')

print('All files read correctly!', end = '     \n')
print('EOS informations saved:')

# Indices go from 1 to N in Fortran, from 0 to N-1 in Python
index_T    = data['eos.thermo'][:, 0].astype(int) - 1
index_nb   = data['eos.thermo'][:, 1].astype(int) - 1
index_yq   = data['eos.thermo'][:, 2].astype(int) - 1

# Baryon matter density
logrho     = np.log10(data['eos.nb'] * m_n)  # log_10( 10^nb * m_n )   MeV / fm^3
pointsrho  = len(logrho)

print('\t- rho = m_n * nb')

# Charge fraction
y_q        = data['eos.yq']                  # LINEARE                 Adimensionale
pointsyq   = len(y_q)

print('\t- Y_q')

# Temperature
logtemp    = np.log10(data['eos.t'])         # log_10( 10^T )          MeV
pointstemp = len(logtemp)

print('\t- T')

# Map from (i_T, i_nb, i_Ye) to 1D index

def maps(i, j, k):
    """
    Defines a function that maps indices from the original matrix to the reshaped matrix calculating the index in the original matrix corresponding to the given indices in the reshaped matrix.
    """
    return i + pointsrho * pointsyq * j + pointsyq * k

 
def reshape_array(original_matrix, pointsrho, pointstemp, pointsyq):
    """
    Reshapes the original matrix into the desired shape using the maps function.
    """
    indices = maps(np.arange(pointsyq)[:, None, None], np.arange(pointstemp)[None, :, None], np.arange(pointsrho)[None, None, :])
    # Returns the values from the original matrix at the calculated indices
    return np.take(original_matrix, indices)

# Find quantities and reshape them to match the H5 files in stellarcollapse.org
# To reshape the matrices the method used is faster and equivalent to
# for i in range(pointsyq):
#     for j in range(pointstemp):
#         for k in range(pointsrho):
#             reshaped_matrix[i, j, k] = original_matrix[maps(i, j, k)]
energy   = data['eos.thermo'][:, 9]                              # MeV / fm^3
energy = reshape_array(energy, pointsrho, pointstemp, pointsyq)
# Check if the energy needs to be shifted to compute the log:
# if the minimum energy is greater than zero set the shift to zero, else it is 1% bigger than the minimum value (to avoid div by zero)
energy_shift = np.abs(np.min([np.min(energy), 0])) * 1.01
energy   = energy + energy_shift
logenergy = np.log10(energy)

print('\t- Internal energy ')

pressure = data['eos.thermo'][:, 3] * data['eos.nb'][index_nb]   # MeV / fm^3
pressure = reshape_array(pressure, pointsrho, pointstemp, pointsyq)
# SHIFT??
logpress = np.log10(pressure)

print('\t- Pressure')

entropy  = data['eos.thermo'][:, 4] * data['eos.nb'][index_nb]   # Adimensional
entropy = reshape_array(entropy, pointsrho, pointstemp, pointsyq)

print('\t- Entropy')
#### CS2 and GAMMA #############


################################

#### MASS FRACTIONS ############
# Thede dictionaries might need to be generalized
particle_index = {  0    : 'e',    10   : 'n',     11   : 'p',
                    100  : 'Λ',    110  : 'Σ−',    111  : 'Σ0',  112  : 'Σ+',  120  : 'Ξ−', 121  : 'Ξ0', 
                    4002 : '24He', 3002 : '23He',  3001 : '13H', 2001 : '12H', 
                    999  : 'other'}

baryonic_number = { 0    : 0, 10   : 1, 11   : 1,
                    100  : 1, 110  : 1, 111  : 1, 112  : 1, 120 : 1, 121 : 1,
                    4002 : 4, 3002 : 3, 3001 : 3, 2001 : 2,}

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
            mass_fractions[f'X{particle_index[particle]}'][i] = mass_fraction
        except KeyError:
            mass_fractions[f'X{particle_index[particle]}'] = np.full(len(data['eos.compo']), np.NaN)
            mass_fractions[f'X{particle_index[particle]}'][i] = mass_fraction

print('\t- Mass fractions')
################################

# Create a dictionary to automatically generate the H5 file
ds = {  
        'logrho'   : logrho   , 'pointsrho'   : pointsrho, 
        'y_q'      : y_q      , 'pointsyq'    : pointsyq, 
        'logtemp'  : logtemp  , 'pointstemp'  : pointstemp, 
        'logpress' : logpress , 
        'entropy'  : entropy  , 
        'logenergy': logenergy, 'energy_shift': energy_shift
    }

for key in mass_fractions.keys():
    ds[key] = mass_fractions[key][maps(np.arange(pointsyq)[:, None, None], np.arange(pointstemp)[None, :, None], np.arange(pointsrho)[None, None, :]) ]

with h5py.File(OUTPUT, "w") as f:
    for d in ds:
        dset = f.create_dataset(d, data=ds[d], dtype=np.float64)

print(f'\nHDF5 file is')
print(f'{OUTPUT}')