import os
import h5py
import numpy as np

PATH = '/home/lorenzo/phd/NS_HOT_EOS/EOS/compOSE/FOP(SFHoY)/'
h5_name = 'FOP(SFHoY).h5'

# Files used to create the H5 file. To be generalized to all files
files = ['eos.nb', 'eos.thermo', 'eos.t', 'eos.yq']
# Set the number of lines to skip for each file
skip_rows = {
                'eos.nb' : 2,
                'eos.thermo' : 1,
                'eos.t' : 2,
                'eos.yq' : 2
            }

# Save all the tables in a dictionary
data = {}
for file in files:
    data[file] = np.loadtxt(PATH + file, skiprows=skip_rows[file])
    if file == 'eos.thermo':
        with open(PATH + file) as f:
            m_n, m_p, _ = np.fromstring(f.readline().strip('\n'), dtype = float, sep='\t')


# Indices go from 1 to N in Fortran, from 0 to N-1 in Python
index_T    = data['eos.thermo'][:, 0].astype(int) - 1
index_nb   = data['eos.thermo'][:, 1].astype(int) - 1
index_ye   = data['eos.thermo'][:, 2].astype(int) - 1

# Baryon matter density
rho        = np.exp(data['eos.nb']) * m_n     # e^LOGARITHMIC     MeV / fm^3
pointsrho  = len(rho)

# Electron fraction
y_e        = data['eos.yq']                   # LINEAR         Adimensional
pointsye   = len(y_e)

# Temperature
temp       = np.exp(data['eos.t'])            # e^LOGARITHMIC     MeV
pointstemp = len(temp)

# Map from (i_T, i_nb, i_Ye) to 1D index
maps = lambda i,j,k : i + pointsrho * pointsye * j + pointsye * k

# Find quantities and reshape them to match the H5 files in stellarcollapse.org
# To reshape the matrices the method used is 28 times faster and equivalent to
# for i in range(pointsye):
#     for j in range(pointstemp):
#         for k in range(pointsrho):
#             reshaped_matrix[i, j, k] = original_matrix[maps(i, j, k)]

pressure = data['eos.thermo'][:, 3] * data['eos.nb'][index_nb]   # MeV / fm^3
pressure = pressure[maps(np.arange(pointsye)[:, None, None], np.arange(pointstemp)[None, :, None], np.arange(pointsrho)[None, None, :])]

entropy  = data['eos.thermo'][:, 4] * data['eos.nb'][index_nb]   # Adimensional
entropy  = entropy[maps(np.arange(pointsye)[:, None, None], np.arange(pointstemp)[None, :, None], np.arange(pointsrho)[None, None, :])]

energy   = data['eos.thermo'][:, 9]
energy   = energy[maps(np.arange(pointsye)[:, None, None], np.arange(pointstemp)[None, :, None], np.arange(pointsrho)[None, None, :])]

# Adapt quantities to stellarcollapse format:
logrho   = np.log10(rho)
logtemp  = np.log10(temp)

energy_shift = np.abs(np.min(energy)) * 1.01
logenergy = np.log10(energy + energy_shift)

logpress = np.log10(pressure)


# Create a dictionary to automatically generate the H5 file
ds = {  
        'logrho'   : logrho   , 'pointsrho'   : pointsrho, 
        'y_e'      : y_e      , 'pointsye'    : pointsye, 
        'logtemp'  : logtemp  , 'pointstemp'  : pointstemp, 
        'logpress' : logpress , 
        'entropy'  : entropy  , 
        'logenergy': logenergy, 'energy_shift': energy_shift
    }

with h5py.File(PATH + h5_name, "w") as f:
    for d in ds:
        dset = f.create_dataset(d, data=ds[d], dtype='f')