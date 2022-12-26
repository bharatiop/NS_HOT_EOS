import os
import h5py
import numpy as np

PATH = '/home/lorenzo/phd/NS_HOT_EOS/EOS/compOSE/FOP(SFHoY)/'
h5_name = 'TEST_' + PATH.split('/')[-2] + '.h5'
OUTPUT = os.path.join(PATH, h5_name)

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
    FILE_PATH = os.path.join(PATH, file)
    data[file] = np.loadtxt(FILE_PATH, skiprows=skip_rows[file])
    if file == 'eos.thermo':
        with open(FILE_PATH) as f:
            m_n, m_p, _ = np.fromstring(f.readline().strip('\n'), dtype = float, sep='\t')


# Indices go from 1 to N in Fortran, from 0 to N-1 in Python
index_T    = data['eos.thermo'][:, 0].astype(int) - 1
index_nb   = data['eos.thermo'][:, 1].astype(int) - 1
index_ye   = data['eos.thermo'][:, 2].astype(int) - 1

# Baryon matter density
logrho     = np.log10(data['eos.nb'] * m_n)  # log_10( 10^nb * m_n )   MeV / fm^3
pointsrho  = len(logrho)

# Electron fraction
y_e        = data['eos.yq']                  # LINEARE                 Adimensionale
pointsye   = len(y_e)

# Temperature
logtemp    = np.log10(data['eos.t'])         # log_10( 10^T )          MeV
pointstemp = len(logtemp)

# Map from (i_T, i_nb, i_Ye) to 1D index
maps = lambda i,j,k : i + pointsrho * pointsye * j + pointsye * k

# Find quantities and reshape them to match the H5 files in stellarcollapse.org
# To reshape the matrices the method used is faster and equivalent to
# for i in range(pointsye):
#     for j in range(pointstemp):
#         for k in range(pointsrho):
#             reshaped_matrix[i, j, k] = original_matrix[maps(i, j, k)]

energy   = data['eos.thermo'][:, 9]                              # MeV / fm^3
energy   = energy[maps(np.arange(pointsye)[:, None, None], np.arange(pointstemp)[None, :, None], np.arange(pointsrho)[None, None, :])]
# Check if the energy needs to be shifted to compute the log:
# if the minimum energy is greater than zero set the shift to zero, else it is 1% bigger than the minimum value (to avoid div by zero)
energy_shift = np.abs(np.min([np.min(energy), 0])) * 1.01
energy   = energy_r + energy_shift

pressure = data['eos.thermo'][:, 3] * data['eos.nb'][index_nb]   # MeV / fm^3
pressure = pressure[maps(np.arange(pointsye)[:, None, None], np.arange(pointstemp)[None, :, None], np.arange(pointsrho)[None, None, :])]

entropy  = data['eos.thermo'][:, 4] * data['eos.nb'][index_nb]   # Adimensional
entropy  = entropy[maps(np.arange(pointsye)[:, None, None], np.arange(pointstemp)[None, :, None], np.arange(pointsrho)[None, None, :])]

# Adapt quantities to stellarcollapse format:
logrho   = np.log10(rho)
logtemp  = np.log10(temp)
logenergy = np.log10(energy)
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

with h5py.File(OUTPUT, "w") as f:
    for d in ds:
        dset = f.create_dataset(d, data=ds[d], dtype='f')