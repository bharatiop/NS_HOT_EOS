import os
import h5py
import numpy as np

PATH = '../EOS/compOSE/FOP(SFHoY)/'
files = ['eos.nb', 'eos.thermo', 'eos.t', 'eos.yq']

skip_rows = {
                'eos.nb' : 2,
                'eos.thermo' : 1,
                'eos.t' : 2,
                'eos.yq' : 2
            }

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

rho        = data['eos.nb'] * m_n     # LOGARITHMIC     MeV / fm^3
pointsrho  = len(rho)

y_e        = data['eos.yq']           # LINEAR         Adimensional
pointsye   = len(y_e)

temp       = data['eos.t']            # LOGARITHMIC     MeV
pointstemp = len(temp)

maps = lambda i,j,k : i + pointsrho * pointsye * j + pointsye * k

# Find quantities and reshape them to match the H5 files in stellarcollapse.org
pressure = data['eos.thermo'][:, 3] * data['eos.nb'][index_nb]   # MeV / fm^3
pressure = pressure[maps(np.arange(pointsye)[:, None, None], np.arange(pointstemp)[None, :, None], np.arange(pointsrho)[None, None, :])]

entropy  = data['eos.thermo'][:, 4] * data['eos.nb'][index_nb]   # Adimensional
entropy  = entropy[maps(np.arange(pointsye)[:, None, None], np.arange(pointstemp)[None, :, None], np.arange(pointsrho)[None, None, :])]

energy   = data['eos.thermo'][:, 9]
energy   = energy[maps(np.arange(pointsye)[:, None, None], np.arange(pointstemp)[None, :, None], np.arange(pointsrho)[None, None, :])]

ds = {  'rho'     : rho     , 'pointsrho':pointsrho, 
        'y_e'     : y_e     , 'pointsye':pointsye, 
        'temp'    : temp    , 'pointstemp':pointstemp, 
        'pressure': pressure, 
        'entropy' : entropy , 
        'energy'  : energy
    }

with h5py.File(PATH + "TEST_FOP(SFHoY).h5", "w") as f:
    for d in ds:
        dset = f.create_dataset(d, data=ds[d], dtype='f')