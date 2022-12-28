# NS_HOT_EOS

This repository contains a codebase for creating proper tables for the equations of state (EOS) of hot, dense nuclear matter. The EOS is an important input for modeling the properties of neutron stars, such as their mass, radius, and temperature.
This EOS will be used for a simulation of a single Neutron Star using the Einstein Toolkit.

### Prerequisites

To run this code, you will need the following software and libraries:

- Python 3.6 or later
- NumPy
- SciPy
- Matplotlib

### Running the code

The file 'create_RNSID_table.py' can be executed to generate an HDF5 file compatible with the thorn RNSID; information on the requested input parameters is provided via the flag -h.
The file 'create_H5_table.py' also generates an HDF5 file that will be compatible with an ad-hoc verison of GR_Hydro/EOS_Omni. Paths are currently hardcoded. 