# Simulations for Reference-enchanced Single Particle Imaging

This code contains code for simulating reconstructions from variable-reference holographic data. The algorithm and simulation details are described in the following paper:
> [Reference-enhanced X-ray Single Particle Imaging, Optica **7**(6), 593-601 (2020)](https://doi.org/10.1364/OPTICA.391373)

The program is pure-Python and runs on a CUDA-enabled GPU using the CuPy library. Additionally, the following libraries are used 
 - `numpy` for CPU math
 - `h5py` for data storage
 - `scipy` for image filtration in Shrinkwrap
 - `mpi4py` to run the reconstruction on multiple GPUs

Typically `cupy` and `mpi4py` require custom installations depending upon system versions of CUDA and MPI respectively. Other packages can be installed easily using PyPi or Anaconda.

A default configuration file has been provided which has the same parameters as those described in the paper, along with an explanation of the various options.

## Quick start
To generate data similar to the paper, just run
```
./make_data.py
```
You can use the `-m` option to generate just the real-space object or the `-d` option to just generate the photon data from the object already generated.

The reconstruction can be run after generating the data by running
```
./emc.py 100
```
where the number refers to the number of iterations to run.

The output is saved in Numpy-format `.npy` files in the `data/` directory. For advanced options, run either command with the `-h` option.
