import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm
from mpi4py import MPI
import h5py as h5
import logging

logging.basicConfig(filename='logfile.log', level=logging.DEBUG)

# Initialize MPI and determine the rank and size of the communicator:
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"\nMPI on, I am rank {rank+1} of {size}\n")

comm.Barrier()

file = h5.File("/expanse/lustre/projects/ias102/yhu3/galaxy/snap_059.hdf5", 'r')

# Read the dataset
dens = file["density"][:]
ib = file["i_mag_field"][:]
jb = file["j_mag_field"][:]
kb = file["k_mag_field"][:]
iv = file["i_velocity"][:]
jv = file["j_velocity"][:]
kv = file["k_velocity"][:]
coords = file["coordinates"][:]
smooth_lengths = file["smoothing_length"][:]

# Define the center of the region of interest (in code units)
center = [39881.3, 34909.6, 37374.6]  # h^-1 ckpc, h = 0.704, ckpc is comoving distance

# Define the width of the region of interest (in code units)
width = 50  # code unit

# Define the dimensions of the 3D cube
dims = [512, 512, 512]

# Create an empty numpy array with the specified dimensions
x_grid = np.linspace(center[0] - width / 2, center[0] + width / 2, dims[0])
y_grid = np.linspace(center[1] - width / 2, center[1] + width / 2, dims[1])
z_grid = np.linspace(center[2] - width / 2, center[2] + width / 2, dims[2])

# Create a 3D meshgrid
X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

# Initialize the cubes
density_cube = np.zeros(dims)
ib_cube = np.zeros(dims)
jb_cube = np.zeros(dims)
kb_cube = np.zeros(dims)
iv_cube = np.zeros(dims)
jv_cube = np.zeros(dims)
kv_cube = np.zeros(dims)

# Compute the indices of the grid cells. This is a one-time computation.
indices = np.indices(dims).reshape(3, -1).T

# Compute the coordinates of the grid cell centers. This is also a one-time computation.
grid_coords = np.array([X, Y, Z]).reshape(3, -1).T
X = 0;
Y = 0;
Z = 0;

# Compute the KDTree for the grid cell centers. This is also a one-time computation.
tree = cKDTree(grid_coords)

# Determine which particles this rank is responsible for
n_particles = np.shape(coords)[0]
particles_per_rank = n_particles // size
extra_particles = n_particles % size

# If the number of particles isn't evenly divisible by the number of ranks,
# distribute the remaining particles among the first 'extra_particles' ranks
if rank < extra_particles:
    start_particle = rank * (particles_per_rank + 1)
    end_particle = start_particle + particles_per_rank + 1
else:
    start_particle = rank * particles_per_rank + extra_particles
    end_particle = start_particle + particles_per_rank
        
logging.info(f"Rank {rank} starts looping over particles, total: {end_particle - start_particle}")    
# Iterate over particles, divided among ranks
for p in range(start_particle, end_particle):
    # Find the grid cells within the smoothing length of the particle.
    within_smoothing_length = tree.query_ball_point(coords[p], smooth_lengths[p])
    # Compute the smoothing kernel for these grid cells.
    norm = 0;
    for index in within_smoothing_length:
        dx = grid_coords[index] - coords[p]
        #kernel = np.exp(-np.sum(dx ** 2) / (2 * smooth_lengths[p] ** 2))
        kernel = 1.0 - np.sum(dx ** 2) / (smooth_lengths[p] ** 2)
        norm += kernel
    for index in within_smoothing_length:
        dx = grid_coords[index] - coords[p]
        kernel = 1.0 - np.sum(dx ** 2) / (smooth_lengths[p] ** 2)
        # Add the particle's contribution to the density of the grid cell.
        density_cube[np.unravel_index(index, dims)] += dens[p] * kernel / norm
        ib_cube[np.unravel_index(index, dims)] += ib[p] * kernel / norm
        jb_cube[np.unravel_index(index, dims)] += jb[p] * kernel / norm
        kb_cube[np.unravel_index(index, dims)] += kb[p] * kernel / norm
        iv_cube[np.unravel_index(index, dims)] += (iv[p]-96.37495304968748) * kernel / norm
        jv_cube[np.unravel_index(index, dims)] += (jv[p]+139.25884679321626) * kernel / norm
        kv_cube[np.unravel_index(index, dims)] += (kv[p]-85.24431181119564) * kernel / norm
logging.info(f"Rank {rank} finished looping over all particles")

# Collect results from all ranks
if rank == 0:
    total_density_cube = np.zeros_like(density_cube)
    total_ib_cube = np.zeros_like(ib_cube)
    total_jb_cube = np.zeros_like(jb_cube)
    total_kb_cube = np.zeros_like(kb_cube)
    total_iv_cube = np.zeros_like(iv_cube)
    total_jv_cube = np.zeros_like(jv_cube)
    total_kv_cube = np.zeros_like(kv_cube)
else:
    total_density_cube = None
    total_ib_cube = None
    total_jb_cube = None
    total_kb_cube = None
    total_iv_cube = None
    total_jv_cube = None
    total_kv_cube = None

comm.Barrier()
# Each process will call comm.Reduce, but only root process will receive the results
comm.Reduce([density_cube, MPI.DOUBLE], [total_density_cube, MPI.DOUBLE], op=MPI.SUM, root=0)
comm.Reduce([ib_cube, MPI.DOUBLE], [total_ib_cube, MPI.DOUBLE], op=MPI.SUM, root=0)
comm.Reduce([jb_cube, MPI.DOUBLE], [total_jb_cube, MPI.DOUBLE], op=MPI.SUM, root=0)
comm.Reduce([kb_cube, MPI.DOUBLE], [total_kb_cube, MPI.DOUBLE], op=MPI.SUM, root=0)
comm.Reduce([iv_cube, MPI.DOUBLE], [total_iv_cube, MPI.DOUBLE], op=MPI.SUM, root=0)
comm.Reduce([jv_cube, MPI.DOUBLE], [total_jv_cube, MPI.DOUBLE], op=MPI.SUM, root=0)
comm.Reduce([kv_cube, MPI.DOUBLE], [total_kv_cube, MPI.DOUBLE], op=MPI.SUM, root=0)

if rank == 0:
    #Create a FITS header
    hdr = fits.Header()
    hdr['SIMUL'] = 'Arepo'
    hdr['CTYPE1'] = 'X'
    hdr['CTYPE2'] = 'Y'
    hdr['CTYPE3'] = 'Z'
    hdr['WIDTH'] = float(width)
    hdr['CRVAL1'] = float(center[0])
    hdr['CRVAL2'] = float(center[1])
    hdr['CRVAL3'] = float(center[2])
    hdr['CDELT1'] = float(width / dims[0])
    hdr['CDELT2'] = float(width / dims[1])
    hdr['CDELT3'] = float(width / dims[2])
    hdr['CRPIX1'] = dims[0] // 2
    hdr['CRPIX2'] = dims[1] // 2
    hdr['CRPIX3'] = dims[2] // 2
    hdr['BUNIT'] = 'code_unit'
    hdr['COMMENT'] = "3D density cube with 1 - r^2/h^2 smoothing"

    # Create a new HDUList object
    hdulist = fits.HDUList()

    # Add each data cube as a separate HDU to the HDUList
    hdulist.append(fits.PrimaryHDU(data=total_density_cube, header=hdr))

    # Create headers for the other HDUs
    hdr_ib = fits.Header()
    hdr_ib['EXTNAME'] = 'ib'
    hdulist.append(fits.ImageHDU(data=total_ib_cube, header=hdr_ib))

    hdr_jb = fits.Header()
    hdr_jb['EXTNAME'] = 'jb'
    hdulist.append(fits.ImageHDU(data=total_jb_cube, header=hdr_jb))

    hdr_kb = fits.Header()
    hdr_kb['EXTNAME'] = 'kb'
    hdulist.append(fits.ImageHDU(data=total_kb_cube, header=hdr_kb))

    hdr_iv = fits.Header()
    hdr_iv['EXTNAME'] = 'iv'
    hdulist.append(fits.ImageHDU(data=total_iv_cube, header=hdr_iv))

    hdr_jv = fits.Header()
    hdr_jv['EXTNAME'] = 'jv'
    hdulist.append(fits.ImageHDU(data=total_jv_cube, header=hdr_jv))

    hdr_kv = fits.Header()
    hdr_kv['EXTNAME'] = 'kv'
    hdulist.append(fits.ImageHDU(data=total_kv_cube, header=hdr_kv))

    # Save the FITS file
    hdulist.writeto('/expanse/lustre/projects/ias102/yhu3/galaxy/snap_059_28kpc_smooth_512_gv_sq.fits', overwrite=True)
