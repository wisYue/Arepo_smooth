import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# READ THE FILE
simu = 2
filename = f"/mnt/d/ML_SP/sub-1{simu}-hdfaa.007"
file = h5.File(filename, 'r')

# Infos for title
MS_array = np.array([0.58, 0.66, 0.62, 0.61, 0.59])
MA_array = np.array([1.21, 0.26, 0.50, 0.79, 1.02])
#beta_array = np.array([8.7, 0.08, 0.69, 2.13, 4.5, 7.6, 50.0, 6.08, 3.67, 0.64, 0.4])
MS = MS_array[simu]
MA = MA_array[simu]
#beta = beta_array[simu]
size = 792

# TRANSPOSED MAGNETIC FIELD (Julia to Python)
config = (2, 1, 0)
magx = np.transpose(file["i_mag_field"][:], config)
magy = np.transpose(file["j_mag_field"][:], config)
magz = np.transpose(file["k_mag_field"][:], config)

##################################################################################################################
"""
    Functions to solve field line equations.
"""

def trilinear_interpolation(data, x, y, z):
    """
    Trilinear interpolation with periodic boundary conditions.

    Args:
        data (numpy.ndarray): Data cube.
        x (array): x coordinates for the interpolation.
        y (array): y coordinates for the interpolation.
        z (array): z coordinates for the interpolation.

    Returns:
        interpolated_value (float).
    """
    shape_x, shape_y, shape_z = data.shape

    x = x % shape_x
    y = y % shape_y
    z = z % shape_z

    # Adjust negative coordinates
    x[x < 0] += shape_x
    y[y < 0] += shape_y
    z[z < 0] += shape_z

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    z0 = np.floor(z).astype(int)

    x1 = (x0 + 1) % shape_x
    y1 = (y0 + 1) % shape_y
    z1 = (z0 + 1) % shape_z

    xd = x - x0
    yd = y - y0
    zd = z - z0

    c000 = data[x0, y0, z0]
    c001 = data[x0, y0, z1]
    c010 = data[x0, y1, z0]
    c011 = data[x0, y1, z1]
    c100 = data[x1, y0, z0]
    c101 = data[x1, y0, z1]
    c110 = data[x1, y1, z0]
    c111 = data[x1, y1, z1]

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    c = c0 * (1 - zd) + c1 * zd

    return c

def rk4singlestep(ds, r0):
    """
        RK4 method to solve magnetic field line equation system.

        Args:
            ds (float): length step along the field line.
            r0 (array): initial positions of field lines.

        Returns:
            rout (array).

        r0 .shape = (num_fl, 3) where num_fl = number fo field lines

        k1-4.shape = (3, num_fl)
        k.shape = (3, num_fl)
        k.T = (num_fl, 3)

        rout.shape = (num_fl, 3)
    """
    # Trilinear interpolation  
    k1 = np.array([
        trilinear_interpolation(magx, r0[:, 0], r0[:, 1], r0[:, 2]),
        trilinear_interpolation(magy, r0[:, 0], r0[:, 1], r0[:, 2]),
        trilinear_interpolation(magz, r0[:, 0], r0[:, 1], r0[:, 2])
    ])
    k1 /= np.linalg.norm(k1, axis=0)

    # Midpoint estimate using k1
    r1 = r0 + 0.5 * ds * k1.T
    k2 = np.array([
        trilinear_interpolation(magx, r1[:, 0], r1[:, 1], r1[:, 2]),
        trilinear_interpolation(magy, r1[:, 0], r1[:, 1], r1[:, 2]),
        trilinear_interpolation(magz, r1[:, 0], r1[:, 1], r1[:, 2])
    ])
    k2 /= np.linalg.norm(k2, axis=0)

    # Another midpoint estimate using k2
    r2 = r0 + 0.5 * ds * k2.T
    k3 = np.array([
        trilinear_interpolation(magx, r2[:, 0], r2[:, 1], r2[:, 2]),
        trilinear_interpolation(magy, r2[:, 0], r2[:, 1], r2[:, 2]),
        trilinear_interpolation(magz, r2[:, 0], r2[:, 1], r2[:, 2])
    ])
    k3 /= np.linalg.norm(k3, axis=0)

    # End point estimate using k3
    r3 = r0 + ds * k3.T
    k4 = np.array([
        trilinear_interpolation(magx, r3[:, 0], r3[:, 1], r3[:, 2]),
        trilinear_interpolation(magy, r3[:, 0], r3[:, 1], r3[:, 2]),
        trilinear_interpolation(magz, r3[:, 0], r3[:, 1], r3[:, 2])
    ])
    k4 /= np.linalg.norm(k4, axis=0)

    k = (k1 + 2.0*k2 + 2.0*k3 + k4) / np.linalg.norm(k1 + 2.0*k2 + 2.0*k3 + k4, axis=0)
    k = k.T
    
    rout = r0 + ds*k

    return rout

##################################################################################################################
##################################################################################################################
"""
    Study progressive separation of pairs of field lines initially close to each other (within a correlation patch).
    Distance between both field lines is determined along the abcissa along the field lines (s).
"""

# S = total length of field lines, L_injection = injection scale, L_c = correlation length
S = 50.0*size
L_injection = size/2 # size of the box divided by 2
L_c = L_injection / 5 # Kolmogorov like turbulence
L_dissipation = 20
ds = 0.1

# Calculation of transition length
if MA > 1:
    L_tr = L_injection/MA**3
else:
    L_tr = L_injection*MA**2

# Number of pairs of field lines 
num_pairs = 1000

# Random initial position for the first field line of each pair
fl1 = np.random.randint(0, size, size=(num_pairs, 3))
fl2 = fl1.copy()

# As turbulence is considered axisymetric, initial position of the second field line
# of each pair has the same initial position as the first one but drifted along x axis 
# by a distance within dissipation scale (~20 grid cells) and correlation scale (size of the box / 10)
for i in range(fl2.shape[0]):
    #fl2[i, 0] += np.random.randint(L_dissipation, np.floor(L_c))
    #fl2[i, 0] += np.random.randint(1, 10)
    fl2[i, 0] += 1

# Containers
distances = []

# Parallelized version
num_curv_pts = int(S/ds)
DS = np.arange(num_curv_pts)
num_graph_pts = 128
indices = np.unique(np.floor(np.logspace(0, np.log10(num_curv_pts), num_graph_pts)))
indices = np.abs(indices - indices[0]) / (indices[-1] - indices[0])
minimum_height = 50000 # only conserve field lines with height above this range
z_steps = minimum_height * indices
s_steps = np.floor(z_steps).astype(int)

# Solve field line equations for each pair and save coordinates along s
    # For fl1
R1 = np.zeros((num_pairs, 3, num_curv_pts))
R1[:, :, 0] = fl1
rin1 = fl1.copy()
for k in range(1, num_curv_pts):
    rout1 = rk4singlestep(ds, rin1)
    R1[:, :, k] = rout1
    rin1 = rout1

# For fl2
R2 = np.zeros((num_pairs, 3, num_curv_pts))
R2[:, :, 0] = fl2
rin2 = fl2.copy()
for k in range(1, num_curv_pts):
    rout2 = rk4singlestep(ds, rin2)
    R2[:, :, k] = rout2
    rin2 = rout2


# Study Richardson diffusion along field line abcissa
number_of_considered_pairs = 0
for k in range(num_pairs):

        l1 = R1[k]
        l2 = R2[k]

        # Only conserve field lines with height above this range
        if (l1[2, -1] - l1[2, 0]) < (minimum_height + 1) or (l2[2, -1] - l2[2, 0]) < (minimum_height + 1):
            pass
        else:
            number_of_considered_pairs += 1

        # Get values 
        l1_sampled = l1[:, s_steps]
        l2_sampled = l2[:, s_steps]

        # Measure distance at each steps
        r = l2_sampled - l1_sampled
        d = np.linalg.norm(r, axis=0)

        distances.append(d)

# Post-processing
Distances = np.array(distances)
ensemble_average = np.sqrt(np.mean(Distances ** 2, axis=0))

# Plot data
x = s_steps / L_injection
y = ensemble_average / L_injection
plt.figure()
plt.scatter(x, y, s=5, color='k')
plt.title(f'Yue Hu : $M_A = {MA}$ | Along s')
plt.xlabel(r'$s/L_{\rm inj}$')
plt.ylabel(r'$\langle (\delta r)^2 \rangle^{1/2} / L_{\rm inj}$')
plt.xscale('log')
plt.yscale('log')
x_ticks = [1e-2, 1e-1, 1e0, 1e1, 1e2]
plt.xticks(x_ticks, x_ticks)

# The different slopes
x0, y0 = (s_steps / L_injection)[50], ensemble_average[50]/L_injection
slope = [1/2, 3/4, 3/2]
for s in slope:
    x = np.logspace(np.log10(x0), np.log10(4), 100)
    C = np.log10(y0) - s * np.log10(x0)
    y = 10**(s * np.log10(x) + C)

    plt.plot(x, y, label=f'slope = {s}')

# Plot typical scales
plt.axhline(y=L_tr/L_injection, color='r', linestyle='--', label=r'$L_{\rm tr}/L_{\rm inj}$'+ f' = {L_tr/L_injection:.2f}')
    # L dissipation
plt.axhline(y=L_dissipation/L_injection, color='c', linestyle='dashed', label=r'$L_{\rm diss}/L_{\rm inj}$'+ f' = {L_dissipation/L_injection:.2f}')
    # L correlation
#plt.axvline(x=L_injection/L_injection, color='k', linestyle='dashed')
    # Exit of the box
#plt.axvline(x=size/L_injection, color='k', linestyle='dashed', label=r'$L_{\rm box}/L_{\rm inj}$'+ f' = {size/L_injection:.2f}')

plt.legend()
plt.savefig(f"/mnt/d/sub1{simu}_line_diffusion_Ty.pdf",dpi=300)
plt.close()

#plt.show()
