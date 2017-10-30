"""
Gravity module 

This module exposes a routine to calculate a field
of Gravitational Potential Energy for an arbitrary
list of positions and masses.
"""

import numpy as np
from numba import cuda
from math import ceil, sqrt
import pdb
from .field import FieldException, Field

GRAV_CONST=1.

@cuda.jit
def _gravityField(pos, mass, space_boundary, pot_energy):
    """
    GPU-Kernel to calculate the gravitational potential field.

    The dimensions used for calling this kernel should be:
        3-D of shape (ceil(M_x/16)*N_p, ceil(M_y/8), ceil(M_z/8)). 
        3-D of shape (16, 8, 8). 
    Should be called by a 2-D block corresponding to the number of elements
    in each dimension of U. 

    Parameters
    ----------
    pos : array_like
        array of positions, of shape (N_p, 3)
    mass : array_like
        array of masses, of shape (N_p)
    space_boundary : array_like
        bounds of the potential field. shape (2, 2, 2)
    pot_energy : array_like
        potential field grid is written here. shape (M_x, M_y, M_z)
    """

    # Store off dimensions
    M_x = pot_energy.shape[0]
    M_y = pot_energy.shape[1]
    M_z = pot_energy.shape[2]
    N_p = len(mass)

    # Get the indices
    i_x = cuda.blockIdx.x//N_p * cuda.blockDim.x + cuda.threadIdx.x
    i_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    i_z = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    i_p = cuda.blockIdx.x%N_p

    # If a valid point
    if i_x < M_x and i_y < M_y and i_z < M_z:
        # Grab the point's coordinate
        p_x = pos[i_p, 0]
        p_y = pos[i_p, 1]
        p_z = pos[i_p, 2]

        # Calculate how far it is from the grid point
        dx2 = 0.
        dy2 = 0.
        dz2 = 0.
        if M_x > 1:
            bl = space_boundary[0, 0]
            br = space_boundary[0, 1]
            dx2 = (bl + i_x/(M_x - 1.) * (br - bl) - p_x)**2
        if M_y > 1:
            bl = space_boundary[1, 0]
            br = space_boundary[1, 1]
            dy2 = (bl + i_y/(M_y - 1.) * (br - bl) - p_y)**2
        if M_z > 1:
            bl = space_boundary[2, 0]
            br = space_boundary[2, 1]
            dz2 = (bl + i_z/(M_z - 1.) * (br - bl) - p_z)**2

        # calculate the potential energy contribution
        if dx2 + dy2 + dz2 > 1e-8:
            pot = - GRAV_CONST * mass[i_p] / sqrt( dx2 + dy2 + dz2 )
        else:
            # deal with the case where the particle is ON a node
            pot = - np.inf

        # Update the potential energy field
        cuda.atomic.add(pot_energy, (i_x, i_y, i_z), pot)

def _gravityField_np(pos, mass, space_boundary, pot_energy):
    xs = np.linspace(space_boundary[0,0], space_boundary[0,1], pot_energy.shape[0])
    ys = np.linspace(space_boundary[1,0], space_boundary[1,1], pot_energy.shape[1])
    zs = np.linspace(space_boundary[2,0], space_boundary[2,1], pot_energy.shape[2])
    XS, YS, ZS = np.meshgrid(xs, ys, zs, indexing='ij')

    for i_p, m_i in enumerate(mass):
        pot_energy -= GRAV_CONST * m_i / np.sqrt((pos[i_p, 0] - XS)**2 + (pos[i_p, 1] - YS)**2 + (pos[i_p, 2] - ZS)**2)

    return pot_energy

def calcGravitationalPotential(pos, mass, char_len=[-1, -1, -1], useCuda=True):
    """
    Calculates a field of gravitational potential.

    Parameters
    ----------
    pos : array_like
        array of positions, of shape (N_p, 3)
    mass : array_like
        array of masses, of shape (N_p)
    char_len : array_like
        characteristic lengths of grid dimensions. 
        Grid points in the potential energy field will be separated by this amount.
        There are three cases per dimension:
            < 0 (default) : this dimension will have 128 grid points if
                            positions vary in this dimension, otherwise will be flat
            = 0           : this dimension will have 1 grid point (flat in this dimension)
            > 0           : this dimension will have a grid delta as specified,
                            unless positions do not vary in this dimension

    Returns
    -------
    gravitational_potential_field : field
        The gravitaional field as represented by a field object
    """
    ## Numpy-ify inputs ##
    char_len = np.array(char_len, np.float)
    # Make all gpu bound values single precision
    pos = np.array(pos, np.float32)
    pos = pos.astype(np.float32)
    mass = np.array(mass, np.float32)
    mass = mass.astype(np.float32)

    ## Verify inputs ##
    if pos.shape[0] != mass.shape[0]:
        raise GravityException("Position and mass do not have same number of particles")
    if pos.ndim != 2 and pos.shape[1] != 3:
        raise GravityException("Position is not a (N_p, 3) array")
    if mass.ndim != 1:
        raise GravityException("Mass is not a (N_p) array")

    ## Calculate the bounds ##
    grid_size = [0, 0, 0]
    # get the bounding 'cube' around the particles
    space_boundary = list(zip(pos.min(0), pos.max(0)))
    # pad the bounding 'cube' so it has two grid points beyond
    for idx, bnd in enumerate(space_boundary):
        rng = bnd[1] - bnd[0]
        if char_len[idx] == 0 or rng == 0:
            # If the dimension is flat
            grid_size[idx] = 1
        else:
            # for dimensions with depth
            if char_len[idx] <= 0:
                # Choose a characteristic length which would yield 128 as the appropriate grid size
                char_len[idx] = rng*1.01/123 

            grid_size[idx] = 2*( ceil(rng*.505/ char_len[idx]) + 2 )
            half_space = (grid_size[idx]/2 - .5) * char_len[idx]
            avg = .5*(bnd[1] + bnd[0])
            space_boundary[idx] = (avg - half_space, avg + half_space)

    # Numpy-ify and the make the space boundary single precision
    space_boundary = np.array(space_boundary)
    space_boundary = space_boundary.astype(np.float32)

    ## Calculate field values ##
    # instantiate the potential energy
    pot_energy = np.zeros(grid_size, np.float32)
    if useCuda:
        # Move relevant into GPU memory
        gpu_pos = cuda.to_device(pos)
        gpu_mass = cuda.to_device(mass)
        gpu_space_boundary = cuda.to_device(space_boundary)
        gpu_pot_energy = cuda.to_device(pot_energy)

        # Calculate the gravitational field
        threads_per_block = (16, 8 ,8)
        blocks_per_grid = (ceil(grid_size[0]/16)*len(mass), 
                           ceil(grid_size[1]/8), 
                           ceil(grid_size[2]/8))
        _gravityField[blocks_per_grid, threads_per_block](
                      gpu_pos, gpu_mass, gpu_space_boundary, gpu_pot_energy)

        # Move the gpu back to potential energy back to the host
        gpu_pot_energy.copy_to_host(pot_energy)
    else:
        pot_energy = _gravityField_np(pos, mass, space_boundary, pot_energy)


    ## Construct the field object to return ##
    return space_boundary, pot_energy

class GravityException(FieldException):
    pass
