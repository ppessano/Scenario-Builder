"This module generates the input files to run on MANDYOC"

import numpy as np
import pandas as pd
from builder.grid_interfaces import (
    create_horizontal_grid_2D,
    create_interfaces_2D,
    apply_parameters,
)
from builder.plots import plot_map_layout
from builder.parameters_files import create_velocity_file

###################
# Grid Parameters #
###################
model_name = "model"

x_size = 600 * 1e3  # Horizontal extent of the modelling area (m)
y_size = 400 * 1e3  # Vertical extent of the modelling area (m)
az_width = 100 * 1e3  # Width of the accommodation zone (m)

Nx = 501  # Grid points in the horizontal direction
Ny = 501  # Grid points in the vertical direction

Lx, Ly, x, y, X, Y = create_horizontal_grid_2D(x_size, y_size, az_width, Nx, Ny)

##############
# Interfaces #
##############
sz_width = 5.0 * 1e3  # Main shear zones mean width (m)
sub_width = 5.0 * 1e3  # Subsidiary faults mean width (m)

sz_coords = pd.read_csv("shear_zones_coordinates_transform.csv")
tr_coords = pd.read_csv("terranes_coordinates_transform.csv")

interfaces = create_interfaces_2D(
    sz_coords, sz_width, sub_width, az_width, x, x_size, Nx, Lx, Ly, tr_coords
)

####################
# Layer properties #
####################

# Parameters list: [C, rho, H, A, n, Q, V]

# Accommodation Zone rheological parameters
az_params = [
    0.00004,
    2500.0,
    0.0,
    8.574e-28,
    4.0,
    223.0e3,
    0.0,
]
# Host Rock rheological parameters
host_params = [
    1.0,
    2750.0,
    0.0,
    8.574e-28,
    4.0,
    222.0e3,
    0.0,
]
# Shear Zones rheological parameters
sz_params = [
    1.5,
    2900.0,
    0.0,
    3.981e-16,
    3.0,
    356.0e3,
    0.0,
]
# Terranes rheological parameters
tr_params = [
    0.8,
    2700.0,
    0.0,
    8.574e-28,
    4.0,
    222.0e3,
    0.0,
]

apply_parameters(interfaces, az_params, host_params, sz_params, tr_params)

###############
# Layout Plot #
###############

plot_map_layout(
    model_name=model_name, x=x, Lx=Lx, Ly=Ly, az_width=az_width, interfaces=interfaces
)

#####################
# Velocity Settings #
#####################

vel_array = np.array([1.0, 2.0, 2.0])  # mm/yr
phi_array = np.array([40, -10, -5])

create_velocity_file(
    model_name=model_name,
    vel_array=vel_array,
    phi_array=phi_array,
    Lx=Lx,
    Ly=Ly,
    X=X,
    Y=Y,
    Nx=Nx,
    Ny=Ny,
)

###########################
# Initial thermal setting #
###########################

T = X * 0.0

T = T * 0 + 100

np.savetxt("input_temperature_0.txt", np.reshape(T, (Nx * Ny)), header="T1\nT2\nT3\nT4")

######################
# Mandyoc Parameters #
######################

params = f"""
nx = {Nx}
nz = {Ny}
lx = {Lx}
lz = {Ly}


# Simulation options
multigrid                           = 1             # ok -> soon to be on the command line only
solver                              = direct        # default is direct [direct/iterative]
denok                               = 1.0E-13       # default is 1.0E-4
particles_per_element               = 100           # default is 81
particles_perturb_factor            = 0.0           # default is 0.5 [values are between 0 and 1]
rtol                                = 1.0E-7        # the absolute size of the residual norm (relevant only for iterative methods), default is 1.0E-5
RK4                                 = Euler         # default is Euler [Euler/Runge-Kutta]
Xi_min                              = 1.0E-5        # default is 1.0E-14
random_initial_strain               = 0.25          # default is 0.0
pressure_const                      = 50.0E6        # default is -1.0 (not used)
initial_dynamic_range               = True          # default is False [True/False]
periodic_boundary                   = False         # default is False [True/False]
high_kappa_in_asthenosphere         = True          # default is False [True/False]
K_fluvial                           = 2.0E-7        # default is 2.0E-7
m_fluvial                           = 1.0           # default is 1.0
sea_level                           = 0.0           # default is 0.0
basal_heat                          = 0.0           # default is -1.0

# Surface processes
sp_surface_tracking                 = False         # default is False [True/False]
sp_surface_processes                = False         # default is False [True/False]
sp_dt                               = 1.0E5         # default is 0.0
sp_d_c                              = 1.0           # default is 0.0
plot_sediment                       = False         # default is False [True/False]
a2l                                 = True          # default is True [True/False]

free_surface_stab                   = True          # default is True [True/False]
theta_FSSA                          = 0.5           # default is 0.5 (only relevant when free_surface_stab = True)

# Time constrains
step_max                            = 10000         # Maximum time-step of the simulation
time_max                            = 30.0e6        # Maximum time of the simulation [s]
dt_max                              = 5.0e3         # Maximum time between steps of the simulation [s]
step_print                          = 10            # Make file every <step_print>
sub_division_time_step              = 0.5           # default is 1.0
initial_print_step                  = 0             # default is 0
initial_print_max_time              = 1.0E6         # default is 1.0E6 [years]

# Viscosity
viscosity_reference                 = 1.0E26        # Reference viscosity [Pa.s]
viscosity_max                       = 1.0E24        # Maximum viscosity [Pa.s]
viscosity_min                       = 1.0E19        # Minimum viscosity [Pa.s]
viscosity_per_element               = constant      # default is variable [constant/variable]
viscosity_mean_method               = arithmetic    # default is harmonic [harmonic/arithmetic]
viscosity_dependence                = pressure      # default is depth [pressure/depth]

# External ASCII inputs/outputs
interfaces_from_ascii               = True          # default is False [True/False]
n_interfaces                        = {len(interfaces.keys())}    # Number of interfaces int the interfaces.txt file
variable_bcv                        = False         # default is False [True/False]
temperature_from_ascii              = True          # default is False [True/False]
velocity_from_ascii                 = True          # default is False [True/False]
binary_output                       = False         # default is False [True/False]
sticky_blanket_air                  = True          # default is False [True/False]
precipitation_profile_from_ascii    = False         # default is False [True/False]
climate_change_from_ascii           = False         # default is False [True/False]

print_step_files                    = True          # default is True [True/False]
checkered                           = False         # Print one element in the print_step_filesdefault is False [True/False]

sp_mode                             = 5             # default is 1 [0/1/2]

geoq                                = on            # ok
geoq_fac                            = 100.0         # ok

# Physical parameters
temperature_difference              = 1500.         # ok
thermal_expansion_coefficient       = 3.28e-5       # ok
thermal_diffusivity_coefficient     = 1.0e-6        # ok
gravity_acceleration                = 0.0           # ok
density_mantle                      = 3300.         # ok
external_heat                       = 0.0E-12       # ok
heat_capacity                       = 1250.         # ok

non_linear_method                   = on            # ok
adiabatic_component                 = off           # ok
radiogenic_component                = off           # ok

# Velocity boundary conditions
top_normal_velocity                 = fixed         # ok
top_tangential_velocity             = fixed         # ok
bot_normal_velocity                 = fixed         # ok
bot_tangential_velocity             = fixed         # ok
left_normal_velocity                = fixed         # ok
left_tangential_velocity            = fixed         # ok
right_normal_velocity               = fixed         # ok
right_tangential_velocity           = fixed         # ok

surface_velocity                    = 0.0E-2        # ok
multi_velocity                      = True          # default is False [True/False]

# Temperature boundary conditions
top_temperature                     = fixed         # ok
bot_temperature                     = fixed         # ok
left_temperature                    = fixed         # ok
right_temperature                   = fixed         # ok

rheology_model                      = 9             # ok
T_initial                           = 3             # ok
"""

with open("param.txt", "w") as f:
    for line in params.split("\n"):
        line = line.strip()
        if len(line):
            f.write(" ".join(line.split()) + "\n")
