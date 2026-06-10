# -*- coding: utf-8 -*-
"""
Code to produce figure 6 of Jones et al
@author: srgj201
"""
import matplotlib.pyplot as plt
import numpy as np
from PGEN_functions_complete import numerical_solve_LTO
from PGEN_functions_complete import namelist, physical_constants
from PGEN_functions_complete import calc_esat_from_T
Fig_path = 'Figures/'

# =============================================================================
# Set up environmental conditions
# =============================================================================
nl   = namelist()               # Class containing all model parameters
pc   = physical_constants()     # Class containing all physical constants
N    = 50                       # Number of temperature points to plot

ca   = 40.0                     # Atmospheric CO2 partial pressure (Pa)
swp  = -0.1                     # Soil water potential             (MPa)
pa   = 101325.0                 # Atmospheric pressure             (Pa)
oa   = 20900.0                  # Atmospheric O2 partial pressure  (Pa)
Is   = 500                      # Absorbed shortwave radiation     (Wm-2)
ra   = 10                       # Aerodymaic resistance            (s m-1)
ea   = 500                      # Atmospheric vapour pressure      (Pa)
Ta   = np.linspace(10.0,45.0,N) # Atmospheric Temperature          (C)

# The model takes VPD as an input so calculate the VPD from Ta and ea
esat     = calc_esat_from_T( Ta ) # Saturated vapour pressure      (Pa)
vpd      = esat - ea              # Vapour pressure deficit        (Pa)

# Set up empty arrays to store calculated photosynthesis, stomatal conductance, ci, and leaf temperature
A_LTO = np.zeros(N)
gs_LTO = np.zeros(N)
ci_LTO = np.zeros(N)
Tl_LTO = np.zeros(N)


# =============================================================================
# Calculate optimum
# =============================================================================
# Iterate through the atmospheric temperature (and vpd) values and calculate optima
for i in range(len(Ta)):
    A_LTO[i], gs_LTO[i], ci_LTO[i], Tl_LTO[i] = numerical_solve_LTO( nl, pc, Ta[i], ca, pa, oa, Is, ra, vpd[i], swp)



# =============================================================================
# Calculate effective g1
# =============================================================================
g1_LTO = ( ca / pa * gs_LTO / A_LTO - 1 ) * np.sqrt(vpd / 1000 ) # The factor of 1000 converts units of vpd to kPa


# =============================================================================
# Set up figure
# =============================================================================
line_colors = {'LTO':'#FFB000',
               'LT':'#DC267F',
               'AT':'#648FFF'}
line_labels = {'LTO':'Leaf Temperature within Optimisation (LTO)',
               'LT':'Leaf Temperature (LT)',
               'AT':'Air Temperature (AT)'}
    
fig = plt.figure()
plt.gca().set_yscale('log')
plt.plot( Ta, g1_LTO, color = line_colors['LTO'], label = line_labels['LTO'])
plt.xlabel('Leaf temperature ($\degree$C)',size = 12)
plt.ylabel('Effective $g_1$ parameter (kPa$^{0.5}$)',size = 12)


fig.savefig(Fig_path + 'Figure_6.jpg', dpi = 300, bbox_inches = 'tight')
