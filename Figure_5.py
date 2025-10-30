# -*- coding: utf-8 -*-
"""
Code to produce figure 5 of Jones et al
@author: srgj201
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PGEN_functions import numerical_solve_LTO, numerical_solve_LT
from PGEN_functions import namelist, physical_constants
from PGEN_functions import calc_esat_from_T

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
ea   = 1000                     # Atmospheric vapour pressure      (Pa)
Ta   = np.linspace(10.0,45.0,N) # Atmospheric Temperature          (C)

# The model takes VPD as an input so calculate the VPD from Ta and ea
esat     = calc_esat_from_T( Ta ) # Saturated vapour pressure      (Pa)
vpd      = esat - ea              # Vapour pressure deficit        (Pa)

# Set up empty arrays to store calculated photosynthesis, stomatal conductance, ci, and leaf temperature
A_LTO = np.zeros(N)
A_LT  = np.zeros(N)

gs_LTO = np.zeros(N)
gs_LT  = np.zeros(N)

ci_LTO = np.zeros(N)
ci_LT  = np.zeros(N)

Tl_LTO = np.zeros(N)
Tl_LT  = np.zeros(N)


# =============================================================================
# Calculate optimum
# =============================================================================
# Iterate through the atmospheric temperature (and vpd) values and calculate optima
for i in range(len(Ta)):
    A_LTO[i], gs_LTO[i], ci_LTO[i], Tl_LTO[i] = numerical_solve_LTO( nl, pc, Ta[i], ca, pa, oa, Is, ra, vpd[i], swp)
    A_LT[i], gs_LT[i], ci_LT[i], Tl_LT[i]     = numerical_solve_LT( nl, pc, Ta[i], ca, pa, oa, Is, ra, vpd[i], swp)


# =============================================================================
# Calculate effective g1
# =============================================================================
g1_LTO = ( ca / pa * gs_LTO / A_LTO - 1 ) * np.sqrt(vpd / 1000 ) # The factor of 1000 converts units of vpd to kPa
g1_LT = ( ca / pa * gs_LT / A_LT - 1 ) * np.sqrt(vpd / 1000 )


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
plt.plot( Tl_LTO, g1_LTO, color = line_colors['LTO'], label = line_labels['LTO'])
plt.plot( Tl_LT, g1_LT, color = line_colors['LT'], label = line_labels['LT'])
plt.xlabel('Leaf temperature ($\degree$C)',size = 12)
plt.ylabel('Effective $g_1$ parameter (kPa$^{0.5}$)',size = 12)
plt.legend(loc ='upper center', bbox_to_anchor = (0.5,-0.17),ncol = 2)
fig.savefig('Figures/Figure_5.jpg', dpi = 300, bbox_inches = 'tight')

