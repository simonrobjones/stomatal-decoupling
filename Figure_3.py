# -*- coding: utf-8 -*-
"""
Code to produce Figure 3 of Jones et al
@author: srgj201
"""

import matplotlib.pyplot as plt
import numpy as np
from PGEN_functions import numerical_solve_LTO, numerical_solve_LT, numerical_solve_AT
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
A_AT  = np.zeros(N)

gs_LTO = np.zeros(N)
gs_LT  = np.zeros(N)
gs_AT  = np.zeros(N)

ci_LTO = np.zeros(N)
ci_LT  = np.zeros(N)
ci_AT  = np.zeros(N)

Tl_LTO = np.zeros(N)
Tl_LT  = np.zeros(N)
Tl_AT  = np.zeros(N)

# =============================================================================
# Calculate optimum
# =============================================================================
# Iterate through the atmospheric temperature (and vpd) values and calculate optima
for i in range(len(Ta)):
    A_LTO[i], gs_LTO[i], ci_LTO[i], Tl_LTO[i] = numerical_solve_LTO( nl, pc, Ta[i], ca, pa, oa, Is, ra, vpd[i], swp)
    A_LT[i], gs_LT[i], ci_LT[i], Tl_LT[i]     = numerical_solve_LT( nl, pc, Ta[i], ca, pa, oa, Is, ra, vpd[i], swp)
    A_AT[i], gs_AT[i], ci_AT[i], Tl_AT[i]     = numerical_solve_AT( nl, pc, Ta[i], ca, pa, oa, Is, ra, vpd[i], swp)


fig,axs = plt.subplots(nrows = 1,ncols = 3,figsize = (20,4),sharey = True)
plt.subplots_adjust(wspace = 0.02,hspace = 0.25)
if type(axs) == np.ndarray:
    axs = axs.reshape(-1)
else:
    axs = [axs]
    

im = axs[0].scatter( A_LTO * 1e6, gs_LTO, c = Ta, marker = 'o', cmap ='coolwarm', edgecolors = 'black', zorder = 0 )
axs[1].scatter( A_LT * 1e6, gs_LT, c = Ta, marker = 'o', cmap ='coolwarm', edgecolors = 'black', zorder = 0 )
axs[2].scatter( A_AT * 1e6, gs_AT, c = Ta, marker = 'o', cmap ='coolwarm', edgecolors = 'black', zorder = 0 )

# cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.7])
# plt.colorbar(im, cax = cbar_ax, label = 'Atmospheric temperature ($\degree C$)', orientation = 'vertical')    

axs[0].set_title('Leaf Temperature within Optimisation (LTO)',size = 15 )
axs[1].set_title('Leaf Temperature (LT)',size = 15 )
axs[2].set_title('Air Temperature (AT)',size = 15 )

axs[0].set_ylabel('$g_{sc}$ (mol/m$^2$/s)',size = 15)
for ax in axs:
    ax.set_xlabel( '$A$ ($\mu$mol/m$^2$/s)', size = 15 )

cbar_ax = fig.add_axes([0.15, -0.15, 0.7, 0.06])
cbar = plt.colorbar(im, cax = cbar_ax, orientation = 'horizontal')    
cbar.set_label('Atmospheric temperature ($\degree C$)',size = 15)

fig.savefig('Figures/Figure_3.jpg', dpi = 300, bbox_inches = 'tight')
