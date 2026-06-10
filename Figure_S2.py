# -*- coding: utf-8 -*-
"""
Code to produce figure  of Jones et al
@author: srgj201
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PGEN_functions_complete import numerical_solve_LTO, numerical_solve_LT, numerical_solve_AT
from PGEN_functions_complete import namelist, physical_constants

Fig_path = 'C:/Users/srgj201/OneDrive - University of Exeter/Documents/Postdoc/SOX/High_T_decoupling/Jones et al 25/Figures/Revision_2/'

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
Is   = 500.0                    # Absorbed shortwave radiation     (Wm-2)
ra   = 10.0                     # Aerodymaic resistance            (s m-1)
vpd  = 1000.0                   # Atmospheric vapour pressure      (Pa)
Ta   = np.linspace(10.0,45.0,N) # Atmospheric Temperature          (C)

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
A_LTO, gs_LTO, ci_LTO, Tl_LTO = numerical_solve_LTO( nl, pc, Ta, ca, pa, oa, Is, ra, vpd, swp = swp )
A_LT,  gs_LT,  ci_LT,  Tl_LT  = numerical_solve_LT( nl, pc, Ta, ca, pa, oa, Is, ra, vpd, swp = swp)
A_AT,  gs_AT,  ci_AT,  Tl_AT  = numerical_solve_AT( nl, pc, Ta, ca, pa, oa, Is, ra, vpd, swp = swp )

# =============================================================================
# Figure
# =============================================================================
fig,axs = plt.subplots(nrows = 2,ncols = 2,figsize = (18,11))
plt.subplots_adjust(wspace = 0.2,hspace = 0.25)
if type(axs) == np.ndarray:
    axs = axs.reshape(-1)
else:
    axs = [axs]

line_colors = {'LTO':'#FFB000',
               'LT':'#DC267F',
               'AT':'#648FFF'}
line_labels = {'LTO':'Leaf Temperature within Optimisation (LTO)',
               'LT':'Leaf Temperature (LT)',
               'AT':'Air Temperature (AT)'}
    
line_style = 'none'
plot_schemes = ['LTO','LT','AT']
custom_lines  = [Line2D([], [], color = line_colors[scheme], ls = line_style, marker = 'o' , markerfacecolor = 'none') for scheme in plot_schemes]
fig.legend(custom_lines,[line_labels[s] for s in plot_schemes],loc = 'upper center',bbox_to_anchor = (0.5,0.05), fontsize = 12, ncol = len(plot_schemes))


# Photosynthesis
axs[0].plot( Ta, 1e6 * A_LTO, color = line_colors['LTO'],
            ls = line_style, marker = 'o', markerfacecolor = 'none' )
axs[0].plot( Ta, 1e6 * A_LT, color = line_colors['LT'],
              ls = line_style, marker = 'o', markerfacecolor = 'none' )
axs[0].plot( Ta, 1e6 * A_AT, color = line_colors['AT'],
            ls = line_style, marker = 'o', markerfacecolor = 'none' )
axs[0].axhline( 0, color = 'black', lw = 1.0 )
axs[0].set_ylabel('$A$ ($\mu$mol/m$^2$/s)',size = 15)
axs[0].set_xlabel('Atmospheric Temperature ($\degree$C)',size = 15)
axs[0].text( 0.01, 0.98, '(a)', transform = axs[0].transAxes, 
            ha = 'left', va = 'top', size = 12)

# Stomatal conductance
axs[1].plot( Ta, gs_LTO, color = line_colors['LTO'], 
             ls = line_style, marker = 'o', markerfacecolor = 'none' )
axs[1].plot( Ta, gs_LT, color = line_colors['LT'], 
              ls = line_style, marker = 'o', markerfacecolor = 'none' )
axs[1].plot( Ta, gs_AT, color = line_colors['AT'], 
              ls = line_style, marker = 'o', markerfacecolor = 'none' )
axs[1].axhline( 0, color = 'black', lw = 1.0 )
axs[1].set_ylabel('$g_{sc}$ (mol/m$^2$/s)',size = 15)
axs[1].set_xlabel( 'Atmospheric Temperature ($\degree$C)', size = 15 )
axs[1].text( 0.01, 0.98, '(b)', transform = axs[1].transAxes, 
             ha = 'left', va = 'top', size = 12 )

# dT
axs[2].plot( Ta, Tl_LTO - Ta, line_colors['LTO'],
             ls = line_style, marker = 'o', markerfacecolor = 'none' )
axs[2].plot( Ta, Tl_LT - Ta, line_colors['LT'],
              ls = line_style, marker = 'o', markerfacecolor = 'none' )
axs[2].plot( Ta, Tl_AT - Ta, line_colors['AT'],
              ls = line_style, marker = 'o', markerfacecolor = 'none' )
axs[2].set_ylabel('$T_{l}-T_{a}$ ($\degree$C)',size = 15)
axs[2].set_xlabel( 'Atmospheric Temperature ($\degree$C)', size = 15 )
axs[2].text( 0.01, 0.98, '(c)', transform = axs[2].transAxes, 
             ha = 'left', va = 'top', size = 12 )

# ci
axs[3].plot( Ta, ci_LTO, line_colors['LTO'],
             ls = line_style, marker = 'o', markerfacecolor = 'none' )
axs[3].plot( Ta, ci_LT, line_colors['LT'],
              ls = line_style, marker = 'o', markerfacecolor = 'none' )
axs[3].plot( Ta, ci_AT, line_colors['AT'],
              ls = line_style, marker = 'o', markerfacecolor = 'none' )
axs[3].set_ylabel('C$_i$ (Pa)',size = 15)
axs[3].set_xlabel( 'Atmospheric Temperature ($\degree$C)', size = 15 )
axs[3].text( 0.01, 0.98, '(d)', transform = axs[3].transAxes, 
             ha = 'left', va = 'top', size = 12 )

fig.savefig(Fig_path + 'Figure_S2.jpg', dpi = 300, bbox_inches = 'tight')

