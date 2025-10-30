# -*- coding: utf-8 -*-
"""
Code to produce Figure 3 of Jones et al
@author: srgj201
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PGEN_functions import numerical_solve_LTO, numerical_solve_LT, numerical_solve_AT
from PGEN_functions import namelist, physical_constants
from PGEN_functions import calc_esat_from_T
from PGEN_functions import calc_Tleaf

# =============================================================================
# Set up environmental conditions (excluding soil water potential)
# =============================================================================
nl   = namelist()                 # Class containing all model parameters
pc   = physical_constants()       # Class containing all physical constants
N_T  = 1000                       # Number of temperature points (needs to be high so that the derivatives 
                                  #                               can be numerically determined accurately)
ca   = 40.0                       # Atmospheric CO2 partial pressure (Pa)
swp  = -0.1                       # Soil water potential             (MPa)
pa   = 101325.0                   # Atmospheric pressure             (Pa)
oa   = 20900.0                    # Atmospheric O2 partial pressure  (Pa)
Is   = 500                        # Absorbed shortwave radiation     (Wm-2)
ra   = 10                         # Aerodymaic resistance            (s m-1)
ea   = 1000                       # Atmospheric vapour pressure      (Pa)
Ta   = np.linspace(25.0,41.0,N_T) # Atmospheric Temperature          (C)

# The model takes VPD as an input so calculate the VPD from Ta and ea
esat     = calc_esat_from_T( Ta ) # Saturated vapour pressure      (Pa)
vpd      = esat - ea              # Vapour pressure deficit        (Pa)

# =============================================================================
# Set up soil water potential values
# We will iterate over a number (N_swp) of soil water potential values
# =============================================================================
N_swp        = 20                                        # Number of soil water potential values to iterate over

swp_crit     = nl.Pcrit + nl.h * pc.g * pc.rho_w * pc.w  # We can calculate the soil water potential at which 
                                                         # the plant is completely water stressed as the SWP that
                                                         # would result in a pre-dawn water potential equal to the
                                                         # critical leaf water potential (Pcrit). We will deal with
                                                         # the response separately here

swp_vals     = np.linspace(-0., -1.7, N_swp - 1)         # Set up the array of soil water potential values

# =============================================================================
# Set up arrays to store the critical temperatures
# =============================================================================
Tcrit_vals = np.zeros(N_swp) # The critical air temperatures at which stomatal conductance decouples from photosynthesis
Tshut_vals = np.zeros(N_swp) # The critical air temperatures at which stomata shut

# =============================================================================
# Set up arrays to store optimal photosynthesis (A), stomatal conductance (gs)
# internal leaf CO2 (ci) and leaf temperature (Tl)
# =============================================================================
A  = np.zeros((N_swp-1,N_T))
gs = np.zeros((N_swp-1,N_T)) 
ci = np.zeros((N_swp-1,N_T))
Tl = np.zeros((N_swp-1,N_T))

# =============================================================================
# Iterate through soil water potential values and calculate critical temperatures
# =============================================================================
for i,swp_val in enumerate(swp_vals):
    print(i)    
    # For each Ta and VPD calculate the optimum and store results in arrays
    for j in range(N_T):
        A[i,j], gs[i,j], ci[i,j], Tl[i,j] = numerical_solve_LTO( nl, pc, Ta[j], ca, pa, oa, Is, ra, vpd[j], swp_val)
    
    # Calculate the gradients of A and gs with respect to Ta numerically using np.gradient
    dgs_dTa       = np.gradient(gs[i,:], Ta)
    dA_dTa        = np.gradient(A[i,:], Ta)
    
    # Find where gs and A are decoupled by looking for where their gradients are opposite
    arr           = ( dgs_dTa>0 ) * ( dA_dTa < 0 )
    
    # The critical decoupling air temperature is the minimum air temperature at which the gradients are opposite
    idx_crit      = np.where(np.gradient(1*arr)>0)[0][-1]
    Tcrit_vals[i] = Ta[idx_crit]
    
    # Because of the way we have set up the simulation we can find the critical temperature at which stomata close
    # as the first Ta value at which stomatal conductance is its smallest value.
    idx_shut        = np.argmin(gs[i,:])
    Tshut_vals[i]  = Ta[idx_shut]
    
# =============================================================================
# Deal with complete water stress. I.e. swp = swp_crit
# =============================================================================
# Append the critical soil water potential to array of values
swp_vals = np.append(swp_vals,swp_crit)

# The stomata close when the leaf temperature exceeds the maximum temperature
# for photosynthesis (Tmax). As the soil moisture declines, stomatal conductance
# decreases and hence when the plant is fully water stressed we can calculate
# the critical air temperature at wich stomata close by calculating what air
# temperature causes a leaf temperature equal to Tmax with stomata closed
vpd_off        = calc_esat_from_T( nl.Tmax ) - ea
dT0            = calc_Tleaf( pc, gs_h2o_m_s = 0, Ta = nl.Tmax, Is = Is, pa = pa, ra = ra, D = vpd_off ) - nl.Tmax
Tshut_vals[-1]  = nl.Tmax - dT0

# Since stomata are closed for all temperatures the critical decoupling air temperature is equal to to the shutting temperature
# (Hence gs does not decouple from A since the range of temperatures between critical and shutting temperatures is 0)
Tcrit_vals[-1] = nl.Tmax - dT0

# =============================================================================
# Set up figure
# There are three states:
#    1. stomata are closed
#    2. gs is decoupled from A
#    3. gs is coupled to A
# =============================================================================
# Create arrays for entire span of T and swp ranges to be plotted
Tplot   = np.concatenate( (Tcrit_vals, np.flip(Tshut_vals) ) )
swpplot = np.concatenate( ( swp_vals, np.flip(swp_vals) ) )

plt.figure()
# Plot the boundaries between the three states
plt.plot( Tcrit_vals, swp_vals,color = 'black', ls = '--')                                      # Boundary between decoupled and coupled
plt.plot( Tshut_vals, swp_vals,color = 'black', ls = '--')                                      # Boundary between decoupled and closed
plt.plot( [ Tcrit_vals[0], Tcrit_vals[-1] ], [swp_crit, swp_crit], ls = '--', color = 'black')  # Boundary between coupled and closed

# Colour in areas
                                                                                                # Coupled (left white)
plt.fill_between( Tplot, swpplot, swp_vals[0], color = 'grey', alpha = 0.5, edgecolor = None )  # Decoupled
plt.fill_between( Tplot, -2.5, np.concatenate(( np.ones(N_swp) * swp_crit, np.flip(swp_vals))), # Closed
                 color = 'purple', alpha = 0.5, edgecolor = None)

# Add labels
plt.text(0.35,0.7,'Decoupling', ha = 'left', size = 12,transform = plt.gca().transAxes)
plt.text(0.15,0.4,'No decoupling', ha = 'center', size = 12,transform = plt.gca().transAxes)
plt.text(0.5,0.1,'Stomata closed', size = 12,ha = 'center',transform = plt.gca().transAxes)

# Figure formatting
plt.xlim(Tplot.min(), Tplot.max())
plt.ylim(-2.5,swp_vals[0])
plt.xlabel('Air temperature ($\degree$C)',size = 12)
plt.ylabel('Soil water potential (MPa)',size = 12)

plt.savefig('Figures/Figure_3.jpg',dpi = 300)