# -*- coding: utf-8 -*-
"""
Code to produce Figure 4 of Jones et al
@author: srgj201
"""

import matplotlib.pyplot as plt
import numpy as np
from PGEN_functions_complete import numerical_solve_LTO
from PGEN_functions_complete import namelist, physical_constants
from PGEN_functions_complete import calc_esat_from_T
Fig_path = 'Figures/'

# =============================================================================
# Set up environmental conditions (excluding soil water potential)
# =============================================================================
nl   = namelist()                 # Class containing all model parameters
pc   = physical_constants()       # Class containing all physical constants
N_T  = 500                        # Number of temperature points (needs to be high so that the derivatives 
                                  # can be numerically determined accurately)
ca   = 40.0                       # Atmospheric CO2 partial pressure (Pa)
pa   = 101325.0                   # Atmospheric pressure             (Pa)
oa   = 20900.0                    # Atmospheric O2 partial pressure  (Pa)
Is   = 500                        # Absorbed shortwave radiation     (Wm-2)
ra   = 10                         # Aerodymaic resistance            (s m-1)
ea   = 500                       # Atmospheric vapour pressure       (Pa)
Ta   = np.linspace(25.0,50.0,N_T) # Atmospheric Temperature          (C)

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

swp_vals     = np.linspace(-0., -1.999, N_swp )          # Set up the array of soil water potential values

# =============================================================================
# Set up arrays to store the critical temperatures
# =============================================================================
Tcrit_vals    = np.zeros(N_swp)   # The critical air temperatures at which stomatal conductance decouples from photosynthesis
Tshut_vals = np.zeros(N_swp)      # The critical air temperatures at which stomata shut

# =============================================================================
# Set up arrays to store optimal photosynthesis (A), stomatal conductance (gs)
# internal leaf CO2 (ci) and leaf temperature (Tl)
# =============================================================================
A  = np.zeros((N_swp,N_T))
gs = np.zeros((N_swp,N_T)) 
ci = np.zeros((N_swp,N_T))
Tl = np.zeros((N_swp,N_T))

# =============================================================================
# Iterate through soil water potential values and calculate critical temperatures
# =============================================================================
plt.figure()
for i,swp_val in enumerate(swp_vals):
    print(i)    
    # For each Ta and VPD calculate the optimum and store results in arrays
    
    A[i,:], gs[i,:], ci[i,:], Tl[i,:] = numerical_solve_LTO( nl, pc, Ta, ca, pa, oa, Is, ra, vpd, swp_val)
    
    
    plt.plot( Ta, gs[i,:])
    
    # Calculate the gradients of A and gs with respect to Ta numerically using np.gradient
    dgs_dTa       = np.gradient(gs[i,:], Ta)
    dA_dTa        = np.gradient(A[i,:], Ta)
    
    # Find where gs and A are decoupled by looking for where their gradients are opposite
    arr           = ( dgs_dTa>0 ) * ( dA_dTa < 0 )
    
    
    # The critical decoupling air temperature is the minimum air temperature at which the gradients are opposite
    idx_crit      = np.where(np.gradient(1*arr)>0)[0][0]
    Tcrit_vals[i] = Ta[idx_crit]
    
    # We can find the critical temperature at which stomata close
    # as the first Ta value at which stomatal conductance is its smallest value.
    idx_shut          = np.where(np.gradient(1*arr)<0)[0][0]
    Tshut_vals[i]     = Ta[idx_shut]


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

fig = plt.figure()
# Plot the boundaries between the three states
plt.plot( Tcrit_vals, swp_vals,color = 'black', ls = '--')                                            # Boundary between decoupled and coupled
plt.plot( Tshut_vals, swp_vals,color = 'black', ls = '--')                                         # Boundary between decoupled and closed
plt.plot( [ Tcrit_vals.min(), Tshut_vals[-1] ], [swp_crit, swp_crit], ls = '--', color = 'black')   # Boundary between coupled and closed

# Colour in areas
plt.fill_between( Tplot, swpplot, swp_vals[0], color = 'grey', alpha = 0.5, edgecolor = None )        # Decoupled
plt.fill_between( Tplot, -2.5, np.concatenate(( np.ones(N_swp) * swp_crit, np.flip(swp_vals))), # Closed
                 color = 'purple', alpha = 0.5, edgecolor = None)
# Coupled (left white)

# Add labels
plt.text(0.35,0.7,'Decoupling', ha = 'left', size = 12,transform = plt.gca().transAxes)
plt.text(0.145,0.3,'No decoupling', ha = 'center', size = 12,transform = plt.gca().transAxes)
plt.text(0.5,0.1,'Stomata closed', size = 12,ha = 'center',transform = plt.gca().transAxes)

# Figure formatting
plt.xlim(Tplot.min(), Tplot.max())
plt.ylim(-2.5,swp_vals[0])
plt.xlabel('Air temperature ($\degree$C)',size = 12)
plt.ylabel('Soil water potential (MPa)',size = 12)

fig.savefig(Fig_path + 'Figure_4.jpg', dpi = 300, bbox_inches = 'tight')
