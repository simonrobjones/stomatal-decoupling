# -*- coding: utf-8 -*-
"""
Functions required to produce the theoretical plots from Jones et al

@author: srgj201
"""

import numpy as np
from scipy.optimize import minimize_scalar

# =============================================================================
# Set up classes containing values for physical constants and model parameters
# =============================================================================

class physical_constants():
    def __init__( self ):        
        self.cp            = 1003.5    # Heat capacity of air                                            (J kg-1 K-1)
        self.Lc0           = 2.501e6   # Latent heat of condensation of water at 0degc                   (J kg-1)
        self.h_planck      = 6.626e-34 # Planck constant                                                 (m2 kg s-1)
        self.Na            = 6.022e23  # Avagadros constant                                              (mol-1)
        self.c_light       = 3.0e8     # Speed of light                                                  (m s-1)
        self.g             = 9.81      # Accelrarion due to gravity                                      (m s-2)
        self.rho_w         = 997.      # Density of water                                                (kg m-3)
        self.w             = 1.0e-6    # Converts Pa to MPa                                              (MPa Pa-1)
        self.lambda_par    = 550e-9    # Wave length of photosynthetically active radiation              (m)
        self.sigma_sb      = 5.67e-8   # Stefan-Boltzman constant                                        (W m-2 K-4)
        self.dTa_s         = -20.0     # The difference in the actual and apparent temperatue of the air (C)
    def rho_air( self, T, pa ):        # Density of air                                                  (kg m-3)
        Rs = 287.0500676               # Specific gas constant for dry air                               (J kg-1 K-1)
        return pa / ( ( T + 273.15) * Rs )
    
class namelist:
    def __init__( self ):
        # Parameters
        self.Pcrit          = -2.0     # Critical LWP                                              (MPa)
        self.rp             = 200.0    # Plant hydraulic resistance                                (mol-1 m2 s MPa)
        self.h              = 10.0     # Canopy height                                             (m)
        self.Tmin           = 0.0      # Minimum temperature parameter for photosynthesis          (deg C)
        self.Topt           = 35.0     # Optimum temperature parameter for photosynthesis          (deg C)
        self.Tmax           = 40.0     # Maximum temperature parameter for photosynthesis          (deg C)
        self.alpha          = 0.3      # The apparent quantum yield of electron transport          (mol electrons mol−1 photon)
        self.theta          = 0.9      # Non-rectangular hyperbola smoothing parameter             (-)
        self.vcmax25        = 39.5e-6  # Maximum rate of carboxylation at 25CC                     (µmol m2 s−1)
        self.jmax25         = 63.2e-6  # Light-saturated potential electron transport rate at 25C  (µmol m2 s−1)
        self.gs_co2_mol_min = 1.0e-10 

#%%
# =============================================================================
# The Leaf Temperature within Optimisation (LTO) approach    
# =============================================================================
def numerical_solve_LTO( nl, pc, Ta, ca, pa, oa, Is, ra, vpd, swp):
    # Calculate maximum possible stomatal conductance
    vpd_mol        = vpd / pa
    Ppd            = swp - nl.h * pc.g * pc.rho_w * pc.w
    gs_co2_mol_max = ( Ppd - nl.Pcrit ) / ( 1.6 * nl.rp * vpd_mol )
    
    f = lambda gs_co2_mol: -objective_function_LTO( gs_co2_mol, nl, pc, Ta, ca, pa, oa, Is, ra, vpd, swp)
    
    # Extract optimal gs to CO2 and calculate optimal leaf temperature, ci, and fluxes
    result = minimize_scalar( f, bounds = (0.0, gs_co2_mol_max), method = 'Bounded')
    gs_co2_mol_opt = result.x
    
    # Check to make sure the objective function is still positive at gs_opt. Otherwise we set gs to zero
    obj_opt = -f(gs_co2_mol_opt)
    if obj_opt <= 0:
        gs_co2_mol_opt = nl.gs_co2_mol_min
    
    gs_h2o_mol_opt = gs_co2_mol_opt * 1.6
    gs_h2o_m_s_opt = gs_h2o_mol_opt * 8.314462 * ( Ta + 273.15 ) / pa
    T_leaf_opt     = calc_Tleaf( pc, gs_h2o_m_s_opt, Ta, Is, pa, ra, vpd )
    A_opt          = calc_A_from_gs( pc, nl, T_leaf_opt, gs_co2_mol_opt, ca, pa, Is, oa)
    ci_opt         = ca - A_opt * pa / gs_co2_mol_opt
    return A_opt, gs_co2_mol_opt, ci_opt, T_leaf_opt
    
def objective_function_LTO( gs_co2_mol, nl, pc, Ta, ca, pa, oa, Is, ra, vpd, swp):
    """
    Calculate the objective function for simplified PGEN model with the leaf 
    energy balance integrated using the Leaf Temperature within Optimisation (LTO) 
    approach
    
    Inputs:
    gs_co2_mol = Stomatal conductance to CO2               (mol m-2 s-1)
    nl         = namelist                                  (class containing model parameters )
    pc         = physical constants                        (class containing values of physical constants) 
    Ta         = Air temperature                           (C)
    ca         = Partial pressure of CO2 in the atmosphere (Pa)
    pa         = Atmospheric air pressure                  (Pa)
    oa         = Partial pressure of O2 in the atmosphere  (Pa)
    Is         = Absorbed incoming short-wave radiation    (W m-2)
    ra         = Aerodynamic resistance to water vapour    (s m-1)
    vpd        = Vapour pressure deficit                   (Pa)  
    swp        = Soil water potential                      (MPa)
    
    Outputs:
    obj        = Objective function ( A * f_psi_leaf )    
    """
    
    # Calculate vpd in mol mol-1
    vpd_mol   = vpd / pa
    
    # Calculate pre-dawn leaf water potential
    Ppd   = swp - nl.h * pc.g * pc.rho_w * pc.w
    
    # Calculate stomatal conductance to H20 
    gs_h2o_mol = gs_co2_mol * 1.6                              # Stomatal conductance to H2O (mol m-2 s-1)
    gs_h2o_m_s = gs_h2o_mol * 8.314462 * ( Ta + 273.15 ) / pa  # Stomatal conductance to H2O (m s-1)
     
    # Calculate leaf temperature and fluxes from gs
    Tl              = calc_Tleaf( pc, gs_h2o_m_s, Ta, Is, pa, ra, vpd )      # N.B Leaf temperature is a function of gs WITHIN the objective function
    A               = calc_A_from_gs( pc, nl, Tl, gs_co2_mol, ca, pa, Is, oa)
    E               = 1.6 * vpd_mol * gs_h2o_mol
    lwp             = Ppd - E * nl.rp
    F               = calc_cost_from_lwp( nl, lwp )
    
    # Calculate objective function
    obj               = np.clip(A * F, 0, None)
    
    return obj
#%%
# =============================================================================
# The Leaf Temperature (LT) approach    
# =============================================================================
def numerical_solve_LT( nl, pc, Ta, ca, pa, oa, Is, ra, vpd, swp):
    # Calculate maximum possible stomatal conductance
    vpd_mol        = vpd / pa
    Ppd            = swp - nl.h * pc.g * pc.rho_w * pc.w
    gs_co2_mol_max = ( Ppd - nl.Pcrit ) / ( 1.6 * nl.rp * vpd_mol )
    
    Tl_i           = Ta
    convergence_Tl = False
    while not convergence_Tl:
        f = lambda gs_co2_mol: -objective_function_LT( gs_co2_mol, nl, pc, Ta, ca, pa, oa, Is, ra, vpd, swp, Tl_i)
    
        # Extract optimal gs to CO2 and calculate optimal leaf temperature
        result = minimize_scalar( f, bounds = (0.0, gs_co2_mol_max), method = 'Bounded')
        gs_co2_mol_opt = result.x
        
        # Check the optimal gs still gives a positive objective function (otherwise set gs to 0)
        obj_opt = -f(gs_co2_mol_opt)
        if obj_opt<=0:
            gs_co2_mol_opt = nl.gs_co2_mol_min
        
        gs_h2o_mol_opt = gs_co2_mol_opt * 1.6
        gs_h2o_m_s_opt = gs_h2o_mol_opt * 8.314462 * ( Ta + 273.15 ) / pa
        T_leaf_opt     = calc_Tleaf( pc, gs_h2o_m_s_opt, Ta, Is, pa, ra, vpd )
        
        # Check for convergence in Tleaf
        if np.abs(T_leaf_opt - Tl_i) < 0.001:
            convergence_Tl = True
        else:
            Tl_i = T_leaf_opt
            
    # Calculate optimal A, Tleaf, gs and ci
    gs_h2o_mol_opt = gs_co2_mol_opt * 1.6
    gs_h2o_m_s_opt = gs_h2o_mol_opt * 8.314462 * ( Ta + 273.15 ) / pa
    A_opt          = calc_A_from_gs( pc, nl, T_leaf_opt, gs_co2_mol_opt, ca, pa, Is, oa)
    ci_opt         = ca - A_opt * pa / gs_co2_mol_opt
    return A_opt, gs_co2_mol_opt, ci_opt, T_leaf_opt
    
def objective_function_LT( gs_co2_mol, nl, pc, Ta, ca, pa, oa, Is, ra, vpd, swp, Tl_i):
    """
    Calculate the objective function for simplified PGEN model with the leaf 
    energy balance integrated using the Leaf Temperature (LT) approach.
    
    Inputs:
    gs_co2_mol = Stomatal conductance to CO2               (mol m-2 s-1)
    nl         = namelist                                  (class containing model parameters )
    pc         = physical constants                        (class containing values of physical constants) 
    Ta         = Air temperature                           (C)
    ca         = Partial pressure of CO2 in the atmosphere (Pa)
    pa         = Atmospheric air pressure                  (Pa)
    oa         = Partial pressure of O2 in the atmosphere  (Pa)
    Is         = Absorbed incoming short-wave radiation    (W m-2)
    ra         = Aerodynamic resistance to water vapour    (s m-1)
    vpd        = Vapour pressure deficit                   (Pa)  
    swp        = Soil water potential                      (MPa)
    Tl_i       = Current leaf temperature                  (C)
    
    Outputs:
    obj        = Objective function ( A * f_psi_leaf )    
    """
    
    # Calculate vpd in mol mol-1
    vpd_mol   = vpd / pa
    
    # Calculate pre-dawn leaf water potential
    Ppd   = swp - nl.h * pc.g * pc.rho_w * pc.w
    
    # Calculate stomatal conductance to H20 
    gs_h2o_mol = gs_co2_mol * 1.6                              # Stomatal conductance to H2O (mol m-2 s-1)
     
    # Calculate leaf temperature and fluxes from gs
    A               = calc_A_from_gs( pc, nl, Tl_i, gs_co2_mol, ca, pa, Is, oa) # NB Leaf temperature is NOT calculated inside the objective function
    E               = 1.6 * vpd_mol * gs_h2o_mol
    lwp             = Ppd - E * nl.rp
    F               = calc_cost_from_lwp( nl, lwp )
    
    # Calculate objective function
    obj               = np.clip(A * F, 0, None)
    
    return obj
#%%
# =============================================================================
# The Air Temperature (AT) approach    
# =============================================================================
def numerical_solve_AT( nl, pc, Ta, ca, pa, oa, Is, ra, vpd, swp):
    # Calculate maximum possible stomatal conductance
    vpd_mol        = vpd / pa
    Ppd            = swp - nl.h * pc.g * pc.rho_w * pc.w
    gs_co2_mol_max = ( Ppd - nl.Pcrit ) / ( 1.6 * nl.rp * vpd_mol )
    
    # Create function to minimise (just -1 times the objective function)
    f = lambda gs_co2_mol: -objective_function_AT( gs_co2_mol, nl, pc, Ta, ca, pa, oa, Is, ra, vpd, swp)
    
    # Solve numerically using the minimize_scalar method from scipy
    result = minimize_scalar( f, bounds = (0.0, gs_co2_mol_max), method = 'Bounded')
    
    # Extract optimal gs to CO2 and calculate optimal leaf temperature, ci, and fluxes
    gs_co2_mol_opt = result.x
    
    # Check to make sure the objective function is still positive. Otherwise we set optimal stomatal conductance to zeor
    obj_opt = -f(gs_co2_mol_opt)
    if obj_opt<=0:
        gs_co2_mol_opt = nl.gs_co2_mol_min
    
    # Calculate optimal leaf temperature, photosynthesis and ci
    T_leaf_opt     = Ta
    A_opt          = calc_A_from_gs( pc, nl, T_leaf_opt, gs_co2_mol_opt, ca, pa, Is, oa)
    ci_opt         = ca - A_opt * pa / gs_co2_mol_opt
    
    return A_opt, gs_co2_mol_opt, ci_opt, T_leaf_opt
    
def objective_function_AT( gs_co2_mol, nl, pc, Ta, ca, pa, oa, Is, ra, vpd, swp):
    """
    Calculate the objective function for simplified PGEN model without a leaf 
    energy balance integrated. I.e. using the Air Temperature (AT) approach
    
    Inputs:
    gs_co2_mol = Stomatal conductance to CO2               (mol m-2 s-1)
    nl         = namelist                                  (class containing model parameters )
    pc         = physical constants                        (class containing values of physical constants) 
    Ta         = Air temperature                           (C)
    ca         = Partial pressure of CO2 in the atmosphere (Pa)
    pa         = Atmospheric air pressure                  (Pa)
    oa         = Partial pressure of O2 in the atmosphere  (Pa)
    Is         = Absorbed incoming short-wave radiation    (W m-2)
    ra         = Aerodynamic resistance to water vapour    (s m-1)
    vpd        = Vapour pressure deficit                   (Pa)  
    swp        = Soil water potential                      (MPa)
    
    Outputs:
    obj        = Objective function ( A * f_psi_leaf )    
    """
    
    # Calculate vpd in mol mol-1
    vpd_mol   = vpd / pa
    
    # Calculate pre-dawn leaf water potential
    Ppd   = swp - nl.h * pc.g * pc.rho_w * pc.w
    
    # Calculate stomatal conductance to H20 
    gs_h2o_mol = gs_co2_mol * 1.6                              # Stomatal conductance to H2O (mol m-2 s-1)
     
    # Calculate leaf temperature and fluxes from gs
    Tl              = Ta                                                   # N.B. Leaf temperature is always equal to air temperature
    A               = calc_A_from_gs( pc, nl, Tl, gs_co2_mol, ca, pa, Is, oa)
    E               = 1.6 * vpd_mol * gs_h2o_mol
    lwp             = Ppd - E * nl.rp
    F               = calc_cost_from_lwp( nl, lwp )
    
    # Calculate objective function
    obj               = np.clip(A * F, 0, None)
    
    return obj

#%%
# =============================================================================
# Simplified PGEN functions
# =============================================================================

def calc_Tleaf( pc, gs_h2o_m_s, Ta, Is, pa, ra, D ):
    """
    Calculate leaf temperature following Jones (2013) Plants and Microclimate: A Quantitative Approach to Environmental Plant Physiology
    Inputs
    pc         = physical constants                     (class containing values of physical constants) 
    gs_h2o_m_s = Stomatal conductance to H2O            (m s-1)
    Ta         = Air temperature                        (C)
    Is         = Absorbed short-wave radiation          (W m-2)
    pa         = Atmospheric air pressure               (Pa)
    ra         = Aerodynamic resistance to water vapour (s m-1)
    D          = Vapour pressure deficit                (Pa)
    """
    # Calculate the derivative of esat with respect to T at Ta
    desat_dT = calc_desat_dT_from_T( Ta ) 
    
    # Calculate the parallel resistance to heat loss from the leaf by convection and radiation
    gR       = 4 * pc.sigma_sb * ( Ta + 273.15 ) ** 3 / ( pc.rho_air( Ta, pa ) * pc.cp )
    rHR      = 1.0 / ( 1.0 / ( 1.25 * ra ) + gR )
    
    # Calculate isothermal raditation
    Ts       = Ta + pc.dTa_s                               # Apparent radiative temperature of the atmpshere (C)
    Tb       = Ta                                          # Background temperature                          (C)
    Rni      = ( Is + pc.sigma_sb * ( Ts + 273.15 ) ** 4 + # Isothermal radiation                            (W m-2)
                      pc.sigma_sb * ( Tb + 273.15 ) ** 4 - 
                  2 * pc.sigma_sb * ( Ta + 273.15 ) ** 4
                ) 
    # Calculate the psychometer "constant" (gamma)
    gamma    = pa * pc.cp / ( 0.622 * pc.Lc0 )
    
    # Finally calculate leaf temperature
    num1     = Rni * ( ra * gs_h2o_m_s + 1 ) * rHR * gamma / ( pc.rho_air(Ta, pa) * pc.cp )
    num2     = - rHR * D * gs_h2o_m_s
    denom    = gamma * ( ra * gs_h2o_m_s + 1 ) + desat_dT * rHR * gs_h2o_m_s
    return Ta + ( num1 + num2 ) / denom

def calc_esat_from_T(Ta):
    """
    Calculate the vapor pressure at saturation (esat; Pa) as a 
    function of atmospheric temperature, Ta (C) using the 
    August-Roche-Magnus formula
    """
    return 610.94 * np.exp( 17.625 * Ta / ( Ta + 243.04 ) )

def calc_desat_dT_from_T(Ta):
    """
    Calculate the gradient of vapor pressure at saturation (esat; Pa) 
    with respect to atmospheric temperature as a 
    function of atmospheric temperature, Ta (C) using the 
    first derivative of the August-Roche-Magnus formula.
    """
    return 610.94 * 17.625 * 243.04 * np.exp( 17.625 * Ta / ( Ta + 243.04 ) ) / ( Ta + 243.04 ) ** 2

def calc_A_from_gs( pc, nl, Tleaf, gs_co2_mol, ca, pa, Is, Oa):
    """
    Calculate photosynthesis from stomatal conductance. The equations for each
    limiting rate (Ac and Aj) by combining the equation for Ax in terms of ci
    with the equilibrium diffusion equation A = gs(ca-ci)/pa to elimiate ci and
    rearrange for gs
    
    Inputs:
    pc         = physical constants                        (class containing values of physical constants) 
    nl         = namelist                                  (class containing model parameters )
    Tleaf      = Leaf temperature                          (C)
    gs_co2_mol = Stomatal conductance to CO2               (mol m-2 s-1)
    ca         = Partial pressure of CO2 in the atmosphere (Pa)
    Is         = Absorbed short-wave radiation             (mol quanta m-2 s-1)
    Oa         = Partial pressure of O2 in the atmosphere  (Pa)
    """
    # Calculate photosynthetically active radiation (Ipar, mol quanta m-2 s-1)
    fpar    = 0.5
    Ephoton = pc.h_planck * pc.c_light / pc.lambda_par
    Ipar    = fpar * Is / ( Ephoton * pc.Na )
    
    # Calculate the CO2 compensation point in the absence of dark respiration,
    # and Michaelis-menten parameters for carboxylation
    Kc, Ko, gamma_star = calc_photosynthetic_params( Tleaf, Oa )
    
    # Calculate the temperature sensitivities of Jmax and Vcmax
    fT         = calc_fT(nl, Tleaf )
    Jmax       = nl.jmax25 * fT
    Vcmax      = nl.vcmax25 * fT
    
    # Calculate dark respiration
    rd         = 0.1 * Vcmax
    
    # Ribulose-bisphosphate regeneration-limited photosynthesis
    J          = calc_J( nl, Jmax, Ipar )
    a          = 1.0
    b          = - ( ( J / 4) + ( ca + 2 * gamma_star ) * gs_co2_mol / pa - rd )
    c          = ( J / 4 ) * ( ca - gamma_star ) * gs_co2_mol / pa - ( ca + 2 * gamma_star ) * rd * gs_co2_mol / pa
    Wl         = ( -b - ( b**2.0 - 4.0 * a * c )**0.5 ) / ( 2.0 * a )
    
    # Carboxylation-limited photosynthesis
    K          = Kc * ( 1 + Oa / Ko )
    a          = 1.0
    b          = - ( Vcmax + ( ca + K ) * gs_co2_mol / pa - rd )
    c          = Vcmax * ( ca - gamma_star ) * gs_co2_mol / pa - ( ca + K ) * rd * gs_co2_mol / pa
    Wc          = ( -b - ( b**2.0 - 4.0 * a * c )**0.5 ) / ( 2.0 * a )
    
    # A = np.clip( Wl, Wc, None )
    a = 0.9
    b = Wc + Wl
    c = Wc * Wl
    A = ( b - ( b**2 - 4 * a * c ) ) / ( 2 * a )
    return np.clip(A,-rd,None)

def calc_photosynthetic_params( T_leaf, Oa ):
    """
    Calculate the CO2 compensation point in the absence of dark respiration,
    and Michaelis-menten parameters for carboxylation
    
    Inputs:
    T_leaf = Leaf temperature                   (C)
    Oa     = Atmospheric partial pressure of O2 (Pa)
    
    Returns
    Kc, Ko     = Michaelis-Menten parameters for carboxylation         (Pa)
    gamma_star = CO2 compensation point in the abscence of respiration (Pa)
    """
    # Calculate the specificity of Rubisco
    tau = 2710 * 0.57 ** ( 0.1 * ( T_leaf - 25.0 ))
    # Calculate CO2 compensation point
    gamma_star = Oa / ( 2 * tau )
    # Calculate Michaelis-Menten constants
    Ko         = 1e3 * np.exp( 12.3772 - 23.72 / ( 0.008314 * ( 273.15 + T_leaf ) ) )
    Kc         = np.exp( 35.9774 - 80.99 / ( 0.008314 * ( 273.15 + T_leaf ) ) )
    return Kc, Ko, gamma_star

def calc_fT( nl, T_leaf ):
    """
    Calculate the temperature sensitivity of Jmax and Vcmax
    
    Inputs:
    nl     = namelist         (class containing model parameters )
    T_leaf = Leaf temperature (C)
    
    Returns:
    f      = Temperature function for Jmax and Vcmax  
    """
    T_leaf = np.clip( T_leaf, nl.Tmin, nl.Tmax )
    f = ( ( nl.Tmax - T_leaf ) / ( nl.Tmax - nl.Topt ) * 
         ( ( T_leaf - nl.Tmin ) / ( nl.Topt - nl.Tmin ) ) ** ( ( nl.Topt - nl.Tmin ) / ( nl.Tmax - nl.Topt ) ) )
    
    # Prevent f from going negative
    f = np.clip( f, 0, None )
    return f

def calc_J( nl, Jmax, Ipar ):
    """
    Calculate the potential rate of electron transport
    
    Inputs:
    nl   = namelist                                          (class containing model parameters )
    Jmax = Light-saturated potential electron transport rate (mol m-2 s-1)
    Ipar = Absorbed photosynthetically active radiation      (mol quanta m-2 s-1)

    Returns:
    r2   = The correct root of the quadratic for J
    """
    # J is the solution to the quadratic: theta * J^2 - (alpha*Ipar + Jmax) + alpha*Ipar*Jmax = 0
    a = nl.theta
    b = -( nl.alpha * Ipar + Jmax )
    c = nl.alpha * Ipar * Jmax

    # r1 = ( -b + ( b**2 - 4 * a * c) ** 0.5 ) / ( 2 * a ) # The unused root of the quadratic
    r2 = ( -b - ( b**2 - 4 * a * c) ** 0.5 ) / ( 2 * a )
    return r2

def calc_cost_from_lwp( nl, lwp ):
    """
    Calculate the water cost function (f_psi_leaf) from leaf water potential
    
    Inputs:
    nl  = namelist             (class containing model parameters )
    lwp = Leaf water potential (MPa)
    
    Returns:
    f_psi_leaf (unitless)
        
    """
    return np.clip( 1 - lwp / nl.Pcrit, 0, None )



