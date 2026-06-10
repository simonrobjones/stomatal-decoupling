# -*- coding: utf-8 -*-
"""
@author: srgj201
"""

import numpy as np
from scipy.optimize import minimize_scalar
from itertools import product
import pandas as pd
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
        self.Rgas          = 8.314     # Molar gas constant                                              (J K-1 mol-1)
    def rho_air( self, T, pa ):        # Density of air                                                  (kg m-3)
        Rs = 287.0500676               # Specific gas constant for dry air                               (J kg-1 K-1)
        return pa / ( ( T + 273.15) * Rs )    
    

class namelist:
    def __init__( self ):
        # Parameters
        self.Pcrit          = -2.0     # Critical LWP                                              (MPa)
        self.rp             = 200.0    # Plant hydraulic resistance                                (mol-1 m2 s MPa)
        self.h              = 0.0      # Canopy height                                             (m)
        self.Ea             = 40.0e3   # Minimum temperature parameter for photosynthesis          (deg C)
        self.Topt           = 35.0     # Optimum temperature parameter for photosynthesis          (deg C)
        self.eta            = 5.0      # Maximum temperature parameter for photosynthesis          (deg C)
        self.alpha          = 0.3      # The apparent quantum yield of electron transport          (mol electron mol−1 photon)
        self.theta          = 0.9      # Non-rectangular hyperbola smoothing parameter             (dimensionless)
        self.vcmax25        = 39.5e-6  # Maximum rate of carboxylation at 25CC                     (µmol m2 s−1)
        self.jmax25         = 63.2e-6  # Light-saturated potential electron transport rate at 25C  (µmol m2 s−1)
        self.fd             = 0.01     # Dark respiration coefficient                              (dimensionless)
        self.q10            = 2.0      # Q10 value for dark respiration  
        self.omega          = 0.17
        self.alpha_C4       = 0.06     # The apparent quantum yield of electron transport for C4   (mol electrons mol−1 photon)



# =============================================================================
# PGEN Functions        
# =============================================================================
def FEB( pc, gs_h2o_m_s, Ta, Iabs, pa, ra, D, Tl ):
    """
    The energy balance equation equated to zero. Required for Newton-Raphson stepping of Tleaf in LT method
    Inputs
    pc         = physical constants                     (class containing values of physical constants) 
    gs_h2o_m_s = Stomatal conductance to H2O            (m s-1)
    Ta         = Air temperature                        (C)
    Iabs       = Absorbed short-wave radiation          (W m-2)
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
    Ts       = Ta + pc.dTa_s                                 # Apparent radiative temperature of the atmpshere (C)
    Tb       = Ta                                            # Background temperature                          (C)
    Rni      = ( Iabs + pc.sigma_sb * ( Ts + 273.15 ) ** 4 + # Isothermal radiation                            (W m-2)
                      pc.sigma_sb * ( Tb + 273.15 ) ** 4 - 
                  2 * pc.sigma_sb * ( Ta + 273.15 ) ** 4
                ) 
    # Calculate the psychometer "constant" (gamma)
    gamma    = pa * pc.cp / ( 0.622 * pc.Lc0 )
    
    # Finally calculate leaf temperature
    num1     = Rni * ( ra * gs_h2o_m_s + 1 ) * rHR * gamma / ( pc.rho_air(Ta, pa) * pc.cp )
    num2     = - rHR * D * gs_h2o_m_s
    denom    = gamma * ( ra * gs_h2o_m_s + 1 ) + desat_dT * rHR * gs_h2o_m_s
    return Ta - Tl + ( num1 + num2 ) / denom

def d_FEB_dT( pc, gs_h2o_m_s, Ta, Iabs, pa, ra, D, Tl ):
    """Derivative of the energy balance equation with respect to Tleaf. Required for Newton-Raphson stepping of Tleaf in LT method
       For the simplified equation from Jones (2013) this is just equal to -1 but inputs left in case full EB equation used later.
    """
    return -1

def calc_Tleaf( pc, gs_h2o_m_s, Ta, Iabs, pa, ra, D ):
    """
    Calculate leaf temperature following Jones (2013) Plants and Microclimate: A Quantitative Approach to Environmental Plant Physiology
    Inputs
    pc         = physical constants                     (class containing values of physical constants) 
    gs_h2o_m_s = Stomatal conductance to H2O            (m s-1)
    Ta         = Air temperature                        (C)
    Iabs       = Absorbed short-wave radiation          (W m-2)
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
    Rni      = ( Iabs + pc.sigma_sb * ( Ts + 273.15 ) ** 4 + # Isothermal radiation                            (W m-2)
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

def calc_A_from_gs( pc, nl, Tleaf, gs_co2_mol, ca, pa, Ipar, Oa, Ea, eta, Topt, vcmax25, jmax25, fd = None, q10 = None, rd_func = None, C4 = False):
    if C4:
        A = calc_A_from_gs_C4(  pc, nl, Tleaf, gs_co2_mol, ca, pa, Ipar, Ea, eta, Topt, vcmax25, fd = fd, q10 = q10, rd_func = rd_func )
    else:
        A = calc_A_from_gs_C3( pc, nl, Tleaf, gs_co2_mol, ca, pa, Ipar, Oa, Ea, eta, Topt, vcmax25, jmax25, fd = fd, q10 = q10, rd_func = rd_func )
    return A

def calc_A_from_gs_C3( pc, nl, Tleaf, gs_co2_mol, ca, pa, Ipar, Oa, Ea, eta, Topt, vcmax25, jmax25, fd = None, q10 = None, rd_func = None ):
    """
    Calculate photosynthesis from stomatal conductance. The equations for each
    limiting rate (Ac and Aj) by combining the equation for Ax in terms of ci
    with the equilibrium diffusion equation A = g(ca-ci)/pa to elimiate ci and
    rearrange for g

    """
    # Calculate the CO2 compensation point in the absence of dark respiration,
    # and Michaelis-menten parameters for carboxylation
    Kc, Ko, gamma_star = calc_photosynthetic_params( Tleaf, Oa )
    
    # Calculate the temperature sensitivities of Jmax and Vcmax
    fT         = calc_fT(pc, Tleaf, Ea, eta, Topt )
    Jmax       = jmax25 * fT
    Vcmax      = vcmax25 * fT
    
    # Calculate dark respiration
    if rd_func is None:
        if any(x is None for x in [fd, q10]):
            raise Exception("If rd_func is not supplied then fd and q10 must be provided.")
        else:
            rd25         = fd * vcmax25
            rd           = rd25 * q10 ** ( 0.1 * ( Tleaf - 25.0 ) )
    else:
        rd           = rd_func( Tleaf )
        
    
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
     
    Wc         = ( -b - ( b**2.0 - 4.0 * a * c )**0.5 ) / ( 2.0 * a )
    
    # A = np.clip(Wl,None,Wc)
    a = 0.9
    b = Wc + Wl
    c = Wc * Wl
    A = ( b - ( b**2 - 4 * a * c )**0.5 ) / ( 2.0 * a )
    return np.clip(A,-rd,None)

def calc_A_from_gs_C4( pc, nl, Tleaf, gs_co2_mol, ca, pa, Ipar, Ea, eta, Topt, vcmax25, fd = None, q10 = None, rd_func = None ):

    
    # Calculate the temperature sensitivities of Jmax and Vcmax
    fT         = calc_fT( pc, Tleaf, Ea, eta, Topt )
    Vcmax      = vcmax25 * fT
    
    # Calculate dark respiration
    if rd_func is None:
        if any(x is None for x in [fd, q10]):
            raise Exception("If rd-func is not supplied then fd and q10 must be provided.")
        else:
            rd25         = fd * vcmax25
            rd           = rd25 * q10 ** ( 0.1 * ( Tleaf - 25.0 ) )
    else:
        rd           = rd_func( Tleaf )
    
    # Calculate rates
    Wc = Vcmax
    Wl = nl.alpha_C4 * ( 1 - nl.omega ) * Ipar
    We = gs_co2_mol * ( 2e4 * Vcmax * ca - rd * pa ) / ( pa * ( gs_co2_mol + 2e4 * Vcmax ) ) + rd
    
    # Calculate limitation
    Wp = np.clip( Wc, None, Wl )
    W  = np.clip( Wp, None, We )
        
    # Calculate photosynthesis 
    A = (W - rd)
 
    return A



def calc_fT( pc, T_leaf_C, Ea, eta, Topt ):
    """
    Calculate the temperature sensitivity of Jmax and Vcmax
    
    Inputs:
    nl     = namelist         (class containing model parameters )
    T_leaf = Leaf temperature (C)
    
    Returns:
    f      = Temperature function for Jmax and Vcmax  
    """

    Tk    = T_leaf_C + 273.15
    Toptk = Topt + 273.15

    fA    = np.exp( Ea * ( Tk - Toptk ) / ( Tk * Toptk * pc.Rgas ) )
    GA    = eta * fA / ( ( eta - 1) + fA ** eta )

    return GA

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

    r = ( -b - ( b**2 - 4 * a * c) ** 0.5 ) / ( 2 * a )
    return r

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

def calc_cost_from_lwp_sigmoid( nl, lwp ):
    """
    Calculate the water cost function (f_psi_leaf) from leaf water potential
    
    Inputs:
    nl  = namelist             (class containing model parameters )
    lwp = Leaf water potential (MPa)
    
    Returns:
    f_psi_leaf (unitless)
        
    """
    return 1.0 / ( 1.0 + ( lwp / -2.0 )**5 )

def calc_cost_from_gcrit( gs_co2_mol, gcrit ):
    return np.clip(1 - gs_co2_mol / gcrit,0,None)

def normalize_inputs(*args):
    # Convert everything to arrays
    arrays = [
        np.atleast_1d(x) if not np.isscalar(x) else np.array([x])
        for x in args
    ]

    # Find target length
    lengths = [len(a) for a in arrays]
    max_len = max(lengths)

    # If all are length 1 → done
    if max_len == 1:
        return arrays

    # Broadcast scalars to match max length
    out = []
    for a in arrays:
        if len(a) == 1:
            out.append(np.repeat(a, max_len))
        elif len(a) == max_len:
            out.append(a)
        else:
            raise ValueError("Input lengths are incompatible")
    
    return out

# =============================================================================
# The Leaf Temperature within Optimisation (LTO) approach    
# =============================================================================
def numerical_solve_LTO( nl, pc, Ta, ca, pa, oa, Iabs, ra, vpd, swp = None, 
                         Ipar = None, vpd_air = None, beta = None, rb_co2_mol = None, 
                         rd_func = None, C4 = False, use_sigmoid = False ):
    
    Ta, ca, pa, oa, Iabs, Ipar, ra, rb_co2_mol, vpd, vpd_air, swp, beta = normalize_inputs( Ta, ca, pa, oa, Iabs, Ipar, ra, rb_co2_mol, vpd, vpd_air, swp, beta )
    
    if all(v is None for v in vpd_air):
        vpd_air = vpd
    
    if all(v is None for v in beta):
        if all(v is None for v in swp):
            raise Exception("Must supply either soil water potential or beta parameter")
        Ppd  = swp - nl.h * pc.g * pc.rho_w * pc.w
        beta = ( Ppd - nl.Pcrit ) / ( nl.rp )
    
    if all(v is None for v in rb_co2_mol):
        gb_h2o_m_s = 1.0 / ra
        gb_h2o_mol = gb_h2o_m_s * pa / ( pc.Rgas * ( Ta + 273.15 ) )
        gb_co2_mol = gb_h2o_mol / 1.6
        rb_co2_mol = 1.0 / gb_co2_mol
        
    if all(v is None for v in Ipar):
        # Calculate photosynthetically active radiation (Ipar, mol quanta m-2 s-1)
        fpar    = 0.5
        Ephoton = pc.h_planck * pc.c_light / pc.lambda_par
        Ipar    = fpar * Iabs / ( Ephoton * pc.Na )
        
    if use_sigmoid and all(v is None for v in swp):
        raise Exception("Soil water potential must be supplied if using sigmoidal water potential")
        
    # Calculate maximum stomatal conductance as that which results in the cost function equal to zero
    vpd_mol        = vpd / pa
    gs_co2_mol_max = beta / vpd_mol
    if use_sigmoid:
        gs_co2_mol_max = np.ones(len(vpd))
    
    T_leaf_opt     = np.zeros(len(Ta))
    A_opt          = np.zeros(len(Ta))
    ci_opt         = np.zeros(len(Ta))
    gs_co2_mol_opt = np.zeros(len(Ta))
    
    for i in range((len(Ta))):
        f = lambda gs_co2_mol: -objective_function_LTO( gs_co2_mol, nl, pc, Ta[i], 
                                                        ca[i], pa[i], oa[i], Iabs[i], 
                                                        Ipar[i], ra[i], vpd[i], vpd_air[i], 
                                                        rb_co2_mol[i], gs_co2_mol_max[i], 
                                                        rd_func, C4, use_sigmoid, swp[i] )
        # Calculate optimal stomatal conductance to CO2 (mol m-2 s-1)
        result = minimize_scalar( f, bounds = (0.0, gs_co2_mol_max[i] ), method = 'Bounded')
        gs_co2_mol_opt[i] = result.x
                
        # If cost function at optimal stomatal conductance is zero then shut stomata
        # ( due to finite numerical precision in the optimiser, check to see if cost is small  )
        F_opt = calc_cost_from_gcrit( gs_co2_mol_opt[i], gs_co2_mol_max[i] )
        if (use_sigmoid == False) and (F_opt<=1e-4):
            gs_co2_mol_opt[i] = 1e-3
        

        # Calculate remaining gas exchange        
        gs_h2o_mol_opt = gs_co2_mol_opt[i] * 1.6
        gs_h2o_m_s_opt = gs_h2o_mol_opt * 8.314462 * ( Ta[i] + 273.15 ) / pa[i]
        gt_co2_mol     = 1.0 / ( 1.0 / gs_co2_mol_opt[i] + rb_co2_mol[i] )
        T_leaf_opt[i]  = calc_Tleaf( pc, gs_h2o_m_s_opt, Ta[i], Iabs[i], pa[i], ra[i], vpd_air[i] )
        A_opt[i]       = calc_A_from_gs( pc, nl, T_leaf_opt[i], gt_co2_mol, ca[i], pa[i], Ipar[i], oa[i], 
                                         nl.Ea, nl.eta, nl.Topt, nl.vcmax25, nl.jmax25, nl.fd, nl.q10, C4 = C4)
        ci_opt[i]      = np.clip(ca[i] - A_opt[i] * pa[i] / gt_co2_mol, 0, ca[i] )
    return A_opt, gs_co2_mol_opt, ci_opt, T_leaf_opt
    
def objective_function_LTO( gs_co2_mol, nl, pc, Ta, ca, pa, oa, Iabs, Ipar, ra, vpd, vpd_air, rb_co2_mol, gcrit, rd_func, C4, use_sigmoid, swp ):
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
    Iabs       = Absorbed incoming short-wave radiation    (W m-2)
    ra         = Aerodynamic resistance to water vapour    (s m-1)
    vpd        = Vapour pressure deficit                   (Pa)  
    swp        = Soil water potential                      (MPa)
    
    Outputs:
    obj        = Objective function ( A * f_psi_leaf )    
    """ 
    gc_t       = 1.0 / ( 1.0 / gs_co2_mol + rb_co2_mol ) 
    
    # Calculate stomatal conductance to H20 
    gs_h2o_mol = gs_co2_mol * 1.6                              # Stomatal conductance to H2O (mol m-2 s-1)
    gs_h2o_m_s = gs_h2o_mol * pc.Rgas * ( Ta + 273.15 ) / pa   # Stomatal conductance to H2O (m s-1)
     
    # Calculate leaf temperature and fluxes from gs
    Tl              = calc_Tleaf( pc, gs_h2o_m_s, Ta, Iabs, pa, ra, vpd_air )      # N.B Leaf temperature is a function of gs WITHIN the objective function
    A               = calc_A_from_gs( pc, nl, Tl, gc_t, ca, pa, Ipar, oa, 
                                      Ea = nl.Ea, eta = nl.eta, Topt = nl.Topt, 
                                      vcmax25 = nl.vcmax25, jmax25 = nl.jmax25, 
                                      fd = nl.fd, q10 = nl.q10, rd_func = rd_func,
                                      C4 = C4)
    if use_sigmoid:
        ppd             = swp - nl.h * pc.g * pc.rho_w * pc.w
        lwp             = ppd - 1.6 * nl.rp * vpd * gs_co2_mol / pa
        F               = calc_cost_from_lwp_sigmoid( nl, lwp )
    else:
        F               = calc_cost_from_gcrit( gs_co2_mol, gcrit )

    
    # Calculate objective function
    obj               = A * F
    
    return obj

# =============================================================================
# The Leaf Temperature (LT) approach    
# =============================================================================
def numerical_solve_LT( nl, pc, Ta, ca, pa, oa, Iabs, ra, vpd, swp = None, 
                         Ipar = None, vpd_air = None, beta = None, rb_co2_mol = None, rd_func = None, C4 = False ):
    
    Ta, ca, pa, oa, Iabs, Ipar, ra, rb_co2_mol, vpd, vpd_air, swp, beta = normalize_inputs( Ta, ca, pa, oa, Iabs, Ipar, ra, rb_co2_mol, vpd, vpd_air, swp, beta )
    
    if all(v is None for v in vpd_air):
        vpd_air = vpd
    
    if all(v is None for v in beta):
        if all(v is None for v in swp):
            raise Exception("Must supply either soil water potential or beta parameter")
        Ppd  = swp - nl.h * pc.g * pc.rho_w * pc.w
        beta = ( Ppd - nl.Pcrit ) / nl.rp 
    
    if all(v is None for v in rb_co2_mol):
        gb_h2o_m_s = 1.0 / ra
        gb_h2o_mol = gb_h2o_m_s * pa / ( pc.Rgas * ( Ta + 273.15 ) )
        gb_co2_mol = gb_h2o_mol / 1.6
        rb_co2_mol = 0.0 * 1.0 / gb_co2_mol
        
    if all(v is None for v in Ipar):
        # Calculate photosynthetically active radiation (Ipar, mol quanta m-2 s-1)
        fpar    = 0.5
        Ephoton = pc.h_planck * pc.c_light / pc.lambda_par
        Ipar    = fpar * Iabs / ( Ephoton * pc.Na )
        
    # Calculate maximum stomatal conductance as that which results in the cost function equal to zero
    vpd_mol        = vpd / pa
    gs_co2_mol_max = beta / vpd_mol
    
    T_leaf_opt     = np.zeros(len(Ta))
    A_opt          = np.zeros(len(Ta))
    ci_opt         = np.zeros(len(Ta))
    gs_co2_mol_opt = np.zeros(len(Ta))
    
    for i in range(len(Ta)):

        Tl_i           = Ta[i]
        convergence_Tl = False
        count = 0
        while not ( ( convergence_Tl ) or ( count > 500 ) ):
            f = lambda gs_co2_mol: -objective_function_LT( gs_co2_mol, nl, pc, Ta[i], ca[i], 
                                                                  pa[i], oa[i], Iabs[i], Ipar[i], ra[i], 
                                                                  vpd[i], vpd_air[i], rb_co2_mol[i], 
                                                                  gs_co2_mol_max[i], Tl_i, rd_func, C4 )
        
            # Extract optimal gs to CO2 and calculate optimal leaf temperature
            result = minimize_scalar( f, bounds = (0.0, gs_co2_mol_max[i]), method = 'Bounded')
            gs_co2_mol_opt[i] = result.x
            
            # If objective function at optimal stomatal conductance is negative then shut stomata
            J_opt = -f( gs_co2_mol_opt[i])
            if J_opt < 0:
                gs_co2_mol_opt[i] = 1e-3
            
            gs_h2o_mol_opt = gs_co2_mol_opt[i] * 1.6
            gs_h2o_m_s_opt = gs_h2o_mol_opt * pc.Rgas * ( Ta[i]+ 273.15 ) / pa[i]
            T_leaf_opt[i] = Tl_i - 1.0 * FEB( pc, gs_h2o_m_s_opt, Ta[i], Iabs[i], pa[i], ra[i], vpd_air[i], Tl_i ) / d_FEB_dT( pc, gs_h2o_m_s_opt, Ta[i], Iabs[i], pa[i], ra[i], vpd_air[i], Tl_i)
            # Check for convergence in Tleaf
            if np.abs(T_leaf_opt[i] - Tl_i) < 0.001:
                convergence_Tl = True
            else:
                Tl_i = T_leaf_opt[i]
                count+=1
        
        # Calculate optimal A, Tleaf, gs and ci
        # If cost function at optimal stomatal conductance is zero then shut stomata
        # ( due to finite numerical precision in the optimiser, check to see if cost is very small  )
        J_opt = -f( gs_co2_mol_opt[i])
        if J_opt < 0:
            gs_co2_mol_opt[i] = 1e-3
            
        
        gs_h2o_mol_opt = gs_co2_mol_opt[i] * 1.6
        gs_h2o_m_s_opt = gs_h2o_mol_opt * pc.Rgas * ( Ta[i] + 273.15 ) / pa[i]
        gt_co2_mol     = 1.0 / ( 1.0 / gs_co2_mol_opt[i] + rb_co2_mol[i] )
        T_leaf_opt[i]  = calc_Tleaf( pc, gs_h2o_m_s_opt, Ta[i], Iabs[i], pa[i], ra[i], vpd_air[i] )
        A_opt[i]       = calc_A_from_gs( pc, nl, T_leaf_opt[i], gt_co2_mol, ca[i], pa[i], Ipar[i], oa[i], 
                                         nl.Ea, nl.eta, nl.Topt, nl.vcmax25, nl.jmax25, nl.fd, nl.q10, C4 = C4)
        ci_opt[i]      = np.clip( ca[i] - A_opt[i] * pa[i] / gt_co2_mol, 0, ca[i] )
    return A_opt, gs_co2_mol_opt, ci_opt, T_leaf_opt


def objective_function_LT( gs_co2_mol, nl, pc, Ta, ca, pa, oa, Iabs, Ipar, ra, vpd, vpd_air, rb_co2_mol, gcrit, Tl_i, rd_func, C4 ):
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
    
    # Calculate stomatal conductance to H20 
    
    gc_t       = 1.0 / ( 1.0 / gs_co2_mol + rb_co2_mol )
     
    # Calculate leaf temperature and fluxes from gs
    A               =calc_A_from_gs( pc, nl, Tl_i, gc_t, ca, pa, Ipar, oa, 
                                     Ea = nl.Ea, eta = nl.eta, Topt = nl.Topt, 
                                     vcmax25 = nl.vcmax25, jmax25 = nl.jmax25, 
                                     fd = nl.fd, q10 = nl.q10, rd_func = rd_func, C4 = C4 )
    F               = calc_cost_from_gcrit( gs_co2_mol, gcrit )
    
    # Calculate objective function
    obj               = A * F
    
    return obj


# =============================================================================
# The Air Temperature (AT) approach    
# =============================================================================
def numerical_solve_AT( nl, pc, Ta, ca, pa, oa, Iabs, ra, vpd, swp = None, 
                        Ipar = None, vpd_air = None, beta = None, rb_co2_mol = None, rd_func = None, C4 = False ):
    
    Ta, ca, pa, oa, Iabs, Ipar, ra, rb_co2_mol, vpd, vpd_air, swp, beta = normalize_inputs( Ta, ca, pa, oa, Iabs, Ipar, ra, rb_co2_mol, vpd, vpd_air, swp, beta )
    
    if all(v is None for v in vpd_air):
        print("Setting VPD_air = vpd")
        vpd_air = vpd
    
    if all(v is None for v in beta):
        print("Setting beta")
        if all(v is None for v in swp):
            raise Exception("Must supply either soil water potential or beta parameter")
        Ppd  = swp - nl.h * pc.g * pc.rho_w * pc.w
        beta = ( Ppd - nl.Pcrit ) / nl.rp 
    
    if all(v is None for v in rb_co2_mol):
        print("Setting rb_co2_mol")
        gb_h2o_m_s = 1.0 / ra
        gb_h2o_mol = gb_h2o_m_s * pa / ( pc.Rgas * ( Ta + 273.15 ) )
        gb_co2_mol = gb_h2o_mol / 1.6
        rb_co2_mol = 1.0 / gb_co2_mol
        
    if all(v is None for v in Ipar):
        print("Setting IPAR")
        # Calculate photosynthetically active radiation (Ipar, mol quanta m-2 s-1)
        fpar    = 0.5
        Ephoton = pc.h_planck * pc.c_light / pc.lambda_par
        Ipar    = fpar * Iabs / ( Ephoton * pc.Na )
   
    # Calculate maximum stomatal conductance as that which results in the cost function equal to zero
    vpd_mol        = vpd / pa
    gs_co2_mol_max = beta / vpd_mol
    
    T_leaf_opt     = np.zeros(len(Ta))
    A_opt          = np.zeros(len(Ta))
    ci_opt         = np.zeros(len(Ta))
    gs_co2_mol_opt = np.zeros(len(Ta))
    
    for i in range(len(Ta)):
        f = lambda gs_co2_mol: -objective_function_AT( gs_co2_mol, nl, pc, Ta[i], 
                                                       ca[i], pa[i], oa[i], Iabs[i], 
                                                       Ipar[i], ra[i], vpd[i], vpd_air[i], 
                                                       rb_co2_mol[i], gs_co2_mol_max[i], 
                                                       rd_func, C4 )
        # Calculate optimal stomatal conductance to CO2 (mol m-2 s-1)
        result = minimize_scalar( f, bounds = (0.0, gs_co2_mol_max[i] ), method = 'Bounded')
        gs_co2_mol_opt[i] = result.x
    
        # Check to make sure the objective function is still positive at gs_opt. Otherwise shut stomata
        J_opt = -f(gs_co2_mol_opt[i])
        if J_opt <= 0:
            gs_co2_mol_opt[i] = 1.0e-3
    
        # Calculate remaining gas exchange        
        gt_co2_mol     = 1.0 / ( 1.0 / gs_co2_mol_opt[i] + rb_co2_mol[i] )
        T_leaf_opt[i]  = Ta[i]
        A_opt[i]       = calc_A_from_gs( pc, nl, T_leaf_opt[i], gt_co2_mol, ca[i], pa[i], Ipar[i], oa[i], 
                                         nl.Ea, nl.eta, nl.Topt, nl.vcmax25, nl.jmax25, nl.fd, nl.q10, C4 = C4)
        ci_opt[i]      = np.clip(ca[i] - A_opt[i] * pa[i] / gt_co2_mol, 0, ca[i] )
    return A_opt, gs_co2_mol_opt, ci_opt, T_leaf_opt
    
def objective_function_AT( gs_co2_mol, nl, pc, Ta, ca, pa, oa, Iabs, Ipar, ra, vpd, vpd_air, rb_co2_mol, gcrit, rd_func, C4 ):
    """
    Calculate the objective function for simplified PGEN model with the leaf 
    energy balance integrated using the Air Temperature (AT) 
    approach
    
    Inputs:
    gs_co2_mol = Stomatal conductance to CO2               (mol m-2 s-1)
    nl         = namelist                                  (class containing model parameters )
    pc         = physical constants                        (class containing values of physical constants) 
    Ta         = Air temperature                           (C)
    ca         = Partial pressure of CO2 in the atmosphere (Pa)
    pa         = Atmospheric air pressure                  (Pa)
    oa         = Partial pressure of O2 in the atmosphere  (Pa)
    Iabs       = Absorbed incoming short-wave radiation    (W m-2)
    ra         = Aerodynamic resistance to water vapour    (s m-1)
    vpd        = Vapour pressure deficit                   (Pa)  
    swp        = Soil water potential                      (MPa)
    
    Outputs:
    obj        = Objective function ( A * f_psi_leaf )    
    """
 
    gc_t       = 1.0 / ( 1.0 / gs_co2_mol + rb_co2_mol ) 
     
    # Calculate leaf temperature and fluxes from gs
    Tl              = Ta
    
    # Calculate stomatal conductance to H20 
    A               = calc_A_from_gs( pc, nl, Tl, gc_t, ca, pa, Ipar, oa, 
                                      Ea = nl.Ea, eta = nl.eta, Topt = nl.Topt, 
                                      vcmax25 = nl.vcmax25, jmax25 = nl.jmax25, 
                                      fd = nl.fd, q10 = nl.q10, 
                                      rd_func = rd_func, C4 = C4 )
    
    F               = calc_cost_from_gcrit( gs_co2_mol, gcrit )

    # Calculate objective function
    obj               = A * F
    
    return obj

def save_params( save_path, file_name, Ea = None, Ea_SE = None, eta = None, eta_SE = None, 
                 Topt = None, Topt_SE = None, vcmax25 = None, vcmax25_SE = None, 
                 jmax25 = None, jmax25_SE = None, fd = None, fd_SE = None, 
                 q10 = None, q10_SE = None, beta_LT = None, beta_LT_SE = None,
                 beta_LTO = None, beta_LTO_SE = None, ra = None, ra_SE = None,
                 Is = None, Is_SE = None ):
   
    
    params         = ['Ea','eta','Topt','vcmax25','jmax25','fd','q10','ra','Is','beta_LT','beta_LTO']
    param_metric   = ['Value','SE']
    param_pairs    = list(product(params,param_metric))  
    param_columns  = pd.MultiIndex.from_tuples(param_pairs, names=['first', 'second'])
    df_params      = pd.Series(index = param_columns, dtype = 'float64' )
    
    df_params.loc['Ea','Value']       = Ea
    df_params.loc['Ea','SE']          = Ea_SE
    df_params.loc['eta','Value']      = eta
    df_params.loc['eta','SE']         = eta_SE
    df_params.loc['Topt','Value']     = Topt
    df_params.loc['Topt','SE']        = Topt_SE
    df_params.loc['vcmax25','Value']  = vcmax25
    df_params.loc['vcmax25','SE']     = vcmax25_SE
    df_params.loc['jmax25','Value']   = jmax25
    df_params.loc['jmax25','SE']      = jmax25_SE
    df_params.loc['fd','Value']       = fd
    df_params.loc['fd','SE']          = fd_SE
    df_params.loc['q10','Value']      = q10
    df_params.loc['q10','SE']         = q10_SE
    df_params.loc['beta_LT','Value']  = beta_LT
    df_params.loc['beta_LT','SE']     = beta_LT_SE
    df_params.loc['beta_LTO','Value'] = beta_LTO
    df_params.loc['beta_LTO','SE']    = beta_LTO_SE
    df_params.loc['ra','Value']       = ra
    df_params.loc['ra','SE']          = ra_SE
    df_params.loc['Is','Value']       = Is
    df_params.loc['Is','SE']          = Is_SE
    
    df_params.to_csv( save_path + file_name )
    
def save_out_data( save_path, file_name,
                   A_obs = None, gs_obs = None, ci_obs = None, Tl_obs = None,
                   Ta_obs = None, dT_obs = None, Pa_obs = None, Ca_obs = None,
                   VPD_obs = None, A_LT = None, gs_LT = None, ci_LT = None,
                   Tl_LT = None, dT_LT = None, A_LTO = None, gs_LTO = None, 
                   ci_LTO = None, Tl_LTO = None, dT_LTO = None, A_fit = None ):
    
    out_vars  = ['A','gsc','ci','Tl','Ta','dT','Pa','Ca','VPD']
    out_srcs  = ['LTO','LT','AT','obs']
    out_pairs = list(product(out_vars,out_srcs))  
    columns   = pd.MultiIndex.from_tuples(out_pairs, names=['first', 'second'])
    df_out    = pd.DataFrame( columns = columns, dtype = 'float64' )
    
    df_out.loc[:,('A','obs')]    = A_obs
    df_out.loc[:,('gsc','obs')]  = gs_obs
    df_out.loc[:,('ci','obs')]   = ci_obs
    df_out.loc[:,('Tl','obs')]   = Tl_obs
    df_out.loc[:,('Ta','obs')]   = Ta_obs
    df_out.loc[:,('dT','obs')]   = dT_obs
    df_out.loc[:,('Pa','obs')]   = Pa_obs
    df_out.loc[:,('Ca','obs')]   = Ca_obs
    df_out.loc[:,('VPD','obs')]  = VPD_obs

    df_out.loc[:,('Ta','LTO')]  = Ta_obs
    df_out.loc[:,('Ta','LT')]   = Ta_obs
    df_out.loc[:,('Ta','AT')]   = Ta_obs

    df_out.loc[:,('A','LT')]    = A_LT
    df_out.loc[:,('gsc','LT')]  = gs_LT
    df_out.loc[:,('ci','LT')]   = ci_LT
    df_out.loc[:,('Tl','LT')]   = Tl_LT
    df_out.loc[:,('dT','LT')]   = dT_LT

    df_out.loc[:,('A','LTO')]   = A_LTO
    df_out.loc[:,('gsc','LTO')] = gs_LTO
    df_out.loc[:,('ci','LTO')]  = ci_LTO
    df_out.loc[:,('Tl','LTO')]  = Tl_LTO
    df_out.loc[:,('dT','LTO')]  = dT_LTO

    df_out.loc[:,('A_farq','mod')] = A_fit
    
    df_out.to_csv( save_path + file_name )
