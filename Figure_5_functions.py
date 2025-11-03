# -*- coding: utf-8 -*-
"""
Functions required to produce figure 4 of Jones et al

NOTE: Requires the data file from Diao et al (2024) which can
be found at https://doi.org/10.1111/nph.19558

@author: srgj201
"""

import numpy as np
from scipy.optimize import minimize_scalar
from PGEN_functions import calc_cost_from_lwp, calc_Tleaf
from loess import loess_1d
from tqdm import tqdm

#%%
# =============================================================================
# The Leaf Temperature within Optimisation (LTO) approach    
# =============================================================================
def numerical_solve_Diao_LTO( pc, Ta, ca, pa, oa, Is, ra, vpd, gcrit, Jmax_params, species):
    # Calculate maximum possible stomatal conductance
    gs_co2_mol_max = gcrit / 1.6
    gs_co2_mol_min = 1e-10
    
    T_leaf_opt = np.zeros(len(Ta))
    A_opt      = np.zeros(len(Ta))
    ci_opt     = np.zeros(len(Ta))
    gs_h2o_opt = np.zeros(len(Ta))
    
    for i in range(len(Ta)):
    
        f = lambda gs_co2_mol: -objective_function_Diao_LTO( gs_co2_mol, pc, Ta[i], ca, pa, oa, Is, ra, vpd, gcrit, Jmax_params, species)
        
        # Extract optimal gs to CO2 and calculate optimal leaf temperature, ci, and fluxes
        result = minimize_scalar( f, bounds = (0.0, gs_co2_mol_max), method = 'Bounded')
        gs_co2_mol_opt = result.x
        
        # Check to make sure the objective function is still positive at gs_opt. Otherwise we set gs to zero
        obj_opt = -f(gs_co2_mol_opt)
        if obj_opt <= 0:
            gs_co2_mol_opt = gs_co2_mol_min
        
        gs_h2o_opt[i]  = gs_co2_mol_opt * 1.6
        gs_h2o_m_s_opt = gs_h2o_opt[i] * 8.314462 * ( Ta[i] + 273.15 ) / pa
        T_leaf_opt[i]  = calc_Tleaf( pc, gs_h2o_m_s_opt, Ta[i], Is, pa, ra, vpd )
        A_opt [i]      = calc_A_from_gs_Diao( pc, T_leaf_opt[i], gs_co2_mol_opt, ca, pa, oa, Jmax_params, species )
        ci_opt[i]      = ca - A_opt[i] * pa / gs_co2_mol_opt
    return A_opt, gs_h2o_opt, ci_opt, T_leaf_opt
    
def objective_function_Diao_LTO( gs_co2_mol, pc, Ta, ca, pa, oa, Is, ra, vpd, gcrit, Jmax_params, species ):
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
    
    # Calculate stomatal conductance to H20 
    gs_h2o_mol = gs_co2_mol * 1.6                              # Stomatal conductance to H2O (mol m-2 s-1)
    gs_h2o_m_s = gs_h2o_mol * 8.314462 * ( Ta + 273.15 ) / pa  # Stomatal conductance to H2O (m s-1)
     
    # Calculate leaf temperature and fluxes from gs
    Tl              = calc_Tleaf( pc, gs_h2o_m_s, Ta, Is, pa, ra, vpd )      # N.B Leaf temperature is a function of gs WITHIN the objective function
    A               = calc_A_from_gs_Diao( pc, Tl, gs_co2_mol, ca, pa, oa, Jmax_params, species )
    F               = calc_cost_from_gcrit( gs_h2o_mol, gcrit )
    
    # Calculate objective function
    obj               = np.clip(A * F, 0, None)
    
    return obj
#%%
# =============================================================================
# The Leaf Temperature (LT) approach    
# =============================================================================
def numerical_solve_Diao_LT( pc, Ta, ca, pa, oa, Is, ra, vpd, gcrit, Jmax_params, species):
    # Calculate maximum possible stomatal conductance
    gs_co2_mol_max = gcrit / 1.6
    gs_co2_mol_min = 1e-10
    
    T_leaf_opt = np.zeros(len(Ta))
    A_opt      = np.zeros(len(Ta))
    ci_opt     = np.zeros(len(Ta))
    gs_h2o_opt = np.zeros(len(Ta))
    
    for i in range(len(Ta)):
        Tl_i           = Ta[i]
        convergence_Tl = False
        while not convergence_Tl:
            f = lambda gs_co2_mol: -objective_function_Diao_LT( gs_co2_mol, pc, Ta[i], ca, pa, oa, Is, ra, vpd, gcrit, Tl_i, Jmax_params, species )
        
            # Extract optimal gs to CO2 and calculate optimal leaf temperature
            result = minimize_scalar( f, bounds = (0.0, gs_co2_mol_max), method = 'Bounded')
            gs_co2_mol_opt = result.x
            
            # Check the optimal gs still gives a positive objective function (otherwise set gs to 0)
            obj_opt = -f(gs_co2_mol_opt)
            if obj_opt<=0:
                gs_co2_mol_opt = gs_co2_mol_min
            
            gs_h2o_mol_opt = gs_co2_mol_opt * 1.6
            gs_h2o_m_s_opt = gs_h2o_mol_opt * 8.314462 * ( Ta[i] + 273.15 ) / pa
            Tl_opt         = calc_Tleaf( pc, gs_h2o_m_s_opt, Ta[i], Is, pa, ra, vpd )
            
            # Check for convergence in Tleaf
            if np.abs(Tl_opt - Tl_i) < 0.001:
                convergence_Tl = True
            else:
                Tl_i = Tl_opt
                
        # Calculate optimal A, Tleaf, gs and ci
        gs_h2o_opt[i]  = gs_co2_mol_opt * 1.6
        A_opt[i]       = calc_A_from_gs_Diao( pc, Tl_opt, gs_co2_mol_opt, ca, pa, oa, Jmax_params, species )
        ci_opt[i]      = ca - A_opt[i] * pa / gs_co2_mol_opt
        T_leaf_opt[i]  = Tl_opt
        
    return A_opt, gs_h2o_opt, ci_opt, T_leaf_opt
    
def objective_function_Diao_LT( gs_co2_mol, pc, Ta, ca, pa, oa, Is, ra, vpd, gcrit, Tl_i, Jmax_params, species ):
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
    gs_h2o_mol = gs_co2_mol * 1.6  # Stomatal conductance to H2O (mol m-2 s-1)
     
    # Calculate leaf temperature and fluxes from gs
    A               = calc_A_from_gs_Diao( pc, Tl_i, gs_co2_mol, ca, pa, oa, Jmax_params, species )
    F               = calc_cost_from_gcrit( gs_h2o_mol, gcrit )
    
    # Calculate objective function
    obj               = np.clip(A * F, 0, None)
    
    return obj
#%%
# =============================================================================
# The Air Temperature (AT) approach    
# =============================================================================
def numerical_solve_Diao_AT( pc, Ta, ca, pa, oa, Is, ra, vpd, gcrit, Jmax_params, species):
    # Calculate maximum possible stomatal conductance
    gs_co2_mol_max = gcrit / 1.6
    gs_co2_mol_min = 1e-10
    
    T_leaf_opt = np.zeros(len(Ta))
    A_opt      = np.zeros(len(Ta))
    ci_opt     = np.zeros(len(Ta))
    gs_h2o_opt = np.zeros(len(Ta))
    for i in range(len(Ta)):
    
        # Create function to minimise (just -1 times the objective function)
        f = lambda gs_co2_mol: -objective_function_Diao_AT( gs_co2_mol, pc, Ta[i], ca, pa, oa, Is, ra, vpd, gcrit, Jmax_params, species)
        
        # Solve numerically using the minimize_scalar method from scipy
        result = minimize_scalar( f, bounds = (0.0, gs_co2_mol_max), method = 'Bounded')
        
        # Extract optimal gs to CO2 and calculate optimal leaf temperature, ci, and fluxes
        gs_co2_mol_opt = result.x
        
        # Check to make sure the objective function is still positive. Otherwise we set optimal stomatal conductance to zeor
        obj_opt = -f(gs_co2_mol_opt)
        if obj_opt<=0:
            gs_co2_mol_opt = gs_co2_mol_min
        
        # Calculate optimal leaf temperature, photosynthesis and ci
        T_leaf_opt[i]     = Ta[i]
        A_opt[i]          = calc_A_from_gs_Diao( pc, T_leaf_opt[i], gs_co2_mol_opt, ca, pa, oa, Jmax_params, species )
        ci_opt[i]         = ca - A_opt[i] * pa / gs_co2_mol_opt
        gs_h2o_opt[i]     = gs_co2_mol_opt * 1.6
    
    return A_opt, gs_h2o_opt, ci_opt, T_leaf_opt
    
def objective_function_Diao_AT( gs_co2_mol, pc, Ta, ca, pa, oa, Is, ra, vpd, gcrit, Jmax_params, species):
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
  
    # Calculate stomatal conductance to H20 
    gs_h2o_mol = gs_co2_mol * 1.6                              # Stomatal conductance to H2O (mol m-2 s-1)
     
    # Calculate leaf temperature and fluxes from gs
    Tl              = Ta                                                   # N.B. Leaf temperature is always equal to air temperature
    A               = calc_A_from_gs_Diao( pc, Tl, gs_co2_mol, ca, pa, oa, Jmax_params, species )
    F               = calc_cost_from_gcrit( gs_h2o_mol, gcrit )
    
    # Calculate objective function
    obj               = np.clip(A * F, 0, None)
    
    return obj

#%%
# =============================================================================
# Simplified PGEN functions
# =============================================================================

def calc_A_from_gs_Diao( pc, Tleaf, gs_co2_mol, ca, pa, Oa, Jmax_params, species):
    """
    Calculate photosynthesis from stomatal conductance. For figure 4 we assume
    that photosynthesis is only given by the RuBP regenration rate. Rd is calculated
    using the measure fits from Diao et al (2024).
    
    Inputs:
    pc         = physical constants                        (class containing values of physical constants) 
    nl         = namelist                                  (class containing model parameters )
    Tleaf      = Leaf temperature                          (C)
    gs_co2_mol = Stomatal conductance to CO2               (mol m-2 s-1)
    ca         = Partial pressure of CO2 in the atmosphere (Pa)
    Is         = Absorbed short-wave radiation             (mol quanta m-2 s-1)
    Oa         = Partial pressure of O2 in the atmosphere  (Pa)
    """
    # Calculate the CO2 compensation point in the absence of dark respiration,
    # and Michaelis-menten parameters for carboxylation
    Kc, Ko, gamma_star = calc_photosynthetic_params( Tleaf, Oa )
    
    # Calculate the maximum rate of photosynthesis
    J = calc_J_Diao( Tleaf, *Jmax_params)
    
    # Calculate dark respiration
    rd         = calc_rd_Diao( Tleaf, species )
    
    # Ribulose-bisphosphate regeneration-limited photosynthesis
    a          = 1.0
    b          = - ( ( J ) + ( ca + 2 * gamma_star ) * gs_co2_mol / pa - rd )
    c          = ( J ) * ( ca - gamma_star ) * gs_co2_mol / pa - ( ca + 2 * gamma_star ) * rd * gs_co2_mol / pa
    Wl         = ( -b - ( b**2.0 - 4.0 * a * c )**0.5 ) / ( 2.0 * a )
    A = Wl
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
   

def calc_J_Diao( T_leaf, Tmin, Tmax, Topt, Jmax ):
    """
    Calculate the electron transport rate
    
    Inputs:
    T_leaf = Leaf temperature                                     (C)
    Tmin   = Minimum temperature parameter for electron transport (C)
    Tmax   = Maximum temperature parameter for electron transport (C)
    Topt   = Optimum temperature parameter for electron transport (C)
    Jmax   = Light-saturated potential electron transport rate    (µmol m2 s−1) 
    Returns:
    J      = Electron transport rate
    """
    T_leaf = np.clip( T_leaf, Tmin, Tmax )
    f = ( ( Tmax - T_leaf ) / ( Tmax - Topt ) * 
         ( ( T_leaf - Tmin ) / ( Topt - Tmin ) ) ** ( ( Topt - Tmin ) / ( Tmax - Topt ) ) )
    J = ( Jmax ) * np.clip( f, 0, None )
    return J

def calc_rd_Diao( T_leaf, species ):
    """
    Calculate Rd using the fitted equations from Diao et al (2024)
    """
    if species == 'Fagus':
        A = 0.11245121
        B = 0.05257809
    elif species == 'Quercus':
        A = 0.10518226
        B = 0.06035728
    elif species == 'Tilia':
        A = 0.08973579
        B = 0.05236533
    else:
        raise Exception("'%s' is not a valid species"%(species))
    rd         = 0.5 * 1e-6 * A * np.exp( B * T_leaf )  
    
    return rd


def calc_cost_from_gcrit( gs_h2o_mol, gcrit ):
    return np.clip( 1 - gs_h2o_mol / gcrit, 0, None )

#%%
# =============================================================================
# Functions for the LOESS smoother
# =============================================================================
def bootstrap_loess(x, y, frac=0.5, n_boot=100, x_pred=None, degree = 1):
    if x_pred is None:
        x_pred = np.linspace(np.min(x), np.max(x), 100)
    
    # Store predictions from each bootstrap
    preds = np.zeros((n_boot, len(x_pred)))
    
    for i in tqdm(range(n_boot)):
        # Sample with replacement
        idx = np.random.choice(len(x), size=len(x), replace=True)
        x_sample = x[idx]
        y_sample = y[idx]

        # Fit loess on bootstrap sample
        model    = loess_1d.loess_1d(x_sample, y_sample, xnew=x_pred, frac=frac, degree = degree)
        preds[i] = model[1]
    
    # Compute mean and standard error
    mean_fit = np.mean(preds, axis=0)
    se_fit = np.std(preds, axis=0, ddof=1)
    
    return x_pred, mean_fit, se_fit





