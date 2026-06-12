# -*- coding: utf-8 -*-
"""
Code to fit PGEN to data from Slot and Winter (2017)
https://doi.org/10.1111/nph.14469
@author: srgj201
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PGEN_functions_complete import namelist, physical_constants
from PGEN_functions_complete import numerical_solve_LTO, numerical_solve_LT
from PGEN_functions_complete import calc_A_from_gs_C3, calc_esat_from_T
from PGEN_functions_complete import calc_Tleaf, save_params, save_out_data
from scipy.optimize import curve_fit
import math
from sklearn.metrics import r2_score


# =============================================================================
# Load in data
# =============================================================================
path     = ''
filename = 'Slot_Winter_17_raw.csv'
df       = pd.read_excel( path + filename )

# =============================================================================
# Set up namelists
# =============================================================================
nl = namelist()
pc = physical_constants()


# =============================================================================
# Set up figures
# =============================================================================
unique_species = df['Species'].unique()
N_species      = 25
# unique_species = ['Spondias mombin']

ncols = math.ceil( N_species**0.5 )
nrows = math.ceil( N_species / ncols )
fig,axs = plt.subplots(ncols = ncols, nrows = nrows, figsize = (6*ncols, 4*nrows))
if type(axs) == np.ndarray:
    axs = axs.reshape(-1)
else:
    axs = [axs]
    
ncols = math.ceil( N_species**0.5 )
nrows = math.ceil( N_species / ncols )
fig_A,axs_A = plt.subplots(ncols = ncols, nrows = nrows, figsize = (6*ncols, 4*nrows))
if type(axs_A) == np.ndarray:
    axs_A = axs_A.reshape(-1)
else:
    axs_A = [axs_A]

ncols = math.ceil( N_species**0.5 )
nrows = math.ceil( N_species / ncols )
fig_dT,axs_dT = plt.subplots(ncols = ncols, nrows = nrows, figsize = (6*ncols, 4*nrows))
if type(axs_dT) == np.ndarray:
    axs_dT = axs_dT.reshape(-1)
else:
    axs_dT = [axs_dT]   
    
ncols = math.ceil( N_species**0.5 )
nrows = math.ceil( N_species / ncols )
fig_ci,axs_ci = plt.subplots(ncols = ncols, nrows = nrows, figsize = (6*ncols, 4*nrows))
if type(axs_ci) == np.ndarray:
    axs_ci = axs_ci.reshape(-1)
else:
    axs_ci = [axs_ci]
    
count = 0
for i,species in enumerate(unique_species):
    df_species = df[ df['Species'] == species ]
    df_species = df_species.dropna( subset = ['Photo', 'Cond', 'Tair', 'Tleaf', 'VpdL', 'CO2R','Ci'])
    if df_species.empty:
        continue
    print(species)
    # Extract data
    A          = df_species['Photo']
    gs_h2o_mol = df_species['Cond']
    Ta         = df_species['Tair']
    Tl         = df_species['Tleaf']
    ci_ppm     = df_species['Ci']
    ca_ppm     = df_species['CO2R'] 
    VPD_leaf   = df_species['VpdL'] * 1e3
    Pa         = 101325.0
    SVP_air    = calc_esat_from_T( Ta ) 
    SVP_leaf   = calc_esat_from_T( Tl )
    VP_air     = SVP_leaf - VPD_leaf
    VPD_air    = SVP_air - VP_air
    gs_co2_mol = gs_h2o_mol / 1.6
    gs_h2o_m_s = gs_h2o_mol * 8.314462 * ( Ta + 273.15 ) / Pa
    dT         = Tl - Ta
    PAR        = 1000e-6 * 0.85
    Oa_pc      = 20.0
    Oa         = Oa_pc * Pa / 100.0
    Ca         = ca_ppm * Pa / 1e6
    Ci         = ci_ppm * Pa / 1e6
    gt_co2_mol = A / ( ca_ppm - ci_ppm )
    rb_co2_mol = np.clip(1.0 / gt_co2_mol - 1.0 / gs_co2_mol,0,None)
    
    # =============================================================================
    # Fit Enerfy Balance
    # =============================================================================
    f = lambda gs, Is, ra: calc_Tleaf( pc, gs, Ta, Is, Pa, ra, VPD_air ) - Ta
    
    popt, pcov = curve_fit( f, gs_h2o_m_s, dT, p0 = [300,10], bounds = ([0,0.1],[1000,10000]))
    Is, ra = popt
    Is_SE, ra_SE = np.sqrt( np.diag(pcov) )

    
    dT_an = calc_Tleaf( pc, gs_h2o_m_s, Ta, Is, Pa, ra, VPD_air ) - Ta
    
    # Only use species for which the leaf temperature is consistent with the energy balance equation
    r2 = r2_score( dT, dT_an )
    if r2<0.5:
        continue
    
    # =============================================================================
    # Fit photosynthetic parameters
    # =============================================================================

    f = lambda gt_co2_mol, Ea, eta, Topt, vcmax25, \
               jmax25, fd, q10, : calc_A_from_gs_C3( pc, nl, Tl, gt_co2_mol, Ca, 
                                                     Pa, PAR, Oa, Ea * 1.0e3, eta, 
                                                     Topt, vcmax25, jmax25, fd, 
                                                     q10, rd_func = None ) * 1.0e6
    
    popt, pcov = curve_fit( f, gt_co2_mol, A, p0 = [40, 20, 30, 1e-4, 1e-4, 0.01, 2.0],
                            bounds = ([1,1,20,1e-15,1e-7, 0.0, 1.0],[200,50,50,1e-3,1e-3,0.1,3.0]),
                            maxfev = 5000)    

    
    Ea, eta, Topt, vcmax25, jmax25, fd, q10 = popt   
    Ea_SE, eta_SE, Topt_SE, vcmax25_SE, jmax25_SE, fd_SE, q10_SE = np.sqrt( np.diag( pcov ) )
    
    nl.Ea      = Ea * 1e3
    nl.eta     = eta
    nl.Topt    = Topt
    nl.vcmax25 = vcmax25 
    nl.jmax25  = jmax25
    nl.fd      = fd
    nl.q10     = q10
    
    A_an_gs    = calc_A_from_gs_C3( pc, nl, Tl, gt_co2_mol, Ca, Pa, PAR, Oa, Ea * 1.0e3, 
                                    eta, Topt, vcmax25, jmax25, fd, q10, rd_func = None ) * 1.0e6

    # =============================================================================
    # Fit stomatal conductance
    # =============================================================================
    # Non-linear least squares
    f_gcrit_LTO = lambda Tair, beta: numerical_solve_LTO( nl, pc, Ta, Ca, Pa, Oa, Is, ra, vpd = VPD_leaf, 
                                                          Ipar = PAR, rb_co2_mol = rb_co2_mol, vpd_air = VPD_air, beta = beta )[1]
    
    f_gcrit_LT = lambda Tair, beta: numerical_solve_LT( nl, pc, Ta, Ca, Pa, Oa, Is, ra, 
                                                        vpd = VPD_leaf, Ipar = PAR, 
                                                        rb_co2_mol = rb_co2_mol, vpd_air = VPD_air, 
                                                        beta = beta )[1]


    popt, pcov = curve_fit( f_gcrit_LTO, Ta, gs_co2_mol, p0 = 0.0001, bounds = [0,0.01])#, diff_step = 1.0e-5 )
    # print(popt)
    beta_LTO = popt[0]
    beta_LTO_SE = np.sqrt( np.diag( pcov ) )
    
    popt, pcov = curve_fit( f_gcrit_LT, Ta.values, gs_co2_mol.values, p0 = 0.0001, bounds = [0,0.01])#, diff_step = 1.0e-5 )
    # print(popt)
    beta_LT = popt[0]
    beta_LT_SE = np.sqrt( np.diag( pcov ) )
    
    # =============================================================================
    # Calculate predicted gas exchange
    # =============================================================================
    A_LT, gs_LT, ci_LT, Tl_LT = numerical_solve_LT( nl, pc, Ta.values, Ca.values, Pa, Oa, Is, ra, vpd = VPD_leaf, Ipar = PAR,
                                                    rb_co2_mol = rb_co2_mol, vpd_air = VPD_air.values, beta = beta_LT )

    A_LTO, gs_LTO, ci_LTO, Tl_LTO = numerical_solve_LTO( nl, pc, Ta.values, Ca.values, Pa, Oa, Is, ra, vpd = VPD_leaf, Ipar = PAR,
                                                         rb_co2_mol = rb_co2_mol, vpd_air = VPD_air.values, beta = beta_LTO )


    axs[count].set_title(species)
    axs[count].scatter(Ta, gs_co2_mol, c = VPD_leaf, cmap = 'coolwarm')
    axs[count].plot(Ta.iloc[np.argsort(Ta)], gs_LT[np.argsort(Ta)], color = '#DC267F')
    axs[count].plot(Ta.iloc[np.argsort(Ta)], gs_LTO[np.argsort(Ta)], color = '#FFB000')

    axs_A[count].set_title(species)
    axs_A[count].scatter(Ta, A, c = VPD_leaf, cmap = 'coolwarm')
    axs_A[count].plot(Ta.iloc[np.argsort(Ta)], A_LT[np.argsort(Ta)]*1.0e6, color = '#DC267F')
    axs_A[count].plot(Ta.iloc[np.argsort(Ta)], A_LTO[np.argsort(Ta)]*1.0e6, color = '#FFB000')

    axs_ci[count].set_title(species)        
    axs_ci[count].scatter(Ta, Ci, c = VPD_leaf, cmap = 'coolwarm')
    axs_ci[count].plot(Ta.iloc[np.argsort(Ta)], ci_LT[np.argsort(Ta)], color = '#DC267F')
    axs_ci[count].plot(Ta.iloc[np.argsort(Ta)], ci_LTO[np.argsort(Ta)], color = '#FFB000')
    
    axs_dT[count].set_title(species)
    axs_dT[count].scatter(Ta, dT , c = VPD_leaf, cmap = 'coolwarm')
    axs_dT[count].plot(Ta.iloc[np.argsort(Ta)], Tl_LT[np.argsort(Ta)] - Ta.iloc[np.argsort(Ta)], color = '#DC267F')
    axs_dT[count].plot(Ta.iloc[np.argsort(Ta)], Tl_LTO[np.argsort(Ta)] - Ta.iloc[np.argsort(Ta)], color = '#FFB000')
    
    # =============================================================================
    # Save fitted parameter values + SEs
    # Save predicted and observed gas exchange
    # =============================================================================
    params_path = 'parameter_vals/'
    save_params( params_path, 'SlotWinter24_%s_params.csv'%(species),
                 Ea = Ea, Ea_SE = Ea_SE, eta = eta, eta_SE = eta_SE, Topt = Topt, Topt_SE = Topt_SE, 
                 vcmax25 = vcmax25, vcmax25_SE = vcmax25_SE, jmax25 = jmax25, jmax25_SE = jmax25_SE,
                 fd = fd, fd_SE = fd_SE, q10 = q10, q10_SE = q10_SE, beta_LT = beta_LT, 
                 beta_LT_SE = beta_LT_SE, beta_LTO = beta_LTO, beta_LTO_SE = beta_LTO_SE, ra = ra, 
                 ra_SE = ra_SE, Is = Is, Is_SE = Is_SE )
    out_data_path = 'Modelling_results/'
    save_out_data( out_data_path, 'SlotWinter24_%s_out.csv'%(species),
                   A_obs = A.values, gs_obs = gs_co2_mol.values, ci_obs = Ci.values, Tl_obs = Tl.values,
                   Ta_obs = Ta.values, dT_obs = dT.values, Pa_obs = Pa, Ca_obs = Ca.values,
                   VPD_obs = VPD_leaf.values,
                   A_LT = A_LT, gs_LT = gs_LT, ci_LT = ci_LT, Tl_LT = Tl_LT, 
                   dT_LT = Tl_LT - Ta.values,
                   A_LTO = A_LTO, gs_LTO = gs_LTO, ci_LTO = ci_LTO, Tl_LTO = Tl_LTO, 
                   dT_LTO = Tl_LTO - Ta.values,
                   A_fit = A_an_gs.values )
    count+=1
    
fig.suptitle('Stomatal Conductance to CO2', size = 20)
fig_A.suptitle('Photosynthesis', size = 20)
fig_ci.suptitle('Internal leaf CO2 partial pressure', size = 20)
fig_dT.suptitle('Leaf temperature - Air temperature', size = 20)
for f,a in [[fig,axs],[fig_A,axs_A],[fig_ci,axs_ci],[fig_dT,axs_dT]]:
    h,l = a[0].get_legend_handles_labels()
    f.legend(h,l, loc = 'upper center', bbox_to_anchor = [0.5,0.0], ncol = 3, prop={'size': 18}  )
