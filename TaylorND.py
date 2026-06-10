# -*- coding: utf-8 -*-
"""
Code to fit PGEN to unpublished data from Tyeen Taylor
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
from matplotlib.transforms import Bbox

# =============================================================================
# Load in data
# =============================================================================
file_path = ''
file_name = 'Mdat+Meta_WarmSite_251217.csv'

df = pd.read_csv(file_path  + file_name, parse_dates = True )

# =============================================================================
# Set up namelists
# =============================================================================
nl = namelist()
pc = physical_constants()


unique_measurements_total = df['m.id'].unique()
unique_species = df['sp.scr'].unique()
N_species      = len(unique_species)

use_species = unique_species
use_species = unique_species[[0,2,4]]
N_species   = len(use_species)

df_a = df[ ['A','gsw','Meas.Tair','Tleaf','VPDleaf',
            'Ci','Meas.Pa','LeafQ.Qabs','Rabs','gbw',
            'TleafEB','E','Ca','m.id','RHcham','gtc']].dropna()

# Set up figure
ncols = 3
nrows = 3
fig,axs = plt.subplots(ncols = ncols, nrows = nrows, figsize = (7*ncols, 6*nrows))#, sharey = 'row')
if ncols == 1:
    axs = axs[:,np.newaxis]
if nrows == 1:
    axs = axs[np.newaxis,:]
fig.subplots_adjust( wspace=0.2, hspace= 1.0 )

for i,species in enumerate(use_species):
    df_species = df[df['sp.scr'] == species]
    df_species = df_species[ ['A','gsw','Meas.Tair','Tleaf','VPDleaf',
                              'Ci','Meas.Pa','LeafQ.Qabs','Rabs','gbw',
                              'TleafEB','E','Ca','m.id','RHcham','gtc']] 
    df_species = df_species.dropna()

    A          = df_species['A']
    gs_h2o_mol = df_species['gsw']
    gs_co2_mol = gs_h2o_mol / 1.6
    Ta         = df_species['Meas.Tair']
    Tl         = df_species['Tleaf']
    TleafEB    = df_species['TleafEB']
    ci_ppm     = df_species['Ci']
    ca_ppm     = df_species['Ca']
    Pa         = df_species['Meas.Pa'] * 1.0e3
    Ci         = ci_ppm * Pa / 1.0e6
    Ca         = ca_ppm * Pa / 1.0e6
    PAR        = df_species['LeafQ.Qabs'] * 1.0e-6
    Oa         = 20 * Pa / 100
    Rabs       = df_species['Rabs']
    gb_h2o_mol = df_species['gbw']
    gb_co2_mol = gb_h2o_mol / 1.6
    rb_co2_mol = 1.0 / gb_co2_mol
    E          = df_species['E']
    gs_h2o_m_s = gs_h2o_mol * 8.314462 * ( Ta + 273.15 ) / Pa
    gb_h2o_m_s = gb_h2o_mol * 8.314462 * ( Ta + 273.15 ) / Pa
    gt_co2_mol = df_species['gtc']
    VPD_leaf   = df_species['VPDleaf'] * 1.0e3
    dT         = Tl - Ta
    SVP_air    = calc_esat_from_T( Ta )
    SVP_leaf   = calc_esat_from_T( Tl )
    VP_air     = SVP_leaf - VPD_leaf
    VPD_air    = SVP_air - VP_air 
    Ta_K = Ta + 273.15
    
    # =============================================================================
    # Fit Farquhar
    # =============================================================================        
    
    f = lambda gt_co2_mol, Ea, eta, Topt, vcmax25, jmax25, fd, q10, : calc_A_from_gs_C3( pc, nl, Tl, gt_co2_mol, Ca, Pa, PAR, Oa, Ea * 1.0e3, eta, Topt, vcmax25, jmax25, fd, q10, rd_func = None ) * 1.0e6
    popt, pcov = curve_fit( f, gt_co2_mol, A, p0 = [200, 2, 25, 1e-05, 1e-05, 0.1, 2.0],
                            bounds = ([1,1,20,1e-15,1e-7, 0.0, 1.0],[500,50,50,1e-2,1e-2,0.1,3.0]), maxfev = 5000)    
    
    print( popt )
    Ea, eta, Topt, vcmax25, jmax25, fd, q10 = popt   
    Ea_SE, eta_SE, Topt_SE, vcmax25_SE, jmax25_SE, fd_SE, q10_SE = np.sqrt(np.diag(pcov))
    
    nl.Ea      = Ea * 1e3
    nl.eta     = eta
    nl.Topt    = Topt
    nl.vcmax25 = vcmax25 
    nl.jmax25  = jmax25
    nl.fd      = fd
    nl.q10     = q10

    A_an_gs    = calc_A_from_gs_C3( pc, nl, Tl, gt_co2_mol, Ca, Pa, PAR, Oa, 
                                    Ea * 1.0e3, eta, Topt, vcmax25, jmax25, 
                                    fd, q10, rd_func = None ) * 1.0e6

    axs[0,i].scatter(A, A_an_gs, c = Tl, cmap = 'coolwarm' )
    axs[0,i].axline((A.mean(),A.mean()), slope = 1, ls = '--', c = 'grey')
    axs[0,i].set_xlabel('Observed A ($\mu$mol m$^{-2}$ s$^{-1}$)', size = 18)
    axs[0,i].set_ylabel('Fitted A ($\mu$mol m$^{-2}$ s$^{-1}$)', size = 18)
    axs[0,i].set_title( species, size = 18)
  
    # =============================================================================
    # Plot energy balance fit
    # =============================================================================      
    Is = Rabs
    ra = 1.0 / gb_h2o_m_s 
    dT_an = calc_Tleaf( pc, gs_h2o_m_s, Ta, Is, Pa, ra, VPD_air ) - Ta

    axs[1,i].scatter( dT, dT_an, c = VPD_air, cmap = 'coolwarm' )
    axs[1,i].axline( (dT.mean(),dT.mean()), slope = 1, ls = '--', c = 'grey' )
    axs[1,i].set_xlabel('Observed dT ($^o$C)', size = 18)
    axs[1,i].set_ylabel('Fitted dT ($^o$C)', size = 18)
    axs[1,i].set_title(species, size = 18 )

    
    # =============================================================================
    # Fit stomatal conductance
    # =============================================================================
    # Non-linear least squares
    idx = Tl <= 45
    f_gcrit_LTO = lambda Tair, beta: numerical_solve_LTO( nl, pc, Ta, Ca, Pa, Oa, Is, ra, vpd = VPD_leaf, 
                                                          Ipar = PAR, rb_co2_mol = rb_co2_mol, vpd_air = VPD_air, beta = beta )[1]
    
    f_gcrit_LT = lambda Tair, beta: numerical_solve_LT( nl, pc, Ta[idx], Ca[idx].values, Pa[idx], Oa[idx], Is, ra, vpd = VPD_leaf[idx].values, 
                                                        Ipar = PAR[idx].values, rb_co2_mol = rb_co2_mol[idx].values, vpd_air = VPD_air[idx].values, beta = beta )[1]


    popt, pcov = curve_fit( f_gcrit_LTO, Ta, gs_co2_mol, p0 = 0.0001, bounds = [0,0.5], diff_step = 1.0e-5 )
    print(popt)
    beta_LTO = popt[0]
    beta_LTO_SE = np.sqrt( np.diag( pcov ) )
    
    popt, pcov = curve_fit( f_gcrit_LT, Ta[idx].values, gs_co2_mol[idx].values, p0 = 0.0001, bounds = [0,0.5], diff_step = 1.0e-5 )
    print(popt)
    beta_LT = popt[0]
    beta_LT_SE = np.sqrt( np.diag( pcov ) )
    
    # =============================================================================
    # Calculate predicted gas exchange
    # =============================================================================
    A_LT, gs_LT, ci_LT, Tl_LT = numerical_solve_LT( nl, pc, Ta.values, Ca.values, Pa, Oa, Is, ra, vpd = VPD_leaf, Ipar = PAR,
                                                    rb_co2_mol = rb_co2_mol, vpd_air = VPD_air.values, beta = beta_LT )

    A_LTO, gs_LTO, ci_LTO, Tl_LTO = numerical_solve_LTO( nl, pc, Ta.values, Ca.values, Pa, Oa, Is, ra, vpd = VPD_leaf, Ipar = PAR,
                                                         rb_co2_mol = rb_co2_mol, vpd_air = VPD_air.values, beta = beta_LTO )
    
    
    axs[2,i].scatter(Ta, gs_co2_mol, color = 'black', facecolor = 'none', label = 'obs')
    axs[2,i].plot(Ta.iloc[np.argsort(Ta)], gs_LT[np.argsort(Ta)], color = '#DC267F', label = 'LT')
    axs[2,i].plot(Ta.iloc[np.argsort(Ta)], gs_LTO[np.argsort(Ta)], color = '#FFB000', label = 'LTO')
    axs[2,i].set_xlabel('Ta ($^o$C)', size = 18)
    axs[2,i].set_ylabel('g$_{sc}$ ($\mu$mol m$^{-2}$ s$^{-1}$)', size = 18)
    axs[2,i].set_title(species, size = 18 )
    
    # =============================================================================
    # Save fitted parameter values + SEs
    # Save predicted and observed gas exchange
    # =============================================================================
    params_path = 'parameter_vals/'
    save_params( params_path, 'TaylorND_%s_params.csv'%(species),
                 Ea = Ea, Ea_SE = Ea_SE, eta = eta, eta_SE = eta_SE, Topt = Topt, Topt_SE = Topt_SE, 
                 vcmax25 = vcmax25, vcmax25_SE = vcmax25_SE, jmax25 = jmax25, jmax25_SE = jmax25_SE,
                 fd = fd, fd_SE = fd_SE, q10 = q10, q10_SE = q10_SE, beta_LT = beta_LT, 
                 beta_LT_SE = beta_LT_SE, beta_LTO = beta_LTO, beta_LTO_SE = beta_LTO_SE, ra = None, 
                 ra_SE = None, Is = None, Is_SE = None )
    out_data_path = 'Modelling_results/'
    save_out_data( out_data_path, 'TaylorND_%s_out.csv'%(species),
                   A_obs = A.values, gs_obs = gs_co2_mol.values, ci_obs = Ci.values, Tl_obs = Tl.values,
                   Ta_obs = Ta.values, dT_obs = dT.values, Pa_obs = Pa.values, Ca_obs = Ca.values,
                   VPD_obs = VPD_leaf.values,
                   A_LT = A_LT, gs_LT = gs_LT, ci_LT = ci_LT, Tl_LT = Tl_LT, 
                   dT_LT = Tl_LT - Ta.values,
                   A_LTO = A_LTO, gs_LTO = gs_LTO, ci_LTO = ci_LTO, Tl_LTO = Tl_LTO, 
                   dT_LTO = Tl_LTO - Ta.values,
                   A_fit = A_an_gs.values )
