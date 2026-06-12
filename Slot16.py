# -*- coding: utf-8 -*-
"""
Code to fit PGEN to data from Slot et al (2016)
https://doi.org/10.1071/FP15320
@author: srgj201
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PGEN_functions_complete import namelist, physical_constants
from PGEN_functions_complete import numerical_solve_LTO, numerical_solve_LT
from PGEN_functions_complete import calc_A_from_gs_C3
from PGEN_functions_complete import calc_Tleaf, save_params, save_out_data
from scipy.optimize import curve_fit
from matplotlib.transforms import Bbox

# =============================================================================
# Load in data
# =============================================================================
path     = ''
filename = 'Slot_16_raw.csv'
df       = pd.read_csv(path+filename)

# =============================================================================
# Set up namelists
# =============================================================================
nl = namelist()
pc = physical_constants()

unique_species = df['Species'].unique()


# =============================================================================
# Fit photosynthesis, energy balance and stomtatal conductance for each species
# =============================================================================
# Set up figure
ncols = 3
nrows = 3
fig,axs = plt.subplots(ncols = ncols, nrows = nrows, figsize = (7*ncols, 6*nrows))#, sharey = 'row')
if ncols == 1:
    axs = axs[:,np.newaxis]
if nrows == 1:
    axs = axs[np.newaxis,:]
fig.subplots_adjust( wspace=0.2, hspace= 1.0 )

for i,species in enumerate(unique_species):
    #Extract data
    df_species = df[ df['Species'] == species ]

    if species == 'Ficus insipida':
        df_species.loc[ df_species['gs'].idxmax(), 'gs' ] = np.nan
    
    df_species = df_species.dropna(subset = ['CO2 exchange (umol s-1m-2)','gs','Tset','Tleaf','Ci','VPD '])
    A          = df_species['CO2 exchange (umol s-1m-2)']
    gs_h2o_mol = df_species['gs']
    Ta         = df_species['Tset']
    Tl         = df_species['Tleaf']
    VPD_leaf   = df_species['VPD '] * 1e3
    Pa         = 101325.0
    O2_pc      = df_species['O2'] / 100.0
    Oa         = Pa * O2_pc
    ci_ppm     = df_species['Ci'] 
    Ci         = ci_ppm * Pa / 1.0e6
    X          = df_species['Ci.Ca']
    Ca         = Ci / X
    ca_ppm     = Ca * 1.0e6 / Pa
    PAR        = 1000e-6
    gs_co2_mol = gs_h2o_mol / 1.6
    gs_h2o_m_s = gs_h2o_mol * 8.314462 * ( Ta + 273.15 ) / Pa
    dT         = Tl - Ta
    SVPcham    = df_species['saturated vapor pressure (mb)'] * 100
    VPcham     = df_species['out- vapor pressure (Pa)'] * 100
    VPD_air    = SVPcham - VPcham
    

    
    # =============================================================================
    # Fit Energy balance
    # =============================================================================    
    # Non-linear least squares
    f = lambda gs, Is, ra: calc_Tleaf( pc, gs, Ta, Is, Pa, ra, VPD_air ) - Ta

    popt, pcov = curve_fit( f, gs_h2o_m_s, dT, p0 = [300,10], bounds = ([0,0.1],[1000,50]))
    Is, ra = popt
    Is_SE, ra_SE = np.sqrt( np.diag(pcov) )
    print(Is, ra)

    gb_h2o_m_s = 1.0 / ra
    gb_h2o_mol = gb_h2o_m_s * Pa / ( 8.314462 * ( Ta + 273.15 ) )
    gb_co2_mol = gb_h2o_mol / 1.6
    rb_co2_mol = 1.0 / gb_h2o_mol
    gt_co2_mol = 1.0 / ( 1.0 / gs_co2_mol + rb_co2_mol )

    dT_an = calc_Tleaf( pc, gs_h2o_m_s, Ta, Is, Pa, ra, VPD_air ) - Ta
    
    
    # Plot fitted energy balance
    SVPcham_all = df['saturated vapor pressure (mb)'] * 100
    VPcham_all  = df['out- vapor pressure (Pa)'] * 100
    VPD_air_all = SVPcham_all - VPcham_all
    vmin = VPD_air_all.min()
    vmax = VPD_air_all.max()
    axs[1,i].scatter( dT, dT_an, c = VPD_air, cmap = 'coolwarm', vmin = vmin, vmax = vmax )
    axs[1,i].axline( (dT.mean(),dT.mean()), slope = 1, ls = '--', c = 'grey' )
    axs[1,i].set_xlabel('Observed dT ($^o$C)', size = 18)
    axs[1,i].set_ylabel('Fitted dT ($^o$C)', size = 18)
    axs[1,i].set_title(species, size = 18)
    
    # =============================================================================
    # Fit Farquhar
    # =============================================================================  
    
    f = lambda gt_co2_mol, Ea, eta, Topt, vcmax25, \
               jmax25, fd, q10, : calc_A_from_gs_C3( pc, nl, Tl, gt_co2_mol, Ca, 
                                                     Pa, PAR, Oa, Ea * 1.0e3, eta, 
                                                     Topt, vcmax25, jmax25, fd, 
                                                     q10, rd_func = None ) * 1.0e6
    
    popt, pcov = curve_fit( f, gt_co2_mol, A, p0 = [40, 20, 30, 1e-4, 1e-4, 0.01, 2.0],
                            bounds = ([1,1,20,1e-15,1e-7, 0.0, 1.0],[200,50,50,1e-3,1e-3,0.1,3.0]))    

    
    Ea, eta, Topt, vcmax25, jmax25, fd, q10 = popt   
    Ea_SE, eta_SE, Topt_SE, vcmax25_SE, jmax25_SE, fd_SE, q10_SE = np.sqrt( np.diag( pcov ) )
    
    nl.Ea      = Ea * 1e3
    nl.eta     = eta
    nl.Topt    = Topt
    nl.vcmax25 = vcmax25 
    nl.jmax25  = jmax25
    nl.fd      = fd
    nl.q10     = q10
    
    print("Ea = %s \u00B1 %s"%(Ea, Ea_SE))
    print("eta = %s \u00B1 %s"%( eta, eta_SE))
    print("Topt = %s \u00B1 %s"%( Topt, Topt_SE))
    print("vcmax25 = %s \u00B1 %s"%( vcmax25, vcmax25_SE))
    print("jmax25 = %s \u00B1 %s"%( jmax25, jmax25_SE))
    print("fd = %s \u00B1 %s"%( fd, fd_SE))
    print("q10 = %s \u00B1 %s"%( q10, q10_SE))
    
    A_an_gs    = calc_A_from_gs_C3( pc, nl, Tl, gt_co2_mol, Ca, Pa, PAR, Oa, Ea * 1.0e3, 
                                    eta, Topt, vcmax25, jmax25, fd, q10, rd_func = None ) * 1.0e6


    vmin = df['Tleaf'].min()
    vmax = df['Tleaf'].max()
    axs[0,i].scatter(A, A_an_gs, c = Tl, cmap = 'coolwarm', vmin = vmin, vmax = vmax )
    axs[0,i].axline((A.mean(),A.mean()), slope = 1, ls = '--', c = 'grey')
    axs[0,i].set_xlabel('Observed A ($\mu$mol m$^{-2}$ s$^{-1}$)', size = 18)
    axs[0,i].set_ylabel('Fitted A ($\mu$mol m$^{-2}$ s$^{-1}$)', size = 18)
    axs[0,i].set_title( species, size = 18)
    
    # =============================================================================
    # Fit stomatal conductance
    # =============================================================================
    # Non-linear least squares
    idx = Tl <= 45
    f_gcrit_LTO = lambda Tair, beta: numerical_solve_LTO( nl, pc, Ta, Ca, Pa, Oa, Is, ra, vpd = VPD_leaf, 
                                                          Ipar = PAR, rb_co2_mol = rb_co2_mol, vpd_air = VPD_air, beta = beta )[1]
    
    f_gcrit_LT = lambda Tair, beta: numerical_solve_LT( nl, pc, Ta[idx], Ca[idx].values, Pa, Oa[idx], Is, ra, 
                                                        vpd = VPD_leaf[idx].values, Ipar = PAR, 
                                                        rb_co2_mol = rb_co2_mol[idx].values, vpd_air = VPD_air[idx].values, 
                                                        beta = beta )[1]


    if i == 1:
        rp_p0_LTO = 0.0001
    else:
        rp_p0_LTO = 0.0001
    popt, pcov = curve_fit( f_gcrit_LTO, Ta, gs_co2_mol, p0 = rp_p0_LTO, bounds = [0,0.5], diff_step = 1.0e-5 )
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
    save_params( params_path, 'Slot16_%s_params.csv'%(species),
                 Ea = Ea, Ea_SE = Ea_SE, eta = eta, eta_SE = eta_SE, Topt = Topt, Topt_SE = Topt_SE, 
                 vcmax25 = vcmax25, vcmax25_SE = vcmax25_SE, jmax25 = jmax25, jmax25_SE = jmax25_SE,
                 fd = fd, fd_SE = fd_SE, q10 = q10, q10_SE = q10_SE, beta_LT = beta_LT, 
                 beta_LT_SE = beta_LT_SE, beta_LTO = beta_LTO, beta_LTO_SE = beta_LTO_SE, ra = ra, 
                 ra_SE = ra_SE, Is = Is, Is_SE = Is_SE )
    out_data_path = 'Modelling_results/'
    save_out_data( out_data_path, 'Slot16_%s_out.csv'%(species),
                   A_obs = A.values, gs_obs = gs_co2_mol.values, ci_obs = Ci.values, Tl_obs = Tl.values,
                   Ta_obs = Ta.values, dT_obs = dT.values, Pa_obs = Pa, Ca_obs = Ca.values,
                   VPD_obs = VPD_leaf.values,
                   A_LT = A_LT, gs_LT = gs_LT, ci_LT = ci_LT, Tl_LT = Tl_LT, 
                   dT_LT = Tl_LT - Ta.values,
                   A_LTO = A_LTO, gs_LTO = gs_LTO, ci_LTO = ci_LTO, Tl_LTO = Tl_LTO, 
                   dT_LTO = Tl_LTO - Ta.values,
                   A_fit = A_an_gs.values )
    
cbar_labels = ['T$_{leaf}$ ($^o$C)', 'VPD$_{air}$ (Pa)']
for i,row in enumerate(axs[0:2,:]):
    sc = row[0].collections[0]
    bboxes = [ax.get_position() for ax in row]
    bbox = Bbox.union(bboxes)
    cax = fig.add_axes([bbox.x0 + bbox.width * 0.15, bbox.y0 - 0.07, bbox.width * 0.7, 0.02])
    # cax.set_xlabel()
    cbar = fig.colorbar(sc, cax = cax, orientation = 'horizontal')
    cbar.set_label( label = cbar_labels[i], size = 18 )

h,l = axs[-1,-1].get_legend_handles_labels()
fig.legend(h,l, loc = 'upper center', bbox_to_anchor = [0.5,0.09], ncol = 3, prop={'size': 18}  )
