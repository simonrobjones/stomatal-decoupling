# -*- coding: utf-8 -*-
"""
Code to fit PGEN to data for Poplar from Urban et al (2017)
https://doi.org/10.1093/jxb/erx052
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
# Load in data. Data available on request at https://doi.org/10.1093/jxb/erx052
# =============================================================================
path     = ''
filename = 'Poplar_BM.xlsx'

df_wet     = pd.read_excel(path+filename, sheet_name = 'wet-ambient', header = 11, skiprows = [12])
df_dry     = pd.read_excel(path+filename, sheet_name = 'dry', header = 11, skiprows = [12])
df_wet_co2 = pd.read_excel(path+filename, sheet_name = 'wet-high CO2', header = 11, skiprows = [12])
df_all     = pd.concat( [ df_wet, df_dry, df_wet_co2 ], ignore_index = True )

# =============================================================================
# Set up namelists
# =============================================================================
nl = namelist()
pc = physical_constants()

# =============================================================================
# Fit photosynthetic parameters
# Note here we fit parameters to all data
# =============================================================================
# Extract data
df             = df_all.copy()
A_all          = df['Photo']
Pa_all         = df['Press'] * 1e3
Tl_all         = df['Tleaf']
Ta_all         = df['Tair']
ca_ppm_all     = df['CO2S']
Ca_all         = ca_ppm_all * Pa_all * 1.0e-6
PAR_all        = (df['PARi'] - df['PARo']) * 1e-6
gt_co2_mol_all = df['CndCO2']
Oa_pc_all      = 20.0
Oa_all         = Oa_pc_all * Pa_all / 100.0
VP_air_all     = df['vp_kPa'] * 1e3
SVP_air_all    = calc_esat_from_T( Ta_all )
VPD_air_all    = SVP_air_all - VP_air_all
SVP_leaf_all   = calc_esat_from_T( Tl_all )
VPD_leaf_all   = SVP_leaf_all - VP_air_all 

# Non-linear least squares
f = lambda gt_co2_mol, Ea, eta, Topt, vcmax25, jmax25, fd, q10, : calc_A_from_gs_C3( pc, nl, Tl_all, gt_co2_mol, Ca_all, Pa_all, PAR_all, Oa_all, Ea * 1.0e3, eta, Topt, vcmax25, jmax25, fd, q10, rd_func = None ) * 1.0e6
popt, pcov = curve_fit( f, gt_co2_mol_all, A_all, p0 = [2.06847911e+01, 1.37935891e+01, 3.72933730e+01, 9.72280027e-04, 2.02649876e-04, 6.24972076e-03, 1.50000232e+00],
                        bounds = ([1,1,1,1e-15,1e-15, 0.0, 1.0],[200,100,50,1,1,0.1,5.0]))    

Ea, eta, Topt, vcmax25, jmax25, fd, q10 = popt   
Ea_SE, eta_SE, Topt_SE, vcmax25_SE, jmax25_SE, fd_SE, q10_SE = np.sqrt( np.diag( pcov ) )


print("Ea = %s \u00B1 %s"%(Ea, Ea_SE))
print("eta = %s \u00B1 %s"%( eta, eta_SE))
print("Topt = %s \u00B1 %s"%( Topt, Topt_SE))
print("vcmax25 = %s \u00B1 %s"%( vcmax25, vcmax25_SE))
print("jmax25 = %s \u00B1 %s"%( jmax25, jmax25_SE))
print("fd = %s \u00B1 %s"%( fd, fd_SE))
print("q10 = %s \u00B1 %s"%( q10, q10_SE))

# Store results in namelist
nl.Ea      = Ea * 1e3
nl.eta     = eta
nl.Topt    = Topt
nl.vcmax25 = vcmax25 
nl.jmax25  = jmax25
nl.fd      = fd
nl.q10     = q10

# Plot obs vs fitted photosynthesis
A_an_gs    = calc_A_from_gs_C3( pc, nl, Tl_all, gt_co2_mol_all, Ca_all, Pa_all, PAR_all, 
                                Oa_all, Ea * 1.0e3, eta, Topt, vcmax25, jmax25, fd, 
                                q10, rd_func = None ) * 1.0e6
plt.figure()
plt.scatter(A_all, A_an_gs, c = Tl_all, cmap = 'coolwarm' ) 
plt.axline((A_all.mean(),A_all.mean()),slope = 1)

# =============================================================================
# Fit energy balance and stomtatal conductance for each experiment
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



experiment_labels = ['Wet', 'Dry', 'Wet elevated CO2']
save_labels       = ['wet', 'dry', 'wet_co2']
# Loop through each experiment    
for i, df_main in enumerate( [ df_wet, df_dry, df_wet_co2 ] ):
    # Extract data
    df         = df_main.copy()
    Tl         = df['Tleaf']
    Ta         = df['Tair']
    dT         = Tl - Ta
    A          = df['Photo']
    Pa         = df['Press'] * 1e3
    gs_h2o_mol = df['Cond']
    gs_co2_mol = gs_h2o_mol / 1.6
    gt_co2_mol = df['CndCO2']
    gs_h2o_m_s = gs_h2o_mol * 8.314462 * ( Ta + 273.15 ) / Pa
    Ci         = df['Ci_Pa'] 
    
    ca_ppm     = df['CO2S']
    Ca         = ca_ppm * Pa * 1.0e-6
    Oa_pc      = 20.0
    Oa         = Oa_pc * Pa / 100.0
    PAR        = df['PARi'] - df['PARo']
    VP_air     = df['vp_kPa'] * 1e3
    SVP_air    = calc_esat_from_T( Ta )
    VPD_air    = SVP_air - VP_air
    SVP_leaf   = calc_esat_from_T( Tl )
    VPD_leaf   = SVP_leaf - VP_air  
    
    # Plot photosynthesis (fitted vs obs) for each experiment
    A_an_gs    = calc_A_from_gs_C3( pc, nl, Tl, gt_co2_mol, Ca, Pa, PAR, Oa, 
                                    Ea * 1.0e3, eta, Topt, vcmax25, jmax25, 
                                    fd, q10, rd_func = None ) * 1.0e6
    vmin = Tl_all.min()
    vmax = Tl_all.max()
    axs[0,i].scatter(A, A_an_gs, c = Tl, cmap = 'coolwarm', vmin = vmin, vmax = vmax ) 
    axs[0,i].axline((A.mean(),A.mean()), slope = 1, ls = '--', c = 'grey')
    axs[0,i].set_xlabel('Observed A ($\mu$mol m$^{-2}$ s$^{-1}$)', size = 18)
    axs[0,i].set_ylabel('Fitted A ($\mu$mol m$^{-2}$ s$^{-1}$)', size = 18)
    axs[0,i].set_title(experiment_labels[i], size = 18)


    # =============================================================================
    # Determine ra and Is
    # =============================================================================
    f = lambda gs, Is, ra: calc_Tleaf( pc, gs, Ta, Is, Pa, ra, VPD_air ) - Ta

    popt, pcov = curve_fit( f, gs_h2o_m_s, dT, p0 = [300,10], bounds = ([0,0.1],[1000,50]))
    Is, ra = popt
    Is_SE, ra_SE = np.sqrt( np.diag(pcov) )

    
    gb_h2o_m_s = 1.0 / ra
    gb_h2o_mol = gb_h2o_m_s * Pa / ( 8.314462 * ( Ta + 273.15 ) )
    gb_co2_mol = gb_h2o_mol / 1.6
    rb_co2_mol = 1.0 / gb_co2_mol
    dT_an      = calc_Tleaf( pc, gs_h2o_m_s, Ta, Is, Pa, ra, VPD_air ) - Ta
    
    # Plot fitted energy balance
    vmin = VPD_air_all.min()
    vmax = VPD_air_all.max()
    axs[1,i].scatter( dT, dT_an, c = VPD_air, cmap = 'coolwarm', vmin = vmin, vmax = vmax )
    axs[1,i].axline( (dT.mean(),dT.mean()), slope = 1, ls = '--', c = 'grey' )
    axs[1,i].set_xlabel('Observed dT ($^o$C)', size = 18)
    axs[1,i].set_ylabel('Fitted dT ($^o$C)', size = 18)
    axs[1,i].set_title(experiment_labels[i], size = 18)
    
    # =============================================================================
    # Fit stomatal conductance
    # =============================================================================
    # Non-linear least squares
    idx = Tl <= 45
    f_gcrit_LTO = lambda Tair, beta: numerical_solve_LTO( nl, pc, Ta, Ca, Pa, Oa, Is, ra, vpd = VPD_leaf, 
                                                          Ipar = PAR, rb_co2_mol = rb_co2_mol, vpd_air = VPD_air, beta = beta )[1]
    
    f_gcrit_LT = lambda Tair, beta: numerical_solve_LT( nl, pc, Ta[idx], Ca[idx].values, Pa[idx], Oa[idx], Is, ra, vpd = VPD_leaf[idx].values, 
                                                        Ipar = PAR[idx].values, rb_co2_mol = rb_co2_mol[idx].values, vpd_air = VPD_air[idx].values, beta = beta )[1]


    popt, pcov = curve_fit( f_gcrit_LTO, Ta, gs_co2_mol, p0 = 0.01, bounds = [0,0.5], diff_step = 1.0e-5 )
    print(popt)
    beta_LTO = popt[0]
    beta_LTO_SE = np.sqrt( np.diag( pcov ) )
    
    popt, pcov = curve_fit( f_gcrit_LT, Ta[idx].values, gs_co2_mol[idx].values, p0 = 0.01, bounds = [0,0.5], diff_step = 1.0e-5 )
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
    axs[2,i].set_title(experiment_labels[i], size = 18 )
    
    # =============================================================================
    # Save fitted parameter values + SEs
    # Save predicted and observed gas exchange
    # =============================================================================
    params_path = 'parameter_vals/'
    save_params( params_path, 'Urban17_poplar_%s_params.csv'%(save_labels[i]),
                 Ea = Ea, Ea_SE = Ea_SE, eta = eta, eta_SE = eta_SE, Topt = Topt, Topt_SE = Topt_SE, 
                 vcmax25 = vcmax25, vcmax25_SE = vcmax25_SE, jmax25 = jmax25, jmax25_SE = jmax25_SE,
                 fd = fd, fd_SE = fd_SE, q10 = q10, q10_SE = q10_SE, beta_LT = beta_LT, 
                 beta_LT_SE = beta_LT_SE, beta_LTO = beta_LTO, beta_LTO_SE = beta_LTO_SE, ra = ra, 
                 ra_SE = ra_SE, Is = Is, Is_SE = Is_SE )
    out_data_path = 'Modelling_results/'
    save_out_data( out_data_path, 'Urban17_poplar_%s_out.csv'%(save_labels[i]),
                   A_obs = A.values, gs_obs = gs_co2_mol.values, ci_obs = Ci.values, Tl_obs = Tl.values,
                   Ta_obs = Ta.values, dT_obs = dT.values, Pa_obs = Pa.values, Ca_obs = Ca.values,
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

