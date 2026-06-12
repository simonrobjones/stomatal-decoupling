# -*- coding: utf-8 -*-
"""
Code to fit PGEN model to data from Feng et al (2023)
https://doi.org/10.1016/j.envexpbot.2023.105295
@author: srgj201
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PGEN_functions_complete import namelist, physical_constants
from PGEN_functions_complete import numerical_solve_LTO, numerical_solve_LT
from PGEN_functions_complete import calc_A_from_gs_C4, calc_esat_from_T
from PGEN_functions_complete import calc_Tleaf, save_params, save_out_data
from scipy.optimize import curve_fit
from matplotlib.transforms import Bbox

# =============================================================================
# Load in data
# =============================================================================
path           = ''
filename       = 'Feng_23_raw.csv'
df             = pd.read_csv( path + filename )
unique_species = df['species'].unique()

# =============================================================================
# Set up namelists
# =============================================================================
nl = namelist()
pc = physical_constants()

# =============================================================================
# Fit photosynthesis, energy balance and stomtatal conductance for each species
# =============================================================================
# Set up figure
ncols = 4
nrows = 3
fig,axs = plt.subplots(ncols = ncols, nrows = nrows, figsize = (7*ncols, 6*nrows))#, sharey = 'row')
if ncols == 1:
    axs = axs[:,np.newaxis]
if nrows == 1:
    axs = axs[np.newaxis,:]
fig.subplots_adjust( wspace=0.2, hspace= 1.0 )

for i,species in enumerate(unique_species):
    #Extract data
    df_species = df[ df['species'] == species ]
    # Each species has a wet and droughted treatment
    treatments = df_species['Treat'].unique()
    
    # Loop through treatments seperately
    for j,treat in enumerate(treatments):
        # Extract data
        df_treat = df_species[ df_species['Treat'] == treat ][['tair','tleaf','vpdleaf','Ca',
                                                           'Ci','gsw','gbw','gtw','A','E']]
    
        df_treat[ df_treat['Ci'] < 0 ] = np.nan
        df_treat = df_treat.dropna()
        A          = df_treat['A']
        gs_h2o_mol = df_treat['gsw']
        Ta         = df_treat['tair']
        VPD_leaf   = df_treat['vpdleaf'] * 1.0e3
        Ci_ppm     = df_treat['Ci']
        Ca_ppm     = df_treat['Ca']
        Tl         = df_treat['tleaf']
        Pa         = 101325.0
        Ci         = Ci_ppm * Pa / 1.0e6
        Ca         = Ca_ppm * Pa / 1.0e6
        gt_h2o_mol = df_treat['gtw']
        gt_co2_mol = gt_h2o_mol / 1.6
        gbw_mol    = df_treat['gbw']
        gb_co2_mol = gbw_mol / 1.6
        rb_co2_mol = 1.0 / gb_co2_mol
        gbw_m_s    = gbw_mol * 8.314462 * ( Ta + 273.15 ) / Pa
        gs_h2o_m_s = gs_h2o_mol * 8.314462 * ( Ta + 273.15 ) / Pa
        gt_h2o_m_s = gt_h2o_mol * 8.314462 * ( Ta + 273.15 ) / Pa
        ra         = 1.0 / gbw_m_s
        vp_air     = calc_esat_from_T( Tl ) - VPD_leaf
        VPD_air    = calc_esat_from_T( Ta ) - vp_air
        gs_co2_mol = gs_h2o_mol  / 1.6
        dT         = Tl - Ta
        PAR        = 1500e-6
        Is         = 340

        # =============================================================================
        # Fit Photosynthesis
        # =============================================================================
        f = lambda gs, Ea, eta, Topt, vcmax25, fd, q10 : calc_A_from_gs_C4(pc, nl, Tl, gs, Ca, Pa, PAR, Ea * 1e3, eta, Topt, vcmax25, fd, q10 ) * 1e6
        popt, pcov = curve_fit( f, gt_co2_mol, A, p0 = [50, 4, 30, 1e-4, 1e-3, 2],
                                bounds = ( [1,1,20,1e-15, 0.0, 1.0], [200,50,50,1e-3, 0.1, 3.0 ] ) )  
        
        print(popt)
        Ea, eta, Topt, vcmax25, fd, q10 = popt   
        Ea_SE, eta_SE, Topt_SE, vcmax25_SE, fd_SE, q10_SE = np.sqrt( np.diag( pcov ) )
        
        nl.Ea      = Ea * 1e3
        nl.eta     = eta
        nl.Topt    = Topt
        nl.vcmax25 = vcmax25 
        nl.fd      = fd
        nl.q10     = q10
        
        print("Ea = %s \u00B1 %s"%(Ea, Ea_SE))
        print("eta = %s \u00B1 %s"%( eta, eta_SE))
        print("Topt = %s \u00B1 %s"%( Topt, Topt_SE))
        print("vcmax25 = %s \u00B1 %s"%( vcmax25, vcmax25_SE))
        print("fd = %s \u00B1 %s"%( fd, fd_SE))
        print("q10 = %s \u00B1 %s"%( q10, q10_SE))
        
        A_an_gs    = calc_A_from_gs_C4( pc, nl, Tl, gt_co2_mol, Ca, Pa, PAR, Ea * 1e3, eta, Topt, vcmax25, fd, q10 ) * 1e6
        
        vmin = df['tleaf'].min()
        vmax = df['tleaf'].max()
        axs[0,i*2+j].scatter(A, A_an_gs, c = Tl, cmap = 'coolwarm', vmin = vmin, vmax = vmax )
        axs[0,i*2+j].axline((A.mean(),A.mean()), slope = 1, ls = '--', c = 'grey')
        axs[0,i*2+j].set_xlabel('Observed A ($\mu$mol m$^{-2}$ s$^{-1}$)', size = 18)
        axs[0,i*2+j].set_ylabel('Fitted A ($\mu$mol m$^{-2}$ s$^{-1}$)', size = 18)
        axs[0,i*2+j].set_title( treat, size = 18)
        

        
        # =============================================================================
        # Plot energy balance equation
        # =============================================================================
        dT_an = calc_Tleaf( pc, gs_h2o_m_s, Ta, Is, Pa, ra, VPD_air ) - Ta
        
        VPD_leaf_all   = df['vpdleaf'] * 1.0e3
        vp_air_all     = calc_esat_from_T( df['tleaf'] ) - VPD_leaf_all
        VPD_air_all    = calc_esat_from_T( df['tair'] ) - vp_air_all
        vmin = VPD_air_all.min()
        vmax = VPD_air_all.max()
        axs[1,i*2+j].scatter( dT, dT_an, c = VPD_air, cmap = 'coolwarm', vmin = vmin, vmax = vmax )
        axs[1,i*2+j].axline( (dT.mean(),dT.mean()), slope = 1, ls = '--', c = 'grey' )
        axs[1,i*2+j].set_xlabel('Observed dT ($^o$C)', size = 18)
        axs[1,i*2+j].set_ylabel('Predicted dT ($^o$C)', size = 18)
        axs[1,i*2+j].set_title(species, size = 18)
        
        # =============================================================================
        # Fit Stomatal conductance
        # =============================================================================
        
        

        
        f_gcrit_LTO = lambda Tair, beta: numerical_solve_LTO( nl, pc, Ta = Tair, ca = Ca, pa = Pa, oa = None, 
                                                              Iabs = Is, ra = ra, vpd = VPD_leaf,
                                                              Ipar = PAR, rb_co2_mol = rb_co2_mol,
                                                              vpd_air = VPD_air.values, beta = beta, C4 = True )[1]
        
        f_gcrit_LT = lambda Tair, beta: numerical_solve_LT( nl, pc, Ta = Tair, ca = Ca, pa = Pa, oa = None, 
                                                            Iabs = Is, ra = ra, vpd = VPD_leaf,
                                                            Ipar = PAR, rb_co2_mol = rb_co2_mol,
                                                            vpd_air = VPD_air.values, beta = beta, C4 = True )[1]
        
        popt, pcov = curve_fit( f_gcrit_LTO, Ta.values, gs_co2_mol.values, p0 = 0.001, bounds = [0.0,0.1] )
        print(popt)
        beta_LTO = popt[0]
        beta_LTO_SE = np.sqrt( np.diag( pcov ) )
        
        popt, pcov = curve_fit( f_gcrit_LT, Ta.values, gs_co2_mol.values, p0 = 0.001, bounds = [0.0,0.1] )
        print(popt)
        beta_LT     = popt[0]
        beta_LT_SE = np.sqrt( np.diag( pcov ) )
       
        #=============================================================================
        # Calculate predicted gas exchange
        # =============================================================================
        A_LT, gs_LT, ci_LT, Tl_LT = numerical_solve_LT(  nl, pc, Ta = Ta, ca = Ca, pa = Pa, oa = None, 
                                                         Iabs = Is, ra = ra, vpd = VPD_leaf,
                                                         Ipar = PAR, rb_co2_mol = rb_co2_mol,
                                                         vpd_air = VPD_air.values, beta = beta_LT, C4 = True)
        
        A_LTO, gs_LTO, ci_LTO, Tl_LTO = numerical_solve_LTO(  nl, pc, Ta = Ta, ca = Ca, pa = Pa, oa = None, 
                                                              Iabs = Is, ra = ra, vpd = VPD_leaf,
                                                              Ipar = PAR, rb_co2_mol = rb_co2_mol,
                                                              vpd_air = VPD_air.values, beta = beta_LTO, C4 = True)
    
        axs[2,i*2+j].scatter(Ta, gs_co2_mol, color = 'black', facecolor = 'none', label = 'obs')
        axs[2,i*2+j].plot(Ta.iloc[np.argsort(Ta)], gs_LT[np.argsort(Ta)], color = '#DC267F', label = 'LT')
        axs[2,i*2+j].plot(Ta.iloc[np.argsort(Ta)], gs_LTO[np.argsort(Ta)], color = '#FFB000', label = 'LTO')
        axs[2,i*2+j].set_xlabel('Ta ($^o$C)', size = 18)
        axs[2,i*2+j].set_ylabel('g$_{sc}$ ($\mu$mol m$^{-2}$ s$^{-1}$)', size = 18)
        axs[2,i*2+j].set_title(species, size = 18 )
       
        # =============================================================================
        # Save fitted parameter values + SEs
        # Save predicted and observed gas exchange
        # =============================================================================
        params_path = 'parameter_vals/'
        save_params( params_path, 'Feng23_%s_params.csv'%(treat),
                     Ea = Ea, Ea_SE = Ea_SE, eta = eta, eta_SE = eta_SE, Topt = Topt, Topt_SE = Topt_SE, 
                     vcmax25 = vcmax25, vcmax25_SE = vcmax25_SE, jmax25 = None, jmax25_SE = None,
                     fd = fd, fd_SE = fd_SE, q10 = q10, q10_SE = q10_SE, beta_LT = beta_LT, 
                     beta_LT_SE = beta_LT_SE, beta_LTO = beta_LTO, beta_LTO_SE = beta_LTO_SE, ra = None, 
                     ra_SE = None, Is = None, Is_SE = None )
        out_data_path = 'Modelling_results/'
        save_out_data( out_data_path, 'Feng23_%s_out.csv'%(treat),
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
    
