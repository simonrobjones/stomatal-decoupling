# -*- coding: utf-8 -*-
"""
Code to produce figure 4 of Jones et al

This code also produces figures S4 & S5

@author: srgj201
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from Figure_4_functions import numerical_solve_Diao_LTO, numerical_solve_Diao_LT, numerical_solve_Diao_AT
from Figure_4_functions import calc_rd_Diao, calc_J_Diao
from Figure_4_functions import bootstrap_loess
from PGEN_functions import calc_photosynthetic_params, physical_constants, calc_Tleaf
from sklearn.metrics import mean_squared_error as mse

# The LOESS fitting is computationally heavy so the results are provided, however they
# can be re-calculated by setting the following switch to True
re_calc_loess = False

# =============================================================================
# Read in the observed data. This is available from Diao et al (2024)
# (https://doi.org/10.1111/nph.19558)
# =============================================================================
file_name = 'nph19558-sup-0002-tables2.xlsx'
file_path = 'C:/Users/srgj201/OneDrive - University of Exeter/Documents/Postdoc/SOX/High_T_decoupling/'
df        = pd.read_excel(file_path+file_name, sheet_name = 'Table S2')

# Create list of plant species to use
plant_species = ['Fagus','Tilia','Quercus']

# Create a list of all the unique plant codes in the dataset
plant_codes = np.unique(df['Plant code'])


# =============================================================================
# Set a few environmental conditions
# =============================================================================
pa    = 101325.0   # Atmospheric air pressure                                                                               (Pa)
x_co2 = 0.0004     # The CO2 concentration in the cuvette was held constant (Diao et al, 2024  - see measurement protocol ) (mol/mol)
ca    = pa * x_co2 # The resulting partial pressure of CO2 in the cuvette                                                   (Pa)
D     = 800.0      # Vapour pressure deficit was held constant to approximately 800Pa (Diao et al, 2024)                    (Pa)
oa    = 20900.0    # Partial pressure of O2. Not measured but we assume is equal to standard atmospheric concentrations     (Pa)

# Create an instance of the physical constants class and namelist class with model parameters
pc = physical_constants()

# Set pc.dTa_s so that the apparent radiative temperature of the cuvette temperature is equal to the cuvette temperature
pc.dTa_s = 0.0


# =============================================================================
# Set up Figure 4
# =============================================================================
model_labels = ['Leaf Temperature within Optimisation (LTO)', 'Leaf Temperature (LT)', 'Air Temperature (AT)']
model_colors = {'Leaf Temperature within Optimisation (LTO)':'#FFB000',
                      'Leaf Temperature (LT)':'#DC267F',
                      'Air Temperature (AT)':'#648FFF'}

# Create custom legend
custom_lines_model  = [Line2D([0],[0],color = model_colors[s]) for s in model_labels]
legend_labels       = model_labels + [ 'Observations', 'Local polynomial regression' ]
custom_lines_legend = custom_lines_model + [Line2D([], [], color = 'black',marker = 'o',markerfacecolor = 'none',ls = 'none'),
                                            (Patch(facecolor = 'black',alpha = 0.2),Line2D([0], [0], color = 'black',lw = 1.0))]

# Initialise figure
fig_4, axs_4 = plt.subplots( ncols = 3, nrows = 3, figsize = ( 18, 12 ) )
plt.subplots_adjust( wspace = 0.25, hspace = 0.25 )

# Add legend
fig_4.legend(custom_lines_legend,legend_labels,loc = 'upper center',bbox_to_anchor = (0.5,0.05), fontsize = 12, ncol = 5 )

# Label axes
for i, row in enumerate(axs_4):
    for j, ax in enumerate(row):
        if i == 0:
            ax.set_title(plant_species[j],size = 15)
        if i == 2:
            ax.set_xlabel('T$_{cuv}$ ($\degree$C)',size = 15)
        if (i == 0) & (j==0):
            ax.set_ylabel('A ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)',size = 15)
        if (i == 1) & (j==0):
            ax.set_ylabel('g$_{sv}$ (mol H$_2$O m$^{-2}$ s$^{-1}$)',size = 15)
        if (i == 2) & (j==0):
            ax.set_ylabel('T$_{leaf}$ - T$_{cuv}$ ($\degree$C)',size = 15)

# =============================================================================
# Set up Figure S4
# =============================================================================
ncols = 3
nrows = 1
fig_S4, axs_S4 = plt.subplots(ncols = ncols, nrows = nrows,figsize = (6*ncols,4*nrows))
axs_S4         = axs_S4.reshape(-1)

# Axis labels
for i,ax in enumerate(axs_S4):
    ax.set_title(plant_species[i], size = 15)
    ax.set_xlabel('gs (mol H$_2$O m$^{-2}$ s$^{-1}$)',size = 15)
    if i == 0:
        ax.set_ylabel('T$_{leaf} - T_{cuv}$ ($\degree$C)', size = 15)        

# Custom legend
custom_lines_S4  = [Line2D([],[],color = 'red',   marker = 'o', ls = 'none') ,
                    Line2D([],[],color = 'black', marker = 'o', ls = 'none', markerfacecolor = 'none')
                    ]
legend_labels_S4 = ['Non-linear least squares fit','Observations']
fig_S4.legend( custom_lines_S4, legend_labels_S4, loc = 'upper center', bbox_to_anchor = (0.5,-0.03), fontsize = 12, ncol = 4)

# =============================================================================
# Set up Figure S5
# =============================================================================
# Initialise figure
fig_S5, axs_S5 = plt.subplots( ncols = 3, nrows = 3, figsize = ( 18, 12 ) )
plt.subplots_adjust( wspace = 0.25, hspace = 0.25 )

# Add legend (we can use the same lines and labels as Fig 4)
fig_S5.legend(custom_lines_legend,legend_labels,loc = 'upper center',bbox_to_anchor = (0.5,0.05), fontsize = 12, ncol = 5 )

# Label axes
for i, row in enumerate(axs_S5):
    for j, ax in enumerate(row):
        if i == 0:
            ax.set_title(plant_species[j],size = 15)
        if i == 2:
            ax.set_xlabel('T$_{cuv}$ ($\degree$C)',size = 15)
        if (i == 0) & (j==0):
            ax.set_ylabel('A ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)',size = 15)
        if (i == 1) & (j==0):
            ax.set_ylabel('g$_{sv}$ (mol H$_2$O m$^{-2}$ s$^{-1}$)',size = 15)
        if (i == 2) & (j==0):
            ax.set_ylabel('T$_{leaf}$ - T$_{cuv}$ ($\degree$C)',size = 15)

# =============================================================================
# Set up a dictionary to store the normalised root mean square error (NRMSE)
# values
# =============================================================================
NRMSE_values = {'LTO':{'Fagus':{'A':np.nan,'gs':np.nan,'dT':np.nan}, 'Tilia':{'A':np.nan,'gs':np.nan,'dT':np.nan}, 'Quercus':{'A':np.nan,'gs':np.nan,'dT':np.nan}},
                'LT':{'Fagus':{'A':np.nan,'gs':np.nan,'dT':np.nan}, 'Tilia':{'A':np.nan,'gs':np.nan,'dT':np.nan}, 'Quercus':{'A':np.nan,'gs':np.nan,'dT':np.nan}},
                'AT':{'Fagus':{'A':np.nan,'gs':np.nan,'dT':np.nan}, 'Tilia':{'A':np.nan,'gs':np.nan,'dT':np.nan}, 'Quercus':{'A':np.nan,'gs':np.nan,'dT':np.nan}}}

# =============================================================================
# For each species we will:
# - Plot observed A, gs(H2O), dT (Tleaf - Tcuv).
#
# - Fit and plot LOESS model to the observations.
#
# - Fit the temperature sensitivity equation of electron transport rate (J)
#   to the observed data
#
# - Fit the leaf energy balance equation to the observed 
#   relationship of dT on gs. This will determine Is and ra.
#
# - Fit the LTO, LT, and AT models to the observed data. This will determine
#   the gcrit paramater.
#
# - Plot model fits.
#
# - Calculate Normalised Root Mean Square Error (NRMSE) values between 
#   model predictions and observation
# =============================================================================

for i, species in enumerate(plant_species):
    print(species)
    # =============================================================================
    # Plot observed data. Determine and plot loess trend.     
    # =============================================================================
    # Create a list of all the unique plant codes in the dataset of this species
    species_codes = plant_codes[[species in code for code in plant_codes]]
    
    # Extract the species data from the observations
    species_df = df[df['Plant code'].isin(species_codes)]
    
    # Extract the observed data
    T_cuv      = species_df['Tcuv (°C)']             # Cuvette Temperature  (C)
    T_leaf     = species_df['Tleaf (°C)']            # Leaf Temperature     (C)
    A          = species_df['An (μmol CO2 m-2 s-1)'] # Leaf photosynthesis  (micro mol CO2/m2/s)
    gs_h2o_mol = species_df['gs (mmol CO2 m-2 s-1)'] # Stomatal conductance ( mol H20/m2/s ) # NOTE DESPITE DATA HEADER THIS IS ACTUALLY STOMATAL CONDUCTANCE TO H20 (mol H20 m-2 s-1) and is not in mili-mol untis
    chi        = species_df['ci/ca']                 # ci:ca ration         (-)
    ci         = chi * ca                            # The calculated internal CO2 partial pressure assuming ca calculated above
    dT         = T_leaf - T_cuv                      # The difference in leaf and cuvette temperature (C)
    gs_h2o_m_s = gs_h2o_mol * 8.314462 * ( T_cuv + 273.15 ) / pa # Stomatal conductance to water vapout (m/s)

    # Create a space of cuvette temperatures to input into the models
    T_cuv_x   = np.linspace( np.min( T_cuv ), np.max( T_cuv ) )
    T_leaf_x  = np.linspace(np.min(T_leaf),np.max(T_leaf))
    
    # For the loess fit we must remove the missing data so create indicies for where there are NaN values
    A_is_nan_idx   = A.isna()
    gs_is_nan_idx  = gs_h2o_mol.isna()
    ci_is_nan_idx  = ci.isna()
    chi_is_nan_idx = chi.isna()
    dT_is_nan_idx  = dT.isna()
    T_cuv_A        = T_cuv[~A_is_nan_idx]
    T_leaf_A       = T_leaf[~A_is_nan_idx]
    T_cuv_gs       = T_cuv[~gs_is_nan_idx]
    T_cuv_ci       = T_cuv[~ci_is_nan_idx]
    
    # Plot the data for both Fig 4 and Fig S5
    axs_4[0,i].plot(T_cuv, A, color = 'black', marker = 'o', linestyle = 'none',markerfacecolor = 'none', alpha = 1.0)
    axs_4[1,i].plot(T_cuv, gs_h2o_mol, color = 'black', marker = 'o', linestyle = 'none', markerfacecolor = 'none', alpha = 1.0 )
    axs_4[2,i].plot(T_cuv, dT, color = 'black', marker = 'o', linestyle = 'none',markerfacecolor = 'none')
    
    axs_S5[0,i].plot(T_cuv, A, color = 'black', marker = 'o', linestyle = 'none',markerfacecolor = 'none', alpha = 1.0)
    axs_S5[1,i].plot(T_cuv, gs_h2o_mol, color = 'black', marker = 'o', linestyle = 'none', markerfacecolor = 'none', alpha = 1.0 )
    axs_S5[2,i].plot(T_cuv, dT, color = 'black', marker = 'o', linestyle = 'none',markerfacecolor = 'none')
    
    # =============================================================================
    # Fit a LOESS smoother to the observations with a boot-strapping approach.
    # This will give an ucertainty bound as well
    # =============================================================================
    # The loess requires sorted data without missing data so we produce indicies to
    # get data in the correct format:
    idx_lpr_A  = ~A_is_nan_idx
    sort_idx_A = np.argsort(T_cuv[idx_lpr_A].values)
    
    idx_lpr_gs  = ~gs_is_nan_idx
    sort_idx_gs = np.argsort(T_cuv[idx_lpr_gs].values)

    idx_lpr_X   = ~chi_is_nan_idx
    sort_idx_X  = np.argsort(T_cuv[idx_lpr_X].values)
    
    idx_lpr_dT  = ~dT_is_nan_idx
    sort_idx_dT = np.argsort(T_cuv[idx_lpr_dT].values)
    
    # The LOESS is computationally heavy so the results are provided as files. 
    # However the results can be re-calculated by setting re_calc_loess = True
    if re_calc_loess:
        np.random.seed(52)
        # A
        x_fit_A, y_fit_A, y_err_A = bootstrap_loess( T_cuv[idx_lpr_A].values[sort_idx_A], A[idx_lpr_A].values[sort_idx_A], frac = 0.8, n_boot = 1000, degree = 2)
        np.save('LOESS/loess_boot_strap_A_cuv_%s_Jones.npy'%(species), ( x_fit_A, y_fit_A, y_err_A ) )
        
        # gs
        x_fit_g, y_fit_g, y_err_g = bootstrap_loess( T_cuv[idx_lpr_gs].values[sort_idx_gs], gs_h2o_mol[idx_lpr_gs].values[sort_idx_gs], frac = 0.8, n_boot = 1000, degree = 1)
        np.save('LOESS/loess_boot_strap_gs_cuv_%s_Jones.npy'%(species), ( x_fit_g, y_fit_g, y_err_g ) )
        
        # dT
        x_fit_dT, y_fit_dT, y_err_dT = bootstrap_loess( T_cuv[idx_lpr_dT].values[sort_idx_dT], dT[idx_lpr_dT].values[sort_idx_dT], frac = 0.8, n_boot = 1000, degree = 2)
        np.save('LOESS/loess_boot_strap_dT_cuv_%s_Jones.npy'%(species), ( x_fit_dT, y_fit_dT, y_err_dT ) )
        
    else:
        x_fit_A, y_fit_A, y_err_A       = np.load('LOESS/loess_boot_strap_A_cuv_%s_Jones.npy'%(species))
        x_fit_g, y_fit_g, y_err_g       = np.load('LOESS/loess_boot_strap_gs_cuv_%s_Jones.npy'%(species))
        x_fit_dT, y_fit_dT, y_err_dT    = np.load('LOESS/loess_boot_strap_dT_cuv_%s_Jones.npy'%(species))
    
    # =============================================================================
    # Plot the LOESS fit on both Fig 4 and Fig S5    
    # =============================================================================
    # Photosynthesis
    axs_4[0,i].plot(x_fit_A, y_fit_A, color = 'black' )
    axs_4[0,i].fill_between(x_fit_A, y_fit_A - 2 * y_err_A, y_fit_A + 2 * y_err_A, color = 'black', alpha = 0.2 )
    
    axs_S5[0,i].plot(x_fit_A, y_fit_A, color = 'black' )
    axs_S5[0,i].fill_between(x_fit_A, y_fit_A - 2 * y_err_A, y_fit_A + 2 * y_err_A, color = 'black', alpha = 0.2 )
    
    # Stomatal conductance
    axs_4[1,i].plot(x_fit_g, y_fit_g, color = 'black' )
    axs_4[1,i].fill_between(x_fit_g, y_fit_g - 2 * y_err_g, y_fit_g + 2 * y_err_g, color = 'black', alpha = 0.2 )
    
    axs_S5[1,i].plot(x_fit_g, y_fit_g, color = 'black' )
    axs_S5[1,i].fill_between(x_fit_g, y_fit_g - 2 * y_err_g, y_fit_g + 2 * y_err_g, color = 'black', alpha = 0.2 )
    
    # Delta T vs gs
    axs_4[2,i].plot(x_fit_dT, y_fit_dT, color = 'black' )
    axs_4[2,i].fill_between(x_fit_dT, y_fit_dT - 2 * y_err_dT, y_fit_dT + 2 * y_err_dT, color = 'black', alpha = 0.2 )
    
    axs_S5[2,i].plot(x_fit_dT, y_fit_dT, color = 'black' )
    axs_S5[2,i].fill_between(x_fit_dT, y_fit_dT - 2 * y_err_dT, y_fit_dT + 2 * y_err_dT, color = 'black', alpha = 0.2 )
            
    # =============================================================================
    # Fit the temperature sensitivity equation of electron transport rate (J)
    # to the observed data
    # =============================================================================
    p0      = [0.0, 45, 30, 24.0]              # Initial conditions for non-linear least squares ( Tmin, Tmax, Topt, Jmax )
    bounds  = ([-1000,30,20,0],[10,60,45,80])  # Bounds on parameter values for the fit
    
    # Calculate the CO2 compensation point in the abscence of respiration from observed leaf temperature.
    Kc, Ko, gamma_star = calc_photosynthetic_params( T_leaf, oa ) # We only need gamma_star here
    
    # Calculate the species specific dayrespiration (Rd) from observed leaf temperature
    rd = calc_rd_Diao( T_leaf, species )
    
    # Rearrange the RuBP limited rate equation to calculate the "observed" electron transport rate (J)
    J_obs = ( A + rd ) * ( ci + 2 * gamma_star ) / ( ci - gamma_star )
    
    # Fit the temperature sensitivity fT multiplied by a Jmax. We will remove missing data from A and ci
    idx          = ( ~A_is_nan_idx ) & ( ~ci_is_nan_idx )
    popt_J, pcov = curve_fit( calc_J_Diao, T_leaf[idx], J_obs[idx], p0 = p0, maxfev = 5000, bounds = bounds )
    
    J = calc_J_Diao( T_leaf_x, *popt_J )
    plt.figure()
    plt.plot( T_leaf, J_obs, color = 'black', marker = 'o', linestyle = 'none', markerfacecolor = 'none')
    plt.plot( T_leaf_x, J, color = 'red' )  
    
    # Convert the units of Jmax to mol m-2 s-1 from micro-mol m-2 s-1
    popt_J[-1] = popt_J[-1] * 1.0e-6
      
    # =============================================================================
    # Fit the leaf energy balance equation to the observed 
    # relationship of dT on gs to determine Is and ra.
    # =============================================================================
    # We will remove missing data for gs
    idx = ( ~gs_is_nan_idx )
    
    # Define a function of dT vs gs with Is and ra as parameters
    fit_calc_dT_leaf_cuv = lambda gs, Is, ra: calc_Tleaf( pc, gs, T_cuv[idx], Is, pa, ra, D ) - T_cuv[idx]
        
    
    popt, pcov = curve_fit( fit_calc_dT_leaf_cuv, gs_h2o_m_s[idx] , (T_leaf - T_cuv).values[idx], 
                            p0 = [100,10], bounds = ([0,0],[1000,100]), maxfev = 10000 )
    Is, ra   = popt
    print("Rnet = %.4f, ra = %.4f"%(Is,ra))
    
    # =============================================================================
    # Plot the results of the dT fit for Figure S4     
    # =============================================================================
    # Calculate the predicted dT
    dT_model = calc_Tleaf( pc, gs_h2o_m_s, T_cuv, Is, pa, ra, D ) - T_cuv
    # Plot results
    axs_S4[i].plot( gs_h2o_mol, T_leaf - T_cuv, color = 'black', marker = 'o', linestyle = 'none',markerfacecolor = 'none' )
    axs_S4[i].plot( gs_h2o_mol, dT_model, color = 'red', linestyle = 'none', marker = 'o' ) 
    
    
    # =============================================================================
    # Fit the LTO, LT, and AT models to the observed data to determine
    # the gcrit paramater.
    # =============================================================================
    idx_LTO = ( ~gs_is_nan_idx )
    idx_LT  = ( T_leaf.values < popt_J[2]) & ( ~gs_is_nan_idx )
    idx_AT  = ( T_leaf.values < popt_J[2]) & ( ~gs_is_nan_idx )
    
    
    f_gcrit_LTO = lambda T_cuv, gcrit: numerical_solve_Diao_LTO( pc, T_cuv, ca, pa, oa, Is, ra, D, gcrit, popt_J, species)[1]
    popt, pcov = curve_fit( f_gcrit_LTO, T_cuv.values[idx_LTO], gs_h2o_mol.values[idx_LTO] )
    print("gcrit_LTO = %.4f"%(popt[0]))    
    gcrit_LTO = popt[0]
    
    f_gcrit_LT = lambda T_cuv, gcrit: numerical_solve_Diao_LT( pc, T_cuv, ca, pa, oa, Is, ra, D, gcrit, popt_J, species)[1]
    popt, pcov = curve_fit( f_gcrit_LT, T_cuv.values[idx_LT], gs_h2o_mol.values[idx_LT] )
    print("gcrit_LT = %.4f"%(popt[0]))    
    gcrit_LT = popt[0]
    
    f_gcrit_AT = lambda T_cuv, gcrit: numerical_solve_Diao_AT( pc, T_cuv, ca, pa, oa, Is, ra, D, gcrit, popt_J, species)[1]
    popt, pcov = curve_fit( f_gcrit_AT, T_cuv.values[idx_AT], gs_h2o_mol.values[idx_AT] )
    print("gcrit_AT = %.4f"%(popt[0]))    
    gcrit_AT = popt[0]
    
    # =============================================================================
    # Fit the LT and AT models using the entire data range of gs for figure S5
    # =============================================================================
    idx_LT_S5     = ( ~gs_is_nan_idx )
    f_gcrit_LT_S5 = lambda T_cuv, gcrit: numerical_solve_Diao_LT( pc, T_cuv, ca, pa, oa, Is, ra, D, gcrit, popt_J, species)[1]
    popt_S5, pcov = curve_fit( f_gcrit_LT, T_cuv.values[idx_LT_S5], gs_h2o_mol.values[idx_LT_S5] ) 
    gcrit_LT_S5   = popt_S5[0]
    
    idx_AT_S5     = ( ~gs_is_nan_idx )
    f_gcrit_AT_S5 = lambda T_cuv, gcrit: numerical_solve_Diao_AT( pc, T_cuv, ca, pa, oa, Is, ra, D, gcrit, popt_J, species)[1]
    popt_S5, pcov = curve_fit( f_gcrit_AT, T_cuv.values[idx_AT], gs_h2o_mol.values[idx_AT] )  
    gcrit_AT_S5   = popt_S5[0]
    
    # =============================================================================
    # Now we have fitted the model to the data we can calculate the final model
    # output across the range of cuvette temperatures (to be plotted)
    # =============================================================================
    A_LTO_x, gs_LTO_x, ci_LTO_x, T_leaf_LTO_x = numerical_solve_Diao_LTO( pc, T_cuv_x, ca, pa, oa, Is, ra, D, gcrit_LTO, popt_J, species )
    A_LT_x, gs_LT_x, ci_LT_x, T_leaf_LT_x     = numerical_solve_Diao_LT( pc, T_cuv_x, ca, pa, oa, Is, ra, D, gcrit_LT, popt_J, species )
    A_AT_x, gs_AT_x, ci_AT_x, T_leaf_AT_x     = numerical_solve_Diao_AT( pc, T_cuv_x, ca, pa, oa, Is, ra, D, gcrit_AT, popt_J, species )
    
    # Repeat for LT and AT using the gcrit values calculated for Fig S5
    A_LT_x_S5, gs_LT_x_S5, ci_LT_x_S5, T_leaf_LT_x_S5     = numerical_solve_Diao_LT( pc, T_cuv_x, ca, pa, oa, Is, ra, D, gcrit_LT_S5, popt_J, species )
    A_AT_x_S5, gs_AT_x_S5, ci_AT_x_S5, T_leaf_AT_x_S5     = numerical_solve_Diao_AT( pc, T_cuv_x, ca, pa, oa, Is, ra, D, gcrit_AT_S5, popt_J, species )
    
    # =============================================================================
    # Plot model predictions for Fig 4.
    # =============================================================================
    # ========== Photosynthesis ====================== (The factor of 1e6 converts back to micro-mol m-2 s-1)
    axs_4[0,i].plot(T_cuv_x, A_LTO_x * 1.0e6, color = model_colors['Leaf Temperature within Optimisation (LTO)'])
    axs_4[0,i].plot(T_cuv_x, A_LT_x * 1.0e6 , color = model_colors['Leaf Temperature (LT)'])
    axs_4[0,i].plot(T_cuv_x, A_AT_x * 1.0e6 , color = model_colors['Air Temperature (AT)'])
    
    # ========== Stomatal Conductance ================   
    axs_4[1,i].plot(T_cuv_x, gs_LTO_x, color = model_colors['Leaf Temperature within Optimisation (LTO)'])
    axs_4[1,i].plot(T_cuv_x, gs_LT_x, color = model_colors['Leaf Temperature (LT)'])
    axs_4[1,i].plot(T_cuv_x, gs_AT_x, color = model_colors['Air Temperature (AT)'])
    
    # ============== Delta T vs gs ===================
    axs_4[2,i].plot(T_leaf_LTO_x, T_leaf_LTO_x - T_cuv_x, color = model_colors['Leaf Temperature within Optimisation (LTO)'])
    axs_4[2,i].plot(T_leaf_LT_x, T_leaf_LT_x - T_cuv_x, color = model_colors['Leaf Temperature (LT)'])
    axs_4[2,i].plot(T_leaf_AT_x, T_leaf_AT_x - T_cuv_x, color = model_colors['Air Temperature (AT)'])
    
    # =============================================================================
    # Plot model predictions for Fig S5.
    # =============================================================================
    # ========== Photosynthesis ====================== (The factor of 1e6 converts back to micro-mol m-2 s-1)
    axs_S5[0,i].plot(T_cuv_x, A_LTO_x * 1.0e6, color = model_colors['Leaf Temperature within Optimisation (LTO)'])
    axs_S5[0,i].plot(T_cuv_x, A_LT_x_S5 * 1.0e6 , color = model_colors['Leaf Temperature (LT)'])
    axs_S5[0,i].plot(T_cuv_x, A_AT_x_S5 * 1.0e6 , color = model_colors['Air Temperature (AT)'])
    
    # ========== Stomatal Conductance ================   
    axs_S5[1,i].plot(T_cuv_x, gs_LTO_x, color = model_colors['Leaf Temperature within Optimisation (LTO)'])
    axs_S5[1,i].plot(T_cuv_x, gs_LT_x_S5, color = model_colors['Leaf Temperature (LT)'])
    axs_S5[1,i].plot(T_cuv_x, gs_AT_x_S5, color = model_colors['Air Temperature (AT)'])
    
    # ============== Delta T vs gs ===================
    axs_S5[2,i].plot(T_leaf_LTO_x, T_leaf_LTO_x - T_cuv_x, color = model_colors['Leaf Temperature within Optimisation (LTO)'])
    axs_S5[2,i].plot(T_leaf_LT_x_S5, T_leaf_LT_x_S5 - T_cuv_x, color = model_colors['Leaf Temperature (LT)'])
    axs_S5[2,i].plot(T_leaf_AT_x_S5, T_leaf_AT_x_S5 - T_cuv_x, color = model_colors['Air Temperature (AT)'])
    
    # =============================================================================
    # Calculate the NRMSE values between the model predictions and obs    
    # - We must first calculate the model predictions at the measured Tcuv values
    # =============================================================================
    A_LTO, gs_LTO, ci_LTO, T_leaf_LTO = numerical_solve_Diao_LTO( pc, T_cuv.values, ca, pa, oa, Is, ra, D, gcrit_LTO, popt_J, species )
    A_LT, gs_LT, ci_LT, T_leaf_LT     = numerical_solve_Diao_LT( pc, T_cuv.values, ca, pa, oa, Is, ra, D, gcrit_LT, popt_J, species )
    A_AT, gs_AT, ci_AT, T_leaf_AT     = numerical_solve_Diao_AT( pc, T_cuv.values, ca, pa, oa, Is, ra, D, gcrit_AT, popt_J, species )
    
    # Calculate error metrics
    NRMSE_A_LTO  = mse( A[~A.isna()] * 1.0e-6, A_LTO[~A.isna()] ) ** 0.5 / (1.0e-6*A.mean(skipna = True)) * 100
    NRMSE_A_LT   = mse( A[~A.isna()] * 1.0e-6, A_LT[~A.isna()] ) ** 0.5 / (1.0e-6*A.mean(skipna = True)) * 100
    NRMSE_A_AT   = mse( A[~A.isna()] * 1.0e-6, A_AT[~A.isna()] ) ** 0.5 / (1.0e-6*A.mean(skipna = True)) * 100
    NRMSE_gs_LTO = mse( gs_h2o_mol[~gs_h2o_mol.isna()], gs_LTO[~gs_h2o_mol.isna()]   ) ** 0.5 / gs_h2o_mol.mean(skipna = True) * 100
    NRMSE_gs_LT  = mse( gs_h2o_mol[~gs_h2o_mol.isna()], gs_LT[~gs_h2o_mol.isna()] ) ** 0.5 / gs_h2o_mol.mean(skipna = True) * 100
    NRMSE_gs_AT  = mse( gs_h2o_mol[~gs_h2o_mol.isna()], gs_AT[~gs_h2o_mol.isna()] ) ** 0.5 / gs_h2o_mol.mean(skipna = True) * 100
    NRMSE_dT_LTO = mse( dT[~dT.isna()], T_leaf_LTO[~dT.isna()] - T_cuv[~dT.isna()] ) ** 0.5 / dT.mean(skipna = True  ) * 100
    NRMSE_dT_LT  = mse( dT[~dT.isna()], T_leaf_LT[~dT.isna()] - T_cuv[~dT.isna()] ) ** 0.5 / dT.mean(skipna = True) * 100
    NRMSE_dT_AT  = mse( dT[~dT.isna()], T_leaf_AT[~dT.isna()] - T_cuv[~dT.isna()] ) ** 0.5 / dT.mean(skipna = True) * 100
    
    # Store metrics in a dictionary
    NRMSE_values['LTO'][species]['A']  = NRMSE_A_LTO
    NRMSE_values['LT'][species]['A']   = NRMSE_A_LT
    NRMSE_values['AT'][species]['A']   = NRMSE_A_AT
    
    NRMSE_values['LTO'][species]['gs'] = NRMSE_gs_LTO
    NRMSE_values['LT'][species]['gs']  = NRMSE_gs_LT
    NRMSE_values['AT'][species]['gs']  = NRMSE_gs_AT
    
    NRMSE_values['LTO'][species]['dT'] = NRMSE_dT_LTO
    NRMSE_values['LT'][species]['dT']  = NRMSE_dT_LT
    NRMSE_values['AT'][species]['dT']  = NRMSE_dT_AT


fig_4.savefig('Figures/Figure_4.jpg', dpi = 300, bbox_inches = 'tight')    
fig_S4.savefig('Figures/Figure_S4.jpg', dpi = 300, bbox_inches = 'tight')    
fig_S5.savefig('Figures/Figure_S5.jpg', dpi = 300, bbox_inches = 'tight')    

# Add titles to figures to identify them (They are not saved with these titles)
fig_4.suptitle('Figure 4', size = 20)
fig_S4.suptitle('Figure S4', size = 20, y = 1)
fig_S5.suptitle('Figure S5', size = 20)

#%%
# =============================================================================
# Calculate species averaged NRMSE
# =============================================================================
for model in ['LTO','LT','AT']:
    for var in ['A','gs','dT']:
        NMRSE_mean = np.mean([NRMSE_values[model][s][var] for s in plant_species])
        print('%s %s: %s'%(model,var,NMRSE_mean))
        
