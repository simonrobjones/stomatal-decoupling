# -*- coding: utf-8 -*-
"""
Code to produce figure S5 of Jones et al
@author: srgj201
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PGEN_functions_complete import namelist, physical_constants
from PGEN_functions_complete import numerical_solve_LTO, numerical_solve_LT
from PGEN_functions_complete import calc_A_from_gs_C3
from PGEN_functions_complete import calc_Tleaf
from scipy.optimize import curve_fit
import emcee
import corner
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerLine2D

Fig_path = 'Figures/'

# =============================================================================
# MCMC functions
# =============================================================================
def A_mcmc( theta, x ):
    log_Ea, log_eta, log_Topt, log_vcmax25, log_jmax25, log_fd, log_q10, log_sigma = theta
    Ea, eta, Topt, vcmax25, jmax25, fd, q10, sigma = [10**(x) for x in theta]
    A = calc_A_from_gs_C3( pc, nl, Tl, x, Ca, Pa, PAR, Oa, Ea, eta, Topt, vcmax25, jmax25, fd = fd, q10 = q10, rd_func = None ) * 1.0e6
    return A

def log_likelihood( theta, x, y ):
    y_model = A_mcmc(theta, x)
    log_Ea, log_eta, log_Topt, log_vcmax25, log_jmax25, log_fd, log_q10, log_sigma = theta
    sigma = 10 ** (log_sigma)
    return - 0.5 * np.sum( ( ( y - y_model )/ sigma )**2.0 + np.log( 2 * np.pi * sigma ** 2) )

def log_prior(theta):
    log_Ea, log_eta, log_Topt, log_vcmax25, log_jmax25, log_fd, log_q10, log_sigma = theta
    if ( ( 1 < log_Ea < 3 ) and ( 0 < log_eta < 2 ) and ( 1 < log_Topt < 2 ) and ( -6 < log_vcmax25 < -2 ) and ( -6 < log_jmax25 < -2 ) and ( -10 < log_fd < -1 ) and ( 0 < log_q10 < 1 ) and ( -20 < log_sigma < 20 ) ):
        return 0.0
    else:
        return -np.inf

def log_prob( theta, x, y ):
    lp = log_prior(theta)
    if lp == -np.inf:
        return -np.inf
    else:
        return lp + log_likelihood( theta, x, y )
    
def mcmc_main(p0, nwalkers, niter, burn_iter, ndim, log_prob, data, filename ):
    # Set up backend
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    # Set up sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args = data, backend = backend )
    # Burn in to let walkers settle?
    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc( p0, burn_iter, progress = True )
    sampler.reset()
    # Full MCMC
    print("Running production...")
    pos, prob, state = sampler.run_mcmc( p0, niter, progress = True, store = True )
    return sampler, pos, prob, state 

# =============================================================================
# Load in data. Data available on request at 10.1071/FP15320
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

run_mcmc    = False
run_samples = False
# =============================================================================
# Fit photosynthesis, energy balance and stomtatal conductance for each species
# =============================================================================
# Set up figure
ncols = len(unique_species)
nrows = 4
fig,axs = plt.subplots(ncols = ncols, nrows = nrows, figsize = (6*ncols, 4*nrows))#, sharey = 'row', sharex = 'row')
if ncols == 1:
    axs = axs[:,np.newaxis]
axs[0,0].set_ylabel('$g_{sc}$ (mol m$^{-2}$ s$^{-1}$)', size = 20)
axs[1,0].set_ylabel('A ($\mu$mol m$^{-2}$ s$^{-1}$)', size = 20)
axs[2,0].set_ylabel('c$_i$ (Pa)', size = 20)
axs[3,0].set_ylabel('dT ($^o$C)', size = 20)

for i,species in enumerate(unique_species):
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
    f = lambda gs, Is, ra: calc_Tleaf( pc, gs, Ta, Is, Pa, ra, VPD_air ) - Ta

    popt, pcov = curve_fit( f, gs_h2o_m_s, dT, p0 = [300,10], bounds = ([0,0.1],[1000,50]))
    Is, ra = popt
    Is_SE, ra_SE = np.sqrt( np.diag(pcov) )
    print("Is = %s, ra = %s"%(Is, ra))
    
    gb_h2o_m_s = 1.0 / ra
    gb_h2o_mol = gb_h2o_m_s * Pa / ( 8.314462 * ( Ta + 273.15 ) )
    gb_co2_mol = gb_h2o_mol / 1.6
    rb_co2_mol = 1.0 / gb_h2o_mol
    gt_co2_mol = 1.0 / ( 1.0 / gs_co2_mol + rb_co2_mol )

    dT_an = calc_Tleaf( pc, gs_h2o_m_s, Ta, Is, Pa, ra, VPD_air ) - Ta
    
    # =============================================================================
    # Fit Farquhar MCMC
    # =============================================================================
    data      = ( gt_co2_mol, A )
    nwalkers  = 100
    niter     = 1000
    burn_iter = 100
    ndim      = 8
  
    prior_theta = {'Ea':77.8,
                   'eta':3.36,
                   'Topt':36.3,
                   'vcmax25':4.3e-5,
                   'jmax25':0.001,
                   'fd':0.00266,
                   'q10':2.96,
                   'sigma':1.0}
        
    p0            = [ np.array( [np.log10( prior_theta[p] * ( 1.0 + 0.1 * np.random.randn())) for p in ['Ea','eta','Topt','vcmax25','jmax25','fd','q10','sigma']]) for i in range(nwalkers)]
    mcmc_filename = "MCMC_results/" + "chain_Slot16_%s.h5"%(species)
    
    if run_mcmc:
        sampler, pos, prob, state = mcmc_main( p0, nwalkers, niter, burn_iter, ndim, log_prob, data, mcmc_filename )
    else:
        backend = emcee.backends.HDFBackend(mcmc_filename)
        sampler = emcee.EnsembleSampler( nwalkers, ndim, log_prob, args = data, backend = backend )
        
    # =============================================================================
    # Corner plot
    # =============================================================================
    labels        = ['Ea', 'eta', 'Topt', 'vcmax25', 'jmax25', 'fd', 'q10','sigma']
    samples       = sampler.flatchain    
    fig_corner    = corner.corner(samples,show_titles = True,labels=labels,
                                  plot_datapoints = True )
    fig_corner.suptitle(species)


    # =============================================================================
    # Plot data
    # =============================================================================
    line_colors = {'LTO':'#FFB000',
                   'LT':'#DC267F',
                   'AT':'#648FFF'}
    line_labels = {'LTO':'Leaf Temperature within Optimisation (LTO)',
                   'LT':'Leaf Temperature (LT)',
                   'AT':'Air Temperature (AT)'}
    
    axs[0,i].set_title(species, size = 20, style = 'italic')
    axs[0,i].scatter(Ta, gs_co2_mol, color = 'black')
    axs[1,i].scatter(Ta, A, color = 'black')
    axs[2,i].scatter(Ta, Ci, color = 'black' )
    axs[3,i].scatter(Ta, dT, color = 'black' )
    axs[3,i].set_xlabel('Ta ($^o$C)', size = 20)
    
    N_samples = 100
    samples_out_gs_LTO = np.zeros((len(Ta),N_samples))
    samples_out_ci_LTO = np.zeros((len(Ta),N_samples))
    samples_out_dT_LTO = np.zeros((len(Ta),N_samples))
    samples_out_A_LTO  = np.zeros((len(Ta),N_samples))
    
    samples_out_gs_LT  = np.zeros((len(Ta),N_samples))
    samples_out_ci_LT  = np.zeros((len(Ta),N_samples))
    samples_out_dT_LT  = np.zeros((len(Ta),N_samples))
    samples_out_A_LT   = np.zeros((len(Ta),N_samples))
    
    
    if run_samples:
        for j,theta in enumerate(samples[np.random.randint(len(samples), size=N_samples)]):

            log_Ea, log_eta, log_Topt, log_vcmax25, log_jmax25, log_fd, log_q10, log_sigma = theta
            Ea, eta, Topt, vcmax25, jmax25, fd, q10, sigma = [10**(x) for x in theta]
    
            nl.Ea      = Ea * 1e3
            nl.eta     = eta
            nl.Topt    = Topt
            nl.vcmax25 = vcmax25 
            nl.jmax25  = jmax25
            nl.fd      = fd
            nl.q10     = q10
                
            # =============================================================================
            # Fit stomatal conductance
            # =============================================================================
            idx = Tl <= 60
                 
            idx = Tl <= 60
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
            # Plot gs fit
            # =============================================================================
            A_LT, gs_LT, ci_LT, Tl_LT = numerical_solve_LT( nl, pc, Ta.values, Ca.values, Pa, Oa, Is, ra, vpd = VPD_leaf, Ipar = PAR,
                                                            rb_co2_mol = rb_co2_mol, vpd_air = VPD_air.values, beta = beta_LT )
        
            A_LTO, gs_LTO, ci_LTO, Tl_LTO = numerical_solve_LTO( nl, pc, Ta.values, Ca.values, Pa, Oa, Is, ra, vpd = VPD_leaf, Ipar = PAR,
                                                                 rb_co2_mol = rb_co2_mol, vpd_air = VPD_air.values, beta = beta_LTO )
            
            A_LTO = A_LTO * 1.0e6
            A_LT  = A_LT * 1.0e6
            samples_out_gs_LTO[:,j] = gs_LTO[np.argsort(Ta)]
            samples_out_ci_LTO[:,j] = ci_LTO[np.argsort(Ta)]
            samples_out_dT_LTO[:,j] = Tl_LTO[np.argsort(Ta)] - Ta.iloc[np.argsort(Ta)]
            samples_out_A_LTO[:,j]  = A_LTO[np.argsort(Ta)]
            
            samples_out_gs_LT[:,j]  = gs_LT[np.argsort(Ta)]
            samples_out_ci_LT[:,j]  = ci_LT[np.argsort(Ta)]
            samples_out_dT_LT[:,j]  = Tl_LT[np.argsort(Ta)] - Ta.iloc[np.argsort(Ta)]
            samples_out_A_LT[:,j]   = A_LT[np.argsort(Ta)]
        
        LTO_out = [samples_out_gs_LTO, 
                   samples_out_ci_LTO,
                   samples_out_dT_LTO,
                   samples_out_A_LTO
                   ]
        LT_out  = [samples_out_gs_LT, 
                   samples_out_ci_LT,
                   samples_out_dT_LT,
                   samples_out_A_LT
                   ]
        samples_out = np.array([LTO_out, LT_out])
        np.save("MCMC_results/" + "%s_samples.npy"%(species), samples_out )
    else:
        samples_out = np.load("MCMC_results/" + "%s_samples.npy"%(species))
        LTO_out = samples_out[0,:,:,:]
        LT_out  = samples_out[1,:,:,:]
        
        samples_out_gs_LTO = LTO_out[0,:,:]
        samples_out_ci_LTO = LTO_out[1,:,:]
        samples_out_dT_LTO = LTO_out[2,:,:]
        samples_out_A_LTO  = LTO_out[3,:,:]
        
        samples_out_gs_LT  = LT_out[0,:,:]
        samples_out_ci_LT  = LT_out[1,:,:]
        samples_out_dT_LT  = LT_out[2,:,:]
        samples_out_A_LT   = LT_out[3,:,:]
    # =============================================================================
    # Plot 50% ensemble and 1 stdev
    # =============================================================================
    axs[0,i].plot( Ta.iloc[np.argsort(Ta)], np.percentile(samples_out_gs_LTO, 50, axis = 1), color = '#FFB000', lw = 2 )
    axs[0,i].plot( Ta.iloc[np.argsort(Ta)], np.percentile(samples_out_gs_LT, 50, axis = 1), color = '#DC267F', lw = 2 )
    axs[0,i].fill_between( Ta.iloc[np.argsort(Ta)], 
                           np.percentile(samples_out_gs_LTO, 50, axis = 1) - np.std(samples_out_gs_LTO,axis = 1), 
                           np.percentile(samples_out_gs_LTO, 50, axis = 1) + np.std(samples_out_gs_LTO,axis = 1),
                           color = '#FFB000', alpha = 0.5 )
    axs[0,i].fill_between( Ta.iloc[np.argsort(Ta)], 
                           np.percentile(samples_out_gs_LT, 50, axis = 1) - np.std(samples_out_gs_LT,axis = 1), 
                           np.percentile(samples_out_gs_LT, 50, axis = 1) + np.std(samples_out_gs_LT,axis = 1),
                           color = '#DC267F', alpha = 0.5 )
    
    
    axs[1,i].plot( Ta.iloc[np.argsort(Ta)], np.percentile(samples_out_A_LTO, 50, axis = 1), color = '#FFB000', lw = 2 )
    axs[1,i].plot( Ta.iloc[np.argsort(Ta)], np.percentile(samples_out_A_LT, 50, axis = 1), color = '#DC267F', lw = 2 )
    axs[1,i].fill_between( Ta.iloc[np.argsort(Ta)], 
                           np.percentile(samples_out_A_LTO, 50, axis = 1) - np.std(samples_out_A_LTO,axis = 1), 
                           np.percentile(samples_out_A_LTO, 50, axis = 1) + np.std(samples_out_A_LTO,axis = 1),
                           color = '#FFB000', alpha = 0.5 )
    axs[1,i].fill_between( Ta.iloc[np.argsort(Ta)], 
                           np.percentile(samples_out_A_LT, 50, axis = 1) - np.std(samples_out_A_LT,axis = 1), 
                           np.percentile(samples_out_A_LT, 50, axis = 1) + np.std(samples_out_A_LT,axis = 1),
                           color = '#DC267F', alpha = 0.5 )
    
    
    axs[2,i].plot( Ta.iloc[np.argsort(Ta)], np.percentile(samples_out_ci_LTO, 50, axis = 1), color = '#FFB000', lw = 2 )
    axs[2,i].plot( Ta.iloc[np.argsort(Ta)], np.percentile(samples_out_ci_LT, 50, axis = 1), color = '#DC267F', lw = 2 )
    axs[2,i].fill_between( Ta.iloc[np.argsort(Ta)], 
                           np.percentile(samples_out_ci_LTO, 50, axis = 1) - np.std(samples_out_ci_LTO,axis = 1), 
                           np.percentile(samples_out_ci_LTO, 50, axis = 1) + np.std(samples_out_ci_LTO,axis = 1),
                           color = '#FFB000', alpha = 0.5 )
    axs[2,i].fill_between( Ta.iloc[np.argsort(Ta)], 
                           np.percentile(samples_out_ci_LT, 50, axis = 1) - np.std(samples_out_ci_LT,axis = 1), 
                           np.percentile(samples_out_ci_LT, 50, axis = 1) + np.std(samples_out_ci_LT,axis = 1),
                           color = '#DC267F', alpha = 0.5 )
    
    axs[3,i].plot( Ta.iloc[np.argsort(Ta)], np.percentile(samples_out_dT_LTO, 50, axis = 1), color = '#FFB000', lw = 2 )
    axs[3,i].plot( Ta.iloc[np.argsort(Ta)], np.percentile(samples_out_dT_LT, 50, axis = 1), color = '#DC267F', lw = 2 )
    axs[3,i].fill_between( Ta.iloc[np.argsort(Ta)], 
                           np.percentile(samples_out_dT_LTO, 50, axis = 1) - np.std(samples_out_dT_LTO,axis = 1), 
                           np.percentile(samples_out_dT_LTO, 50, axis = 1) + np.std(samples_out_dT_LTO,axis = 1),
                           color = '#FFB000', alpha = 0.5 )
    axs[3,i].fill_between( Ta.iloc[np.argsort(Ta)], 
                           np.percentile(samples_out_dT_LT, 50, axis = 1) - np.std(samples_out_dT_LT,axis = 1), 
                           np.percentile(samples_out_dT_LT, 50, axis = 1) + np.std(samples_out_dT_LT,axis = 1),
                           color = '#DC267F', alpha = 0.5 )

line_colors = {'LTO':'#FFB000',
               'LT':'#DC267F',
               'AT':'#648FFF'}
line_labels = {'LTO':'Leaf Temperature within Optimisation (LTO)',
               'LT':'Leaf Temperature (LT)',
               'AT':'Air Temperature (AT)'}
    
line_style = '-'#'none'
plot_schemes = ['LTO','LT']
custom_lines  = [Line2D([], [], color = line_colors[scheme], ls = line_style, marker = 'o' , markerfacecolor = 'none') for scheme in plot_schemes]

custom_objects = [(mpatches.Patch(facecolor = line_colors[scheme], alpha=0.5, linewidth=0), 
                  Line2D([], [], color = line_colors[scheme], ls = '-')) for scheme in ['LTO','LT']]
fig.legend( custom_objects, 
            [line_labels[s] for s in plot_schemes], 
            handler_map = {'line' : HandlerLine2D(marker_pad = 0)},
            loc = 'upper center',
            bbox_to_anchor = (0.5,0.05), 
            fontsize = 16, 
            ncol = len(plot_schemes))

fig.savefig(Fig_path + 'Figure_S5.jpg', dpi = 300, bbox_inches = 'tight')
