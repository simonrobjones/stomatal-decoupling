# -*- coding: utf-8 -*-
"""
Code to produce Figure 5 of Jones et al
@author: srgj201
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from sklearn.metrics import r2_score
import string
import math

Fig_path = 'Figures/'
# =============================================================================
# Get names of all files
# =============================================================================
data_path = 'Modelling_results/'
file_names = listdir( data_path )

# =============================================================================
# Set up figure
# =============================================================================
ncols = math.ceil( len(file_names)**0.5 )
nrows = math.ceil( len(file_names) / ncols )
fig,axs = plt.subplots(ncols = ncols, nrows = nrows, figsize = (6*ncols, 4.5*nrows))
axs = axs.reshape(-1)
Ta_maxs = []
Ta_mins = []

# =============================================================================
# Find maximum and minimum air temperature across all datasets
# =============================================================================
for f in file_names:
    df = pd.read_csv( data_path + f, header = [0,1] )
    Ta = df.loc[:,('Ta','obs')]
    
    Ta_maxs.append(Ta.max())
    Ta_mins.append(Ta.min())

idx = pd.IndexSlice
Tamax = max(Ta_maxs)
Tamin = min(Ta_mins)

# =============================================================================
# Concatenate all datasets
# =============================================================================
df_all = []
for i,f in enumerate(file_names):
        
    df = pd.read_csv( data_path + f, header = [0,1] )
    df = df.sort_values(by = ('Ta','obs'))
    
    df_all.append(df)
    Ta = df.loc[:,('Ta','obs')]
    gs = df.loc[:,('gsc','obs')]
    gs_LT  = df.loc[:,('gsc','LT')]
    gs_LTO = df.loc[:,('gsc','LTO')]
    axs[i].scatter( Ta, gs )
    axs[i].plot(Ta, gs_LT, color = 'black')
    axs[i].plot(Ta, gs_LTO, color = 'red')

df_all = pd.concat( df_all, ignore_index = True )


plt.rcParams["text.usetex"] = False
ncols = 2
nrows = 4
fig,axs = plt.subplots(ncols = ncols, nrows = nrows, figsize = (6*ncols, 5*nrows), sharey = 'row', sharex = 'row')
if ncols == 1:
    axs = axs[:,np.newaxis]
if nrows == 1:
    axs = axs[np.newaxis,:]
plt.subplots_adjust(hspace = 0.6)
plt.subplots_adjust(wspace = 0.1)

df_all = df_all.sort_values(by = ('Ta','obs'))
var_labels = {'A':'$A_n$ ($\mu$mol m$^{-2}$ s$^{-1}$)',
              'gsc':'$g_{sc}$ (mol m$^{-2}$ s$^{-1}$)',
              'ci':'$c_i$ (Pa)', 
              'dT':'dT ($^o$C)'}



for j,src in enumerate(['LT','LTO']):
        Ta = df_all.loc[:,('Ta','obs')]
        axs[0,j].text( 0.5, 1.2, src, transform = axs[0,j].transAxes, size = 25, ha = 'center')
        for k,var in enumerate(['A','gsc','ci','dT']):
            obs = df_all.loc[:,(var,'obs')]
            mod = df_all.loc[:,(var,src)]
            
            warm_idx = Ta > 35.0
            obs_warm = df_all.loc[warm_idx,(var,'obs')]
            mod_warm = df_all.loc[warm_idx,(var,src)]
            obs_cool = df_all.loc[~warm_idx,(var,'obs')]
            mod_cool = df_all.loc[~warm_idx,(var,src)]
            
            if var == 'A':
                mod = mod*1e6
                mod_warm = mod_warm * 1e6
                mod_cool = mod_cool * 1e6
            
            r2 = r2_score( obs, mod )   
            r2_warm = r2_score( obs_warm, mod_warm )   
            
            im = axs[k,j].scatter( obs, mod, c = Ta, cmap = 'coolwarm', vmin = Tamin, vmax = Tamax)#, marker = markers[i] )
            axs[k,j].axline( (obs.mean(), obs.mean()), slope = 1, ls = '--', color = 'grey' )
            # axs[k,j].set_title( src + ' ' + var, size = 16)
            if var in ['A','gsc']:
                new_line = '\n'
            else:
                new_line = ''
            
            axs[k,j].text( 0.01, 1.03, '\n R$^2$ = %.3f'%(r2), color = 'black', transform = axs[k,j].transAxes, size = 16, ha = 'left')
            axs[k,j].text( 0.99, 1.03, '\n R$^2$ = %.3f (T$_a$>35$^o$C)'%(r2_warm), color = 'red', transform = axs[k,j].transAxes, size = 16, ha = 'right')
            axs[k,j].set_xlabel('Observed %s'%var_labels[var],size = 18)
            axs[k,0].set_ylabel('Modelled %s%s'%(new_line,var_labels[var]),size = 18)          
            
            # axs[k,j].set_aspect('equal', adjustable='datalim')
            # axs[k,j].set_xlim(axs[k,j].get_ylim())

axs_flat = axs.reshape(-1)
for i,ax in enumerate(axs_flat):
    ax.text( 0.01, 0.98, '(%s)'%string.ascii_lowercase[i], transform = ax.transAxes, 
             ha = 'left', va = 'top', size = 16 )

cax = fig.add_axes([0.15, 0.05, 0.7, 0.03])  
cbar = plt.colorbar(im, cax = cax, orientation = 'horizontal')
cbar.set_label('Air temperature', size = 18)
cax.tick_params( axis = 'both', labelsize = 16)

fig.savefig(Fig_path + 'Figure_5.jpg', dpi = 300, bbox_inches = 'tight')
