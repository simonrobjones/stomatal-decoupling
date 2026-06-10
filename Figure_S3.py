# -*- coding: utf-8 -*-
"""
Code to produce figure S3 of Jones et al
@author: srgj201
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
import matplotlib.markers as mmarkers
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
import string
import math
from loess import loess_1d
from tqdm import tqdm
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

plt.style.use('ggplot')
path = 'C:/Users/srgj201/OneDrive - University of Exeter/Documents/Postdoc/SOX/High_T_decoupling/Jones et al 25/Code/Revised code/GCB submission/Modelling_results/'


 
data_labels = {
'Diao24_Fagus sylvatica_out.csv' : '$\it{F.\ sylvatica}$', 
'Diao24_Picea abies_out.csv' : '$\it{P.\ abies}$', 
'Diao24_Quercus petraea_out.csv' : '$\it{Q.\ petraea}$', 
'Diao24_Tilia cordata_out.csv' : '$\it{T.\ cordata}$', 
'Feng23_cccontrol_out.csv' : '$\it{C.\ arborescens}$ \n (ww)', 
'Feng23_ccdry_out.csv' : '$\it{C.\ arborescens}$ \n (d)', 
'Feng23_hacontrol_out.csv' : '$\it{H.\ ammodendron}$ \n (ww)', 
'Feng23_hadry_out.csv' : '$\it{H.\ ammodendron}$ \n (d)', 
'Slot16_Calophyllum longifolium_out.csv' : '$\it{C.\ longifolium}$', 
'Slot16_Ficus insipida_out.csv' : '$\it{F.\ insipida}$', 
'Slot16_Ochroma_out.csv' : '$\it{O.\ pyramdidale}$', 
'SlotWinter24_Adelphia platyrachis_out.csv' : '$\it{A. platyrachis}$', 
'SlotWinter24_Amphilophium paniculatum_out.csv' : '$\it{A.\ paniculatum}$', 
'SlotWinter24_Astronium graveolens_out.csv' : '$\it{A.\ graveolens}$', 
'SlotWinter24_Bonamia trichantha_out.csv' : '$\it{B.\ trichantha}$', 
'SlotWinter24_Brosimum utile_out.csv' : '$\it{B.\ utile}$', 
'SlotWinter24_Cecropia peltata_out.csv' : '$\it{C.\ peltata}$', 
'SlotWinter24_Cordia bicolor_out.csv' : '$\it{C.\ bicolor}$', 
'SlotWinter24_Doliocarpus major_out.csv' : '$\it{D.\ major}$', 
'SlotWinter24_Doliocarpus major_out.csv' : '$\it{D.\ major}$', 
'SlotWinter24_Garcinia madruno_out.csv' : '$\it{G.\ madruno}$', 
'SlotWinter24_Guatteria dumetorum_out.csv' : '$\it{G.\ dumetorum}$', 
'SlotWinter24_Heisteria scandens_out.csv' : '$\it{H.\ scandens}$', 
'SlotWinter24_Luehea seemannii_out.csv' : '$\it{L.\ seemannii}$', 
'SlotWinter24_Passiflora vitifolia_out.csv' : '$\it{P.\ vitifolia}$', 
'SlotWinter24_Serjania mexicana_out.csv' : '$\it{S.\ mexicana}$', 
'SlotWinter24_Spondias mombin_out.csv' : '$\it{S.\ mombin}$', 
'SlotWinter24_Tocoyena pittieri_out.csv' : '$\it{T.\ pittieri}$', 
'SlotWinter24_Vantanea depleta_out.csv' : '$\it{V.\ depleta}$', 
'SlotWinter24_Aristolochia tonduzii_out.csv' : '$\it{A. tonduzzi}$',
'SlotWinter24_Carapa guianensis_out.csv' : '$\it{C. guiamemsis}$',
'SlotWinter24_Manilkara bidentata_out.csv' : '$\it{M. bidentata}$', 
'SlotWinter24_Miconia\xa0minutiflora_out.csv' : '$\it{M. minutiflora}$', 
'SlotWinter24_Nectandra\xa0cuspidata_out.csv' : '$\it{N. cuspidata}$', 
'SlotWinter24_Schefflera\xa0morototoni_out.csv' : '$\it{S. morototoni}$', 
'SlotWinter24_Virola multiflora_out.csv':'$\it{V. multiflora}$',
'TaylorND_Clethra fagifolia_out.csv' : '$\it{C.\ fagifolia}$', 
'TaylorND_Guatteria goudotiana_out.csv' : '$\it{G.\ goudotiana}$', 
'TaylorND_Ilex laurina_out.csv' : '$\it{I.\ laurina}$', 
'Urban17_loblolly_dry_out.csv' : '$\it{P.\ taeda}$ \n (d)', 
'Urban17_loblolly_wet_out.csv' : '$\it{P.\ taeda}$ \n (ww)', 
'Urban17_loblolly_wet_co2_out.csv' : '$\it{P.\ taeda}$ \n (ww eCO$_2$)', 
'Urban17_poplar_dry_out.csv' : '$\it{P.\ deltoides}$ \n (d)', 
'Urban17_poplar_wet_out.csv' : '$\it{P.\ deltoides}$ \n (ww)', 
'Urban17_poplar_wet_co2_out.csv' : '$\it{P.\ deltoides}$ \n (ww eCO$_2$)'
}

file_names = listdir( path )

model_colors = {'LTO':'#FFB000',
               'LT':'#DC267F',
               'AT':'#648FFF'}

y_labels = { 'A':'A ($\mu$ $mol m^{-2}$ $s^{-1}$)',
             'gsc':'g$_{sc}$ (mol $m^{-2}$ $s^{-1}$)',
             'ci':'c$_{i}$ (Pa)',
             'dT':'dT ($^oC$)'}

nrows = math.ceil(len(file_names)/2)
ncols = 10
fig = plt.figure(figsize=(45, 2 * nrows))
gs = gridspec.GridSpec(
    nrows, ncols,
    width_ratios=[1.0,1,1,1,1,1.0,1,1,1,1],
    wspace = 0.4,
    hspace = 0.2
)

axs = np.empty((nrows, ncols), dtype = object)

for i in range(nrows):
    ax = fig.add_subplot(gs[i,0])
    axs[i,0] = ax
    ax.axis("off")
    
    

for i,f in enumerate(file_names):
    df = pd.read_csv( path + f, header = [0,1] )
    df = df.sort_values(by = ('Ta','obs'))
    
    Ta = df.loc[:,('Ta','obs')]
    
    for j,var in enumerate(['A','gsc','ci','dT']):
        
        if i == 0 and j== 1:
            ax = fig.add_subplot( gs[ i % nrows, j + (i//nrows) * 5 + 1 ])
            axs[i%nrows, j + (i//nrows)*5] = ax
            
        else:
            ax = fig.add_subplot(gs[ i%nrows, j + (i//nrows)*5 + 1], sharex = axs[0,1])
        if i%nrows == 0:
            ax.text(0.5,1.2, y_labels[var], size = 25, transform = ax.transAxes, ha = 'center')
        if var == 'A':
            mod_fac = 1e6
            order   = 2
        else:
            order   = 3
            mod_fac = 1
        obs = df.loc[:,(var,'obs')]
        sns.regplot(x=Ta, y = obs, order = order, ax = ax, color = 'black', scatter_kws={'facecolor':'none'}, label = 'Observations')#, ci = 100)
        
        for k, model in enumerate(['LT','LTO']):
            mod = df.loc[:,(var,model)] * mod_fac
            ax.scatter( Ta, mod,  color = model_colors[model], label = model)
        

        ax.set_ylabel( '' )   
        if i%nrows != nrows - 1:
            ax.tick_params(axis='x', labelbottom=False)
            ax.set_xlabel('')
        else:
            ax.set_xlabel('Ta ($^o$C)',size = 25)
           

for i in range(nrows):
    ax = fig.add_subplot(gs[i,5])
    axs[i,5] = ax
    ax.axis("off")

for i,f in enumerate(file_names):
    
    ax = axs[i%nrows,5*(i//nrows)]
    ax.text(-0.,0.5,data_labels[f], transform = ax.transAxes, ha = 'left', size = 24)

line_colors = {'LTO':'#FFB000',
               'LT':'#DC267F',
               'AT':'#648FFF'}
line_labels = {'LTO':'Leaf Temperature within Optimisation (LTO)',
               'LT':'Leaf Temperature (LT)',
               'AT':'Air Temperature (AT)'}
    
handles, labels = axs[0,1].get_legend_handles_labels()

handles = [(Patch(facecolor = 'black',alpha = 0.2),Line2D([0], [0], color = 'black',lw = 1.0))] + handles
labels  = ['Polynomial fit'] + labels
fig.legend(handles, labels, markerscale=3,
           loc = 'upper center', bbox_to_anchor = (0.5,0.06), fontsize = 30, ncol = 4)

#%%
Fig_path = 'C:/Users/srgj201/OneDrive - University of Exeter/Documents/Postdoc/SOX/High_T_decoupling/Jones et al 25/Figures/Revision_2/'
fig.savefig(Fig_path + 'Fig_S4.jpg',dpi = 300, bbox_inches = 'tight')
