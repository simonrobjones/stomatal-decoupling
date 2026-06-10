# -*- coding: utf-8 -*-
"""
Code to produce figure S4 of Jones et al
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
path = 'Modelling_results/'
Fig_path = 'Figures/'

 
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

ncols   = 5
nrows   = 9
fig,axs = plt.subplots( ncols = ncols, nrows = nrows, figsize = (8*ncols, nrows * 6))
axs = axs.reshape(-1)
plt.subplots_adjust(hspace = 0.6)

df_all = []
for f in file_names:
    df    = pd.read_csv( path + f, header = [0,1] )
    df_all.append(df)
df_all = pd.concat( df_all, ignore_index = True )
vmin   = df_all.loc[:,('Tl','obs')].min()
vmax   = df_all.loc[:,('Tl','obs')].max()


for i,f in enumerate(file_names):
    df    = pd.read_csv( path + f, header = [0,1] )
    Tl    = df.loc[:,('Tl','obs')]
    A_obs = df.loc[:,('A','obs')]
    A_fit = df.loc[:,('A_farq','mod')] 
    r2    = r2_score( A_obs, A_fit )
    print(r2)
    im = axs[i].scatter( A_obs, A_fit, c = Tl, cmap = 'coolwarm', vmin = vmin, vmax = vmax)
    axs[i].axline( ( A_obs.mean(), A_obs.mean()), slope = 1, ls = '--', color = 'grey')
    axs[i].set_title( data_labels[f], size = 30)
    axs[i].set_xlabel('Observed A', size = 20)
    axs[i].set_ylabel('Fitted A', size = 20)
    axs[i].text(0.01, 0.9, 'R$^2$ = %.3f'%(r2), transform = axs[i].transAxes, size = 20 )
    

cax = fig.add_axes([0.3, 0.07, 0.4, 0.01])  
cbar = plt.colorbar(im, cax = cax, orientation = 'horizontal')
cbar.set_label('Leaf temperature', size = 30)
cax.tick_params( axis = 'both', labelsize = 20)


fig.savefig(Fig_path + 'Figure_S4.jpg',dpi = 300, bbox_inches = 'tight')
 
