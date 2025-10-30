# -*- coding: utf-8 -*-
"""
Code to create Figure S1 of Jones et al

@author: srgj201
"""
import matplotlib.pyplot as plt
import numpy as np
from PGEN_functions import namelist, physical_constants
from PGEN_functions import calc_fT

# Create instance of namelist class
nl     = namelist()

# Generate range of temperatures to plot
T_leaf = np.linspace(-1,50,1000)

# Calculate temperature sensitivity
f = calc_fT( nl, T_leaf )
f = np.clip(f,0,None)

# Plot results
fig = plt.figure()
plt.plot( T_leaf, f, color = 'black')
plt.axvline(nl.Topt,color = 'grey',ls = '-.')
plt.axvline(nl.Tmax,color = 'grey',ls = '-.')
plt.axvline(nl.Tmin,color = 'grey',ls = '-.')
plt.xticks(ticks = [nl.Tmin, nl.Topt, nl.Tmax ], labels = ['$T_{min}$', '$T_{opt}$', '$T_{max}$'] )
plt.yticks(ticks = [ 0, 1 ], labels = [ '0', '$k_{T_{max}}$' ] )
plt.xlabel('Leaf temperature', size = 12)
plt.ylabel('Temperature sensitivity', size = 12)
fig.savefig('Figures/Figure_S1.jpg', dpi = 300, bbox_inches = 'tight')