

"""
    Stores the plotting settings to be used for plots.
    Import this module so that Tkinter or Agg can be checked

    Version 1 2020-03-29 Abhinav Kumar
"""
import matplotlib

# Check if Tkinter is there otherwise Agg
import imp
try:
    imp.find_module('_Tkinter')
    pass
except ImportError:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

DPI        = 200
ms         = 4
lw         = 2
alpha      = 0.9
size       = (10, 6)
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

# Remember to call update whenever you use a different fontsize
fs         = 15
matplotlib.rcParams.update({'font.size': fs})

IMAGE_DIR  = "images"
dodge_blue =  np.array([0.12, 0.56, 1.0]) #np.array([30, 144, 255])/255.
color1     = (1,0.45,0.45)
color2     = "dodgerblue"

mean_color         = "gold"
covar_color        = "dodgerblue"
covar_color_array  = dodge_blue
unoccluded_color   = "limegreen"
ext_occluded_color = "red"
