#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:20:02 2024

@author: cedar
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import skimage
sys.path.append('/Users/cedar/data/pySimFrac/src')
target_folder = '/Users/cedar/Library/CloudStorage/Box-Box/Research_Cedar/fracture_generation'

# import the SimFrac Module
from pysimfrac import *
import tifffile
import pyvista as pv



def define_fracture(H, A, R, Mis, Aniso):
#Spectual method
# initialize the surface object
    myfrac = SimFrac(h = 1, lx = 256, ly = 256, 
                     method = "spectral", units = 'mm')
    # set the parameters for the spectral method
    myfrac.params['H']['value'] = H
    myfrac.params['mean-aperture']['value'] = A
    myfrac.params['roughness']['value'] = R
    myfrac.params['seed']['value'] = 1 
    myfrac.params['mismatch']['value'] = Mis
    myfrac.params["aniso"]["value"] = Aniso
    myfrac.params["model"]["value"] = "bilinear"
    # myfrac.shear = Shear
    return myfrac
#Gaussian method
# initialize the surface object
# myfrac = SimFrac(h = 0.01, lx = 4, ly = 2, 
#                  method = "gaussian", units = 'mm',
#                  shear = 0.5
#                 )
# # set the parameters for the Gaussian method
# myfrac.params['mean-aperture']['value'] = 1
# myfrac.params['aperture-log-variance']['value'] = 0.02
# myfrac.params['lambda_x']['value'] = 0.1
# myfrac.params['lambda_y']['value'] = 0.2
# myfrac.params['seed']['value'] = 1

# #Box method
# myfrac = SimFrac(h = 0.01, lx = 2, ly = 2, 
#                  method = "box", units = 'mm',
#                  shear = 1
#                 )
# # set the parameters for the Box method
# myfrac.params['mean-aperture']['value'] = 2
# myfrac.params['aperture-log-variance']['value'] = 0.2
# myfrac.params['lambda_x']['value'] = 0.4
# myfrac.params['lambda_y']['value'] = 0.3
# myfrac.params['seed']['value'] = 1

#Combine spectral and Gaussian
#Combined = Spectral.combine_fractures([Gaussian], [0.4, 0.6])
Hexps = np.arange(0.1,1,0.1)  # Hurst exponent (0,1)
As = np.arange(60,100,5)   # MeanAperture
# Rs = np.arange(0.1,1.5,0.1)   # roughness [0,)
Mis = np.arange(0.01,0.1,0.01)    # mismatch [0,1]
Anisos = np.arange(0.1,1,0.1) # aniso ratio (0,1)
# Shears = np.arange(0.1,1.1,0.1) # shear translates the top surface along x-axis

#%%
# test the function of parameters
testfrac = define_fracture(0.3, 44, 6, 0.3, 0.3)
testfrac.create_fracture()
testfrac.voxelize(target_size=256)
testfrac.plot_aperture_field()
# tifffile.imwrite(f'example.tif',testfrac.frac_3D)
# frac = skimage.io.imread(f"example.tif")
testfrac.frac_3D.astype('int8').tofile("/Users/cedar/data/hoomdblue/fno_test/frac_example.raw")
# frac = np.fromfile('example.raw',dtype=np.uint8).reshape([256,256,256])

#%% 3D visualization
pv_img = pv.wrap(frac)
def initialize_plotter():
    # Initialize a PyVista Plotter Object
    plotter_obj = pv.Plotter()

    # Set the background color to white
    plotter_obj.set_background(color='w')
    pv.global_theme.font.color = 'black'

    return plotter_obj


# Wrap 3D NumPy array to PyVista data object

p = initialize_plotter()
# Generate a contoured surface for rough fracture.
contour = pv_img.contour(isosurfaces=[0.5])
p.add_bounding_box(line_width=18, color='black')
p.add_mesh(contour, color=(0, 0.8, 0.27), opacity=1.)
p.show(return_cpos=True, auto_close=False)
p.screenshot(filename=f'frac_State.png',return_img=True)
#%%
# for hexp in Hexps:
#     for A in As:
#         for mis in Mis:
#             myfrac = define_fracture(hexp, A, 6, mis, 0.1)
#             myfrac.create_fracture()
#             # myfrac.top *= 10 #myfrac.h
#             # myfrac.bottom *= 10 #myfrac.h
#             myfrac.voxelize(target_size=256)
#             # myfrac.plot_aperture_field()
#             tifffile.imwrite(f'temp_H{hexp:.1f}_A{A:.0f}_R{6}_Mis{mis:.2f}_Aniso{0.1}.tif',myfrac.frac_3D)
