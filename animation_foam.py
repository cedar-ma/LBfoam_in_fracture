#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 08:32:44 2023

@author: cedar
"""
import imageio.v2 as imageio
import os
import glob
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import re

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def create_gif(tmp_folder):    
    images = []
    
    # foam_files_regex = fr'{tmp_folder}gas*.gif'
    foam_files_regex = fr'{tmp_folder}smooth*.gif'
    filenames = glob.glob(foam_files_regex)
    
    filenames = natural_sort(filenames)
    print(len(filenames))
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(f'{tmp_folder}1volumeFrac.gif', images)
    
    print('done!')
    return 

def find_nearest(array, value):

    array = np.asarray(array)
    ind = (np.abs(array - value)).argmin()

    return array[ind], ind

def calculate_quality_over_sim(nx, ny, nz, tmp_folder, geom_file, data_savename, skip=10):
    # Foam quality calculations
    foam_files_regex = fr'{tmp_folder}volumeData*.vti'
    foam_files = glob.glob(foam_files_regex)

    # Sort lists for correct order
    foam_list = sorted(foam_files)
    print(len(foam_list))

    quality = np.array([])
    # skip = 10
    for i in range(1, len(foam_list), skip):

        foam_mesh = pv.read(foam_list[i])
        print(foam_list[i])

        # print(foam_mesh.array_names)
        # ['velocity', 'pressure', 'adDensity', 'volumeFraction', 'smoothedVolumeFraction', 'bubbleTags', 'disjoiningPressure']
        foam_vof = foam_mesh.get_array('volumeFraction').reshape([nz, ny, nx])
        # plt.imshow(foam_vof[25, :, :])

        fracture = np.fromfile(geom_file, dtype=np.int8).reshape([nx, ny, nz])
        fracture = fracture.transpose([2, 1, 0])
        # plt.figure()
        # plt.imshow(fracture[25, :, :])

        foam_vof_calc = np.where(fracture==0, foam_vof, -1)
        

        fracture_voxels = len(np.where(foam_vof_calc>=0)[0])
        # lamella_voxels = len(np.where(foam_vof_calc>0)[0])
        gas_voxels = len(np.where(foam_vof_calc==0)[0])
        quality = np.append(quality, gas_voxels/fracture_voxels)
        print(quality)

    np.save(data_savename, quality)

    return

sim_folder = '/Users/cedar/Library/CloudStorage/Box-Box/Research_Cedar/LBFOAM/Qingdao_meeting/results/3D/1con_5k/ca-1/tau95_force1e4/sigma2e3/'
# sim_folder = '/Users/cedar/Library/CloudStorage/Box-Box/Research_Cedar/LBFOAM/Qingdao_meeting/results/bubble_size_dis/'
os.chdir('/Users/cedar/Library/CloudStorage/Box-Box/Research_Cedar/LBFOAM/Qingdao_meeting/results')
# print(os.getcwd())
# tmp_folder = f'{sim_folder}/grain_veri/geo_rad_ini/'
# tmp_folder = sim_folder
# tmp_folder_list = 'pool/channel/3barrier/1bubble/small_bubble_100k/tau9573/force1e4_c16e3/'
# tmp_folder = [f'{sim_folder}/tmp/nsmooth/', f'{sim_folder}/tmp/nrough/']
tmp_folder_list = 'pool/channel/verification/multi/28_restart/coord/pi38_force2e5/'
# tmp_folder_list = 'pool/channel/2bubble/10k/pi45_force5e6/4e3/'
# tmp_folder_list = 'pool/channel/2bubble/10k/10k_tau1573_pi52_force3e5/8e3/'
create_gif(tmp_folder_list)
#%%
import moviepy.editor as mp
print("Creating MP4...")

gif = mp.VideoFileClip(f'{tmp_folder_list}1volumeFrac.gif')
gif = gif.speedx(factor=0.5)
gif.write_videofile(f'{tmp_folder_list}1volumeFrac.mp4')

print("Done!")
#%%

nx = 550
ny = 450
nz = 50

geom_file_list = ["input/smooth_fracture.raw",
                  "input/gumbo_fracture.raw"]

# calculate_quality_over_sim(nx, ny, nz, tmp_folder=tmp_folder[0], geom_file=geom_file_list[0],
                            # data_savename='gif/smooth_fracture_data.npy', skip=1)

# calculate_quality_over_sim(nx, ny, nz, tmp_folder=tmp_folder[1], geom_file=geom_file_list[1],
#                             data_savename='gif/rough_fracture_data.npy', skip=1)

# Simulation Quality Numerical Plot #
# quality_data_smooth = np.load('gif/smooth_fracture_data.npy')
# quality_data_rough = np.load('gif/rough_fracture_data.npy')

# smooth_q_63, smooth_ind_63 = find_nearest(quality_data_smooth, 0.06)
# print(smooth_q_63, smooth_ind_63 + 1)
# smooth_q_70, smooth_ind_70 = find_nearest(quality_data_smooth, 0.7)
# smooth_q_80, smooth_ind_80 = find_nearest(quality_data_smooth, 0.8)

# rough_q_63, rough_ind_63 = find_nearest(quality_data_rough, 0.63)
# print(rough_q_63, rough_ind_63 + 1)
# rough_q_60, rough_ind_60 = find_nearest(quality_data_rough, 0.6)
# rough_q_70, rough_ind_70 = find_nearest(quality_data_rough, 0.7)
# rough_q_80, rough_ind_80 = find_nearest(quality_data_rough, 0.8)