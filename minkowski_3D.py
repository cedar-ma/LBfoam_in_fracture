#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:56:44 2023

@author: cedar
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import  (disk)
from skimage.morphology import footprints
from quantimpy import morphology as mp
from quantimpy import minkowski as mk
import glob
import pyvista as pv
from matplotlib.ticker import StrMethodFormatter
from skimage.morphology import (ball)


# image = np.zeros([128,128,128],dtype=bool)
# image[16:113,16:113,16:113] = ball(48,dtype=bool)

# plt.gray()
# plt.imshow(image[:,:,64])
# plt.colorbar()
# plt.show()

# volume, surface, curv, euler = mk.functionals(image)
# print(f'{volume = }')
# print(f'surface = {8*surface}')
# print(f'curvature = {2*np.pi*np.pi*curv/8/surface}')
# print(f'euler = {4/3*np.pi*euler}')

# Compute Minkowski functionals for image with anisotropic resolution



#%%

def calculate_Ca(nx, ny, tmp_folder, geom_file_list, skip=1):
    foam_files_regex = fr'{tmp_folder}volumeData*.vti'
    foam_files = glob.glob(foam_files_regex)

    # Sort lists for correct order
    foam_list = sorted(foam_files)
    print(len(foam_list))
    x = np.linspace(0,nx,nx)
    y = np.linspace(0,ny,ny)
    X, Y = np.meshgrid(x,y)
    plt.figure()
    Ca = np.array([])
    for i in range(1, len(foam_list), skip):
        fracture = np.fromfile(geom_file_list, dtype=np.int8).reshape([nx, ny, 1])
        fracture = fracture.transpose([2, 1, 0])

        # Plotting setup
        m = np.arange(0, fracture.shape[2], 1)
        n = np.arange(0, fracture.shape[1], 1)
        M, N = np.meshgrid(m,n)
        fracture2d = fracture[0, :, :]
        
        foam_mesh = pv.read(foam_list[i])
        print(foam_list[i])
        foam_vof = foam_mesh.get_array('volumeFraction').reshape([1, ny, nx])
        ux = foam_mesh.get_array('velocity')[:,0]
        ux = ux.reshape([ny, nx])
        uy = foam_mesh.get_array('velocity')[:,1]
        uy = uy.reshape([ny, nx])
        norm = np.sqrt(ux**2+uy**2)
        norm_ave = np.mean(norm)
        Cat = (0.35*norm_ave)/3./5e-3
        Ca = np.append(Ca, Cat)
   
    return Ca

def calculate_minkowski_over_sim(nx, ny, nz, tmp_folder, geom_file,  skip=1):
    # Foam quality calculations
    foam_files_regex = fr'{tmp_folder}volumeData*.vti'
    foam_files = glob.glob(foam_files_regex)

    # Sort lists for correct order
    foam_list = sorted(foam_files)
    print(len(foam_list))

    volume = np.array([])
    surface = np.array([])
    curv = np.array([])
    euler = np.array([])
    # skip = 10
    for i in range(0, len(foam_list), skip):

        foam_mesh = pv.read(foam_list[i])
        print(foam_list[i])

        # print(foam_mesh.array_names)
        # ['velocity', 'pressure', 'adDensity', 'volumeFraction', 'smoothedVolumeFraction', 'bubbleTags', 'disjoiningPressure']
        foam_vof = foam_mesh.get_array('volumeFraction').reshape([nz, ny, nx])

        fracture = np.fromfile(geom_file, dtype=np.int8).reshape([nx, ny, nz])
        fracture = fracture.transpose([2, 1, 0])

        foam_vof_calc = np.where(fracture==0, foam_vof, -100) #if inclde interface, change -1 to 100
        
        foam_vof_min = np.where(foam_vof_calc==0 , 1, 0) #if include interface, <1
        plt.gray()
        plt.imshow(foam_vof_min[6,:,:])
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
        plt.show()
        
        ivolume, isurface, icurv, ieuler = mk.functionals(foam_vof_min.astype(bool))
        volume = np.append(volume, ivolume)
        surface = np.append(surface, isurface)
        curv = np.append(curv, icurv)
        euler = np.append(euler, ieuler)

        # fracture_voxels = len(np.where(foam_vof_calc>=0)[0])
        # # lamella_voxels = len(np.where(foam_vof_calc>0)[0])
        # gas_voxels = len(np.where(foam_vof_calc==0)[0])
        # quality = np.append(quality, gas_voxels/fracture_voxels)
    print(f'{volume = }')
    print(f'surface = {8*surface}')
    print(f'curvature = {2*np.pi*np.pi*curv}')
    print(f'euler = {4/3*np.pi*euler}')

    np.savez(f'{tmp_folder}/data.npz', v=volume, s=8*surface, c=2*np.pi*np.pi*curv, u=4/3*np.pi*euler)

    return volume, 8*surface, 2*np.pi*np.pi*curv, 4/3*np.pi*euler
def plot_simulation_minkowski_tauD(areadata, areadata1,areadata2, \
                                  lengthdata, lengthdata1, lengthdata2,\
                                eulerdata, eulerdata1, eulerdata2, total_iter0, total_iter1, total_iter2,  x_spacing):

    # Change plot font formatting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })
    plt.rc('font', size = 12)
    plt.rc('legend', fontsize = 18)
    plt.rc('figure', titlesize = 24)
    plt.rc('axes', labelsize = 20)

    plt.figure(figsize=[12, 10])
    iterations = np.linspace(11500, total_iter0, len(areadata))
    iterations1 = np.linspace(0, total_iter1, len(areadata1))
    iterations2 = np.linspace(11500, total_iter2, len(areadata2))
    # iterations3 = np.linspace(1, total_iter, len(areadata3))
   
    plt.subplot(221)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    plt.xticks(np.arange(0, 30001, x_spacing))
    plt.plot(iterations, areadata, 'k-', label=r'$\tau_D$ = 0.53', linewidth=2)
    plt.plot(iterations1, areadata1, 'g-', label=r'$\tau_D$ = 0.51', linewidth=2)
    plt.plot(iterations2, areadata2, 'r-', label=r'$\tau_D$ = 0.505', linewidth=2)
    # plt.plot(iterations3, areadata3, 'b-', label=r'$\tau$  = 1.3', linewidth=2)
    # plt.plot(iterations, lengthdata, 'bo-', label='Bubble Perimeter', markersize=2)
    # plt.plot(iterations, eulerdata, 'ro-', label='Euler Charateristic', markersize=2)
    plt.xlabel('Iterations')
    plt.ylabel('Bubble Volume')
    plt.xlim([0, 30000])
    # plt.ylim([1,7])
    # plt.grid()
    plt.legend()
    
    # plt.subplot(222)
    # # iterations = np.linspace(1, total_iter, len(lengthdata))
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, total_iter0+1, x_spacing))
    # plt.plot(iterations, Car, 'k-', label=r'rough fracture', linewidth=2)
    # plt.plot(iterations1, Cas, 'g-', label=r'smooth fracture', linewidth=2)
    # # plt.plot(iterations2, lengthdata2, 'r-', label=r'$\tau$ = 1.1', linewidth=2)
    # # plt.plot(iterations3, lengthdata3, 'b-', label=r'$\tau$  = 1.3', linewidth=2)
    # plt.xlabel('Iterations')
    # plt.ylabel('Capillary Number (-)')
    # plt.legend()
    
    plt.subplot(222)
    # iterations = np.linspace(1, total_iter, len(lengthdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    plt.xticks(np.arange(0, 30001, x_spacing))
    plt.plot(iterations, (lengthdata/areadata), 'k-', label=r'$\tau_D$ = 0.53', linewidth=2)
    plt.plot(iterations1, (lengthdata1/areadata1), 'g-', label=r'$\tau_D$ = 0.51', linewidth=2)
    plt.plot(iterations2, lengthdata2/areadata2, 'r-', label=r'$\tau_D$ = 0.505', linewidth=2)
    # plt.plot(iterations3, lengthdata3, 'b-', label=r'$\tau$  = 1.3', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Average curvature')
    plt.xlim([0, 30000])
    # plt.ylim([0.3, 1])
    plt.legend()
    
    plt.subplot(223)
    # iterations = np.linspace(1, total_iter, len(eulerdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    plt.xticks(np.arange(0, 30001, x_spacing))
    plt.plot(iterations, eulerdata, 'k-', label=r'$\tau_D$ = 0.53', linewidth=2)
    plt.plot(iterations1, eulerdata1, 'g-', label=r'$\tau_D$ = 0.51', linewidth=2)
    plt.plot(iterations2, eulerdata2, 'r-', label=r'$\tau_D$ = 0.505', linewidth=2)
    # plt.plot(iterations3, eulerdata3, 'b-', label=r'$\tau$ = 1.3', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Euler Charateristic')
    plt.xlim([0, 30000])
    # plt.ylim([0.8,1.2])
    plt.legend()
    # plt.ylim([0,1])
    # for i in range(len(smooth_point_chosen_index)):
    #     plt.plot(iterations[rough_point_chosen_index[i]], roughdata[rough_point_chosen_index[i]], 'ko')
    plt.subplot(224)
    # iterations = np.linspace(1, total_iter, len(eulerdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    plt.xticks(np.arange(0, 30001, x_spacing))
    # plt.plot(iterations, lengthdata/(2*np.pi*405.30797078), 'k-', label=r'rough fracture', linewidth=2)
    # plt.plot(iterations1, lengthdata1/(2*np.pi*525.01569031), 'g-', label=r'smooth fracture', linewidth=2)
    plt.plot(iterations, (4*np.pi*areadata/((lengthdata)**2)), 'k-', label=r'$\tau_D$ = 0.53', linewidth=2)
    plt.plot(iterations1, (4*np.pi*areadata1/((lengthdata1)**2)), 'g-', label=r'$\tau_D$ = 0.51', linewidth=2)
    plt.plot(iterations2, (4*np.pi*areadata2/((lengthdata2)**2)), 'r-', label=r'$\tau_D$ = 0.505', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Roundness')
    plt.xlim([0, 30000])
    # plt.ylim([0.6,1.1])
    plt.legend()
    
    # plt.suptitle("The influence of geometry on Minkowski parameters")
    plt.tight_layout()
    

    plt.savefig(f'{sim_folder}/../gif/sensitivity/manuscript/tauD.jpg', dpi=800)
    plt.show()

    return



def plot_simulation_minkowski_geometry(areadata, areadata1, \
                                  lengthdata, lengthdata1,\
                                eulerdata, eulerdata1, total_iter0, total_iter1,  x_spacing):

    # Change plot font formatting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })
    plt.rc('font', size = 12)
    plt.rc('legend', fontsize = 18)
    plt.rc('figure', titlesize = 24)
    plt.rc('axes', labelsize = 20)

    plt.figure(figsize=[15, 5])
    # plt.figure(figsize=[12, 10])
    iterations = np.linspace(0, total_iter0, len(areadata))
    iterations1 = np.linspace(0, total_iter1, len(areadata1))
    # iterations2 = np.linspace(1, total_iter, len(areadata2))
    # iterations3 = np.linspace(1, total_iter, len(areadata3))
   
    # plt.subplot(221)
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, total_iter0+1001, x_spacing))
    # plt.plot(iterations, areadata/10856., 'k-', label=r'rough fracture', linewidth=2)
    # plt.plot(iterations1, areadata1/14052., 'g-', label=r'smooth fracture', linewidth=2)
    # # plt.plot(iterations2, areadata2, 'r-', label=r'$\tau$ = 1.1', linewidth=2)
    # # plt.plot(iterations3, areadata3, 'b-', label=r'$\tau$  = 1.3', linewidth=2)
    # # plt.plot(iterations, lengthdata, 'bo-', label='Bubble Perimeter', markersize=2)
    # # plt.plot(iterations, eulerdata, 'ro-', label='Euler Charateristic', markersize=2)
    # plt.xlabel('Iterations')
    # plt.ylabel('Normalized Bubble Volume')
    # plt.xlim([0, 30000])
    # plt.ylim([1,7])
    # # plt.grid()
    # plt.legend()
    
    # plt.subplot(222)
    # # iterations = np.linspace(1, total_iter, len(lengthdata))
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, total_iter0+1, x_spacing))
    # plt.plot(iterations, Car, 'k-', label=r'rough fracture', linewidth=2)
    # plt.plot(iterations1, Cas, 'g-', label=r'smooth fracture', linewidth=2)
    # # plt.plot(iterations2, lengthdata2, 'r-', label=r'$\tau$ = 1.1', linewidth=2)
    # # plt.plot(iterations3, lengthdata3, 'b-', label=r'$\tau$  = 1.3', linewidth=2)
    # plt.xlabel('Iterations')
    # plt.ylabel('Capillary Number (-)')
    # plt.legend()
    
    plt.subplot(131)
    # iterations = np.linspace(1, total_iter, len(lengthdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    plt.xticks(np.arange(0, total_iter0+1001, x_spacing))
    plt.plot(iterations, (lengthdata/areadata)/(2*np.pi*405.30797078/10856.), 'k-', label=r'rough fracture', linewidth=2)
    plt.plot(iterations1, (lengthdata1/areadata1)/(2*np.pi*525.01569031/14052.), 'g-', label=r'smooth fracture', linewidth=2)
    # plt.plot(iterations2, lengthdata2, 'r-', label=r'$\tau$ = 1.1', linewidth=2)
    # plt.plot(iterations3, lengthdata3, 'b-', label=r'$\tau$  = 1.3', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Normalized Average curvature')
    plt.xlim([0, 30000])
    plt.ylim([0.3, 1])
    plt.legend()
    
    plt.subplot(132)
    # iterations = np.linspace(1, total_iter, len(eulerdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    plt.xticks(np.arange(0, total_iter0+1001, x_spacing))
    plt.plot(iterations, eulerdata/(np.pi*14.64225476), 'k-', label=r'rough fracture', linewidth=2)
    plt.plot(iterations1, eulerdata1/(np.pi*18.78028328), 'g-', label=r'smooth fracture', linewidth=2)
    # plt.plot(iterations2, eulerdata2, 'r-', label=r'$\tau$ = 1.1', linewidth=2)
    # plt.plot(iterations3, eulerdata3, 'b-', label=r'$\tau$ = 1.3', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Normalized Euler Charateristic')
    plt.xlim([0, 30000])
    plt.ylim([0.8,1.2])
    plt.legend()
    # plt.ylim([0,1])
    # for i in range(len(smooth_point_chosen_index)):
    #     plt.plot(iterations[rough_point_chosen_index[i]], roughdata[rough_point_chosen_index[i]], 'ko')
    plt.subplot(133)
    # iterations = np.linspace(1, total_iter, len(eulerdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    plt.xticks(np.arange(0, total_iter0+1001, x_spacing))
    # plt.plot(iterations, lengthdata/(2*np.pi*405.30797078), 'k-', label=r'rough fracture', linewidth=2)
    # plt.plot(iterations1, lengthdata1/(2*np.pi*525.01569031), 'g-', label=r'smooth fracture', linewidth=2)
    plt.plot(iterations, (4*np.pi*areadata/((lengthdata)**2))/0.021035346617207427, 'k-', label=r'rough fracture', linewidth=2)
    plt.plot(iterations1, (4*np.pi*areadata1/((lengthdata1)**2))/0.01622720426775893, 'g-', label=r'smooth fracture', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Normalized Roundness')
    plt.xlim([0, 30000])
    plt.ylim([0.6,1.1])
    plt.legend()
    
    # plt.suptitle("The influence of geometry on Minkowski parameters")
    plt.tight_layout()
    

    plt.savefig(f'{sim_folder}/../gif/sensitivity/manuscript/geometry_3.jpg', dpi=800)
    plt.show()

    return

def plot_simulation_minkowski_tau(areadata, areadata1, areadata2, \
                                  lengthdata, lengthdata1,lengthdata2, \
                                      curv, curv1, curv2,\
                                eulerdata, eulerdata1,eulerdata2,  \
                                total_iter0, total_iter1, total_iter2, x_spacing):

    # Change plot font formatting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })
    plt.rc('font', size = 16)
    plt.rc('legend', fontsize = 18)
    plt.rc('figure', titlesize = 24)
    plt.rc('axes', labelsize = 22)

    plt.figure(figsize=[12, 10])
    iterations = np.linspace(0, total_iter0, len(areadata))
    iterations1 = np.linspace(0, total_iter1, len(areadata1))
    iterations2 = np.linspace(0, total_iter2, len(areadata2))
    # iterations3 = np.linspace(0, total_iter3, len(areadata3))
   
    plt.subplot(2,2,1)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, 35000+1, x_spacing))
    plt.plot(iterations, areadata, 'g-', label=r'$\tau$ = 1.15', linewidth=3)
    plt.plot(iterations1, areadata1, 'r-', label=r'$\tau$ = 0.95', linewidth=3)
    plt.plot(iterations2, areadata2, 'b-', label=r'$\tau$ = 0.75', linewidth=3)
    # plt.plot(iterations3, areadata3, 'b-', label=r'$\tau$  = 1.3', linewidth=2)
    # plt.plot(iterations, lengthdata, 'bo-', label='Bubble Perimeter', markersize=2)
    # plt.plot(iterations, eulerdata, 'ro-', label='Euler Charateristic', markersize=2)
    plt.xlabel('Iterations')
    plt.ylabel('Bubble Volume [$lu^3$]')
    plt.xlim([0,total_iter0])
    plt.ylim([0,90000])
    # plt.grid()
    plt.legend()
    
    plt.subplot(2,2,2)
    # iterations = np.linspace(1, total_iter, len(lengthdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, total_iter3+1, x_spacing))
    plt.plot(iterations, lengthdata, 'g-', label=r'$\tau$ = 1.15', linewidth=3)
    plt.plot(iterations1, lengthdata1, 'r-', label=r'$\tau$ = 0.95', linewidth=3)
    plt.plot(iterations2, lengthdata2, 'b-', label=r'$\tau$ = 0.75', linewidth=3)
    # plt.plot(iterations3, lengthdata3, 'b-', label=r'$\tau$  = 1.3', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Surface Area [$lu^2$]')
    plt.xlim([0, total_iter0])
    plt.ylim([0, 75000])
    plt.legend()
    
    plt.subplot(223)
    # iterations = np.linspace(1, total_iter, len(lengthdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, 35000+1, x_spacing))
    plt.plot(iterations, curv, 'g-', label=r'$\tau$ = 1.15', linewidth=3)
    plt.plot(iterations1, curv1, 'r-', label=r'$\tau$ = 0.95', linewidth=3)
    plt.plot(iterations2, curv2, 'b-', label=r'$\tau$ = 0.75', linewidth=3)
    # plt.plot(iterations3, lengthdata3/areadata3, 'b-', label=r'$\tau$  = 1.3', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Integral Mean Curvature [$lu$]')
    plt.xlim([0,total_iter0])
    plt.ylim([0,3500])
    plt.legend()
    
    plt.subplot(224)
    # iterations = np.linspace(1, total_iter, len(eulerdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, 35000+1, x_spacing))
    plt.plot(iterations, eulerdata, 'g-', label=r'$\tau$ = 1.15', linewidth=3)
    plt.plot(iterations1, eulerdata1, 'r-', label=r'$\tau$ = 0.95', linewidth=3)
    plt.plot(iterations2, eulerdata2, 'b-', label=r'$\tau$ = 0.75', linewidth=3)
    # plt.plot(iterations3, eulerdata3, 'b-', label=r'$\tau$ = 1.3', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Euler Charateristic [-]')
    plt.xlim([0,total_iter0])
    plt.ylim([-10,20])
    plt.legend()
    # plt.ylim([0,1])
    # for i in range(len(smooth_point_chosen_index)):
    #     plt.plot(iterations[rough_point_chosen_index[i]], roughdata[rough_point_chosen_index[i]], 'ko')
    # plt.subplot(133)
    # # iterations = np.linspace(1, total_iter, len(eulerdata))
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, 35000+1, x_spacing))
    # plt.plot(iterations, 4*np.pi*areadata/(lengthdata**2), 'g-', label=r'$\tau$ = 0.7', linewidth=2)
    # plt.plot(iterations1, 4*np.pi*areadata1/(lengthdata1**2), 'r-', label=r'$\tau$ = 0.9', linewidth=2)
    # plt.plot(iterations2, 4*np.pi*areadata2/(lengthdata2**2), 'b-', label=r'$\tau$ = 1.1', linewidth=2)
    # # plt.plot(iterations3, 4*np.pi*areadata3/(lengthdata3**2), 'b-', label=r'$\tau$ = 1.3', linewidth=2)
    # plt.xlabel('Iterations')
    # plt.ylabel('Roundness (-)')
    # plt.xlim([0,35000])
    # plt.ylim([0,0.02])
    # plt.legend()
    
    # plt.suptitle("The influence of liquid viscosity on Minkowski parameters")
    plt.tight_layout()
    

    plt.savefig(f'{sim_folder1}/../../figure/3D/Minkowski_tau.jpg', dpi=500)
    plt.show()

    return

def plot_simulation_minkowski_T(areadata, areadata1, areadata2,areadata3,\
                                lengthdata, lengthdata1, lengthdata2,lengthdata3,\
                                eulerdata, eulerdata1, eulerdata2,eulerdata3, \
                                total_iter0, total_iter1, total_iter2, total_iter3,  x_spacing):

    # Change plot font formatting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })
    plt.rc('font', size = 16)
    plt.rc('legend', fontsize = 18)
    plt.rc('figure', titlesize = 24)
    plt.rc('axes', labelsize = 22)

    plt.figure(figsize=[18, 5])
    iterations = np.linspace(0, total_iter0,  len(areadata))
    iterations1 = np.linspace(0, total_iter1, len(areadata1))
    iterations2 = np.linspace(0, total_iter2, len(areadata2))
    iterations3 = np.linspace(0, total_iter3, len(areadata3))
    
    # plt.subplot(221)
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, 30000+1, x_spacing))
    # plt.plot(iterations, areadata, 'k-', label='T = 0.7', linewidth=2)
    # plt.plot(iterations1, areadata1, 'g-', label='T = 1.0', linewidth=2)
    # plt.plot(iterations2, areadata2, 'r-', label='T = 1.3', linewidth=2)
    # plt.plot(iterations3, areadata3, 'b-', label='T = 1.5', linewidth=2)
    # # plt.plot(iterations, lengthdata, 'bo-', label='Bubble Perimeter', markersize=2)
    # # plt.plot(iterations, eulerdata, 'ro-', label='Euler Charateristic', markersize=2)
    # plt.xlabel('Iterations')
    # plt.ylabel('Bubble Volume ($lu^2$)')
    # plt.xlim([0,30000])
    # plt.ylim([0,70000])
    # # plt.grid()
    # plt.legend()
    
    # plt.subplot(232)
    # # iterations = np.linspace(1, total_iter, len(lengthdata))
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, total_iter3+1, x_spacing))
    # plt.plot(iterations, lengthdata, 'k-', label='T = 0.7', linewidth=2)
    # plt.plot(iterations1, lengthdata1, 'g-', label='T = 1.0', linewidth=2)
    # plt.plot(iterations2, lengthdata2, 'r-', label='T = 1.3', linewidth=2)
    # plt.plot(iterations3, lengthdata3, 'b-', label='T = 1.5', linewidth=2)
    # plt.xlabel('Iterations')
    # plt.ylabel('Bubble Perimeter (voxels)')
    # plt.legend()
    
    plt.subplot(131)
    # iterations = np.linspace(1, total_iter, len(lengthdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    plt.xticks(np.arange(0, 30000+1, x_spacing))
    plt.plot(iterations, lengthdata/areadata, 'k-', label='T = 0.7', linewidth=2)
    plt.plot(iterations1, lengthdata1/areadata1, 'g-', label='T = 1.0', linewidth=2)
    plt.plot(iterations2, lengthdata2/areadata2, 'r-', label='T = 1.3', linewidth=2)
    plt.plot(iterations3, lengthdata3/areadata3, 'b-', label='T = 1.5', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Integral curvature ($1/lu$)')
    plt.xlim([0,30000])
    plt.ylim([0.0, 0.16])
    plt.legend()
    
    plt.subplot(132)
    # iterations = np.linspace(1, total_iter, len(eulerdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    plt.xticks(np.arange(0, 30000+1, x_spacing))
    plt.plot(iterations, eulerdata, 'k-', label='T = 0.7', linewidth=2)
    plt.plot(iterations1, eulerdata1, 'g-', label='T = 1.0', linewidth=2)
    plt.plot(iterations2, eulerdata2, 'r-', label='T = 1.3', linewidth=2)
    plt.plot(iterations3, eulerdata3, 'b-', label='T = 1.5', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Euler Charateristic (-)')
    plt.xlim([0,30000])
    plt.ylim([0,60])
    plt.legend()
    # plt.ylim([0,1])
    # for i in range(len(smooth_point_chosen_index)):
    #     plt.plot(iterations[rough_point_chosen_index[i]], roughdata[rough_point_chosen_index[i]], 'ko')
    plt.subplot(133)
    # iterations = np.linspace(1, total_iter, len(eulerdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    plt.xticks(np.arange(0, 30000+1, x_spacing))
    plt.plot(iterations, 4*np.pi*areadata/(lengthdata**2), 'k-', label='T = 0.7', linewidth=2)
    plt.plot(iterations1, 4*np.pi*areadata1/(lengthdata1**2), 'g-', label='T = 1.0', linewidth=2)
    plt.plot(iterations2, 4*np.pi*areadata2/(lengthdata2**2), 'r-', label='T = 1.3', linewidth=2)
    plt.plot(iterations3, 4*np.pi*areadata3/(lengthdata3**2), 'b-', label='T = 1.5', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Roundness (-)')
    plt.xlim([0,30000])
    plt.ylim([0.0,0.05])
    plt.legend()
    
    # plt.suptitle("The influence of temperature on Minkowski parameters")
    plt.tight_layout()
    

    plt.savefig(f'{sim_folder}/../gif/sensitivity/manuscript/T3.jpg', dpi=800)
    plt.show()

    return



def plot_simulation_minkowski_sigma(areadata, areadata1, areadata2, \
                                    lengthdata, lengthdata1, lengthdata2, \
                                        curv, curv1, curv2,\
                                    eulerdata, eulerdata1, eulerdata2,\
                                    total_iter0,total_iter1,total_iter2, x_spacing):

    # Change plot font formatting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })
    plt.rc('font', size = 16)
    plt.rc('legend', fontsize = 18)
    plt.rc('figure', titlesize = 24)
    plt.rc('axes', labelsize = 22)

    plt.figure(figsize=[12, 10])
    iterations = np.linspace(0, total_iter0, len(areadata))
    iterations1 = np.linspace(0, total_iter1, len(areadata1))
    iterations2 = np.linspace(0, total_iter2, len(areadata2))
    
    plt.subplot(221)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, 35000+1, x_spacing))
    plt.plot(iterations, areadata, 'g-', label=r'$\sigma = 2e^{-3}$', linewidth=3)
    plt.plot(iterations1, areadata1, 'r-', label=r'$\sigma = 4e^{-3}$', linewidth=3)
    plt.plot(iterations2, areadata2, 'b-', label=r'$\sigma = 8e^{-3}$', linewidth=3)
    # plt.plot(iterations, lengthdata, 'bo-', label='Bubble Perimeter', markersize=2)
    # plt.plot(iterations, eulerdata, 'ro-', label='Euler Charateristic', markersize=2)
    plt.xlabel('Iterations')
    plt.ylabel('Bubble Volume [$lu^3$]')
    # plt.grid()
    plt.xlim([0,total_iter0])
    plt.ylim([0,90000])
    plt.legend()
    
    plt.subplot(222)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, 35000+1, x_spacing))
    plt.plot(iterations, lengthdata, 'g-', label=r'$\sigma = 2e^{-3}$', linewidth=3)
    plt.plot(iterations1, lengthdata1, 'r-', label=r'$\sigma = 4e^{-3}$', linewidth=3)
    plt.plot(iterations2, lengthdata2, 'b-', label=r'$\sigma = 8e^{-3}$', linewidth=3)
    # plt.plot(iterations, lengthdata, 'bo-', label='Bubble Perimeter', markersize=2)
    # plt.plot(iterations, eulerdata, 'ro-', label='Euler Charateristic', markersize=2)
    plt.xlabel('Iterations')
    plt.ylabel('Surface Area [$lu^2$]')
    # plt.grid()
    plt.xlim([0,total_iter0])
    plt.ylim([0,70000])
    plt.legend()
    
    plt.subplot(223)
    # iterations = np.linspace(1, total_iter, len(lengthdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, 35000+1, x_spacing))
    plt.plot(iterations, lengthdata/areadata, 'g-', label=r'$\sigma = 2e^{-3}$', linewidth=3)
    plt.plot(iterations1, lengthdata1/areadata1, 'r-', label=r'$\sigma = 4e^{-3}$', linewidth=3)
    plt.plot(iterations2, lengthdata2/areadata2, 'b-', label=r'$\sigma = 8e^{-3}$', linewidth=3)
    plt.xlabel('Iterations')
    plt.ylabel('Integral Mean Curvature [$lu$]')
    plt.xlim([0,total_iter0])
    plt.ylim([0.5,3.0])
    plt.legend()
    
    plt.subplot(224)
    # iterations = np.linspace(1, total_iter, len(eulerdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, 35000+1, x_spacing))
    plt.plot(iterations, eulerdata, 'g-', label=r'$\sigma = 2e^{-3}$', linewidth=3)
    plt.plot(iterations1, eulerdata1, 'r-', label=r'$\sigma = 4e^{-3}$', linewidth=3)
    plt.plot(iterations2, eulerdata2, 'b-', label=r'$\sigma = 8e^{-3}$', linewidth=3)
    plt.xlabel('Iterations')
    plt.ylabel('Euler Charateristic [-]')
    plt.xlim([0,total_iter0])
    plt.ylim([-40,30])
    plt.legend()
    # plt.ylim([0,1])
    # for i in range(len(smooth_point_chosen_index)):
    #     plt.plot(iterations[rough_point_chosen_index[i]], roughdata[rough_point_chosen_index[i]], 'ko')

    # plt.subplot(133)
    # # iterations = np.linspace(1, total_iter, len(eulerdata))
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, 35000+1, x_spacing))
    # plt.plot(iterations, 4*np.pi*areadata/(lengthdata**2), 'g-', label=r'$\sigma = 2e^{-3}$', linewidth=2)
    # plt.plot(iterations1, 4*np.pi*areadata1/(lengthdata1**2), 'r-', label=r'$\sigma = 4e^{-3}$', linewidth=2)
    # plt.plot(iterations2, 4*np.pi*areadata2/(lengthdata2**2), 'b-', label=r'$\sigma = 8e^{-3}$', linewidth=2)
    # plt.xlabel('Iterations')
    # plt.ylabel('Roundness (-)')
    # plt.xlim([0,35000])
    # plt.ylim([0,0.02])
    # plt.legend()
    # plt.suptitle("The influence of Surface Tension on Minkowski Parameters During Simulation")
    plt.tight_layout()
    

    # plt.savefig(f'{sim_folder1}/../../figure/3D/Minkowski_sigma.jpg', dpi=800)
    plt.show()

    return

def plot_simulation_minkowski_nabla(areadata, areadata1, areadata2,\
                                    lengthdata, lengthdata1, lengthdata2,\
                                        curv, curv1, curv2,\
                                    eulerdata, eulerdata1, eulerdata2,\
                                    total_iter0,total_iter1, total_iter2,x_spacing):

    # Change plot font formatting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })
    plt.rc('font', size = 16)
    plt.rc('legend', fontsize = 18)
    plt.rc('figure', titlesize = 20)
    plt.rc('axes', labelsize = 22)

    plt.figure(figsize=[12, 10])
    iterations = np.linspace(0, total_iter0, len(areadata))
    iterations1 = np.linspace(0, total_iter1, len(areadata1))
    iterations2 = np.linspace(0, total_iter2, len(areadata2))
    # iterations3 = np.linspace(0, total_iter3, len(areadata3))
    plt.subplot(2,2,1)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # # plt.yticks(np.arange(0, 1, 0.05))
    # ax[0].xticks(np.arange(0, total_iter0+1, x_spacing))
    plt.plot(iterations, areadata, 'g-', label=r'$\Delta p= 1.5e^{-4}$', linewidth=3)
    plt.plot(iterations1, areadata1, 'r-', label=r'$\Delta p= 1.0e^{-4}$', linewidth=3)
    plt.plot(iterations2, areadata2, 'b-', label=r'$\Delta p= 5.0e^{-5}$', linewidth=3)
    # # plt.plot(iterations2, areadata2, 'b-', label=r'$\Delta p$ = 7.5e-6', linewidth=2)
    # # plt.plot(iterations, lengthdata, 'bo-', label='Bubble Perimeter', markersize=2)
    # # plt.plot(iterations, eulerdata, 'ro-', label='Euler Charateristic', markersize=2)
    plt.xlabel('Iterations')
    plt.ylabel('Bubble Volume [$lu^3$]')
    plt.xlim([0,total_iter0])
    plt.ylim([0,40000])
    # # plt.grid()
    plt.legend()
    
    plt.subplot(2,2,2)
    # # iterations = np.linspace(1, total_iter, len(lengthdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, total_iter0+1, x_spacing))
    plt.plot(iterations, lengthdata, 'g-', label=r'$\Delta p= 1.5e^{-4}$', linewidth=3)
    plt.plot(iterations1, lengthdata1, 'r-', label=r'$\Delta p= 1.0e^{-4}$', linewidth=3)
    plt.plot(iterations2, lengthdata2, 'b-', label=r'$\Delta p= 5.0e^{-5}$', linewidth=3)
    plt.xlabel('Iterations')
    plt.ylabel('Surface Area [$lu^2$]')
    plt.xlim([0,total_iter0])
    plt.ylim([0, 60000])
    plt.legend()
    
    plt.subplot(223)
    # iterations = np.linspace(1, total_iter, len(lengthdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, total_iter0+1, x_spacing))
    plt.plot(iterations, lengthdata/areadata, 'g-', label=r'$\Delta p= 1.5e^{-4}$', linewidth=3)
    plt.plot(iterations1, lengthdata1/areadata1, 'r-', label=r'$\Delta p= 1.0e^{-4}$', linewidth=3)
    plt.plot(iterations2, lengthdata2/areadata2, 'b-', label=r'$\Delta p= 5.0e^{-5}$', linewidth=3)
    # plt.plot(iterations3, lengthdata3/areadata3, 'b-', label=r'$\Delta p$= 7.5e-6', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Integral Mean Curvature [$lu$]')
    plt.xlim([0,total_iter0])
    plt.ylim([0.0,3.0])
    plt.legend()
    
    plt.subplot(224)
    # iterations = np.linspace(1, total_iter, len(eulerdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, total_iter0+1, x_spacing))
    plt.plot(iterations, eulerdata, 'g-', label=r'$\Delta p= 1.5e^{-4}$', linewidth=3)
    plt.plot(iterations1, eulerdata1, 'r-', label=r'$\Delta p= 1.0e^{-4}$', linewidth=3)
    plt.plot(iterations2, eulerdata2, 'b-', label=r'$\Delta p= 5.0e^{-5}$', linewidth=3)
    # plt.plot(iterations3, eulerdata3, 'b-', label=r'$\Delta p$= 7.5e-6', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Euler Charateristic [-]')
    plt.xlim([0,total_iter0])
    plt.ylim([-20,30])
    plt.legend()
    
    # plt.subplot(133)
    # # iterations = np.linspace(1, total_iter, len(eulerdata))
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, 35000+1, x_spacing))
    # plt.plot(iterations, 4*np.pi*areadata/(lengthdata**2), 'g-', label=r'$\Delta p$= 6.3e-6', linewidth=2)
    # plt.plot(iterations1, 4*np.pi*areadata1/(lengthdata1**2), 'r-', label=r'$\Delta p$= 6.7e-6', linewidth=2)
    # # plt.plot(iterations2, 4*np.pi*areadata2/(lengthdata2**2), 'b-', label=r'$\Delta p$= 7.5e-6', linewidth=2)
    # # plt.plot(iterations3, 4*np.pi*areadata3/(lengthdata3**2), 'b-', label=r'$\Delta p$= 7.5e-6', linewidth=2)
    # plt.xlabel('Iterations')
    # plt.ylabel('Roundness (-)')
    # plt.xlim([0,35000])
    # plt.ylim([0,0.02])
    # plt.legend()

    
    # plt.suptitle("The influence of Surface Tension on Minkowski Parameters During Simulation")
    plt.tight_layout()
    

    plt.savefig(f'{sim_folder1}/../../figure/3D/Minkowski_nabla.jpg', dpi=800)
    plt.show()

    return

def plot_simulation_minkowski_kpi(areadata, areadata1, areadata2,\
                                lengthdata, lengthdata1, lengthdata2,\
                                    curv, curv1, curv2,
                                eulerdata, eulerdata1, eulerdata2,\
                                total_iter0,total_iter1, total_iter2, x_spacing):

    # Change plot font formatting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })
    plt.rc('font', size = 14)
    plt.rc('legend', fontsize = 18)
    plt.rc('figure', titlesize = 24)
    plt.rc('axes', labelsize = 20)

    plt.figure(figsize=[12, 10])
    iterations = np.linspace(0, total_iter0, len(areadata))
    iterations1 = np.linspace(0, total_iter1, len(areadata1))
    iterations2 = np.linspace(0, total_iter2, len(areadata2))
    
    plt.subplot(2,2,1)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, total_iter0+1, x_spacing))
    

    plt.plot(iterations, areadata, 'g-', label=r'k = 0.020', linewidth=3)
    plt.plot(iterations1, areadata1, 'r-', label=r'k = 0.032', linewidth=3)
    plt.plot(iterations2, areadata2, 'b-', label=r'k = 0.048', linewidth=3)
    # plt.plot(iterations, lengthdata, 'bo-', label='Bubble Perimeter', markersize=2)
    # plt.plot(iterations, eulerdata, 'ro-', label='Euler Charateristic', markersize=2)
    plt.xlabel('Iterations')
    plt.ylabel('Bubble Volume [$lu^3$]')
    plt.xlim([0, total_iter0])
    plt.ylim([0,130000])
    # plt.grid()
    plt.legend()
    
    plt.subplot(2,2,2)
    # iterations = np.linspace(1, total_iter, len(lengthdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, total_iter0+1, x_spacing))
    
    plt.plot(iterations, lengthdata, 'g-', label=r'k = 0.020', linewidth=3)
    plt.plot(iterations1, lengthdata1, 'r-', label=r'k = 0.032', linewidth=3)
    plt.plot(iterations2, lengthdata2, 'b-', label=r'k = 0.048', linewidth=3)
    plt.xlabel('Iterations')
    plt.xlim([0, total_iter0])
    plt.ylim([0,70000])
    plt.ylabel('Surface Area [$lu^2$]')
    plt.legend()
    
    plt.subplot(2,2,3)
    # iterations = np.linspace(1, total_iter, len(eulerdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, total_iter0+1, x_spacing))
    
    plt.plot(iterations, curv, 'g-', label=r'k = 0.020', linewidth=3)
    plt.plot(iterations1, curv1, 'r-', label=r'k = 0.032', linewidth=3)
    plt.plot(iterations2, curv2, 'b-', label=r'k = 0.048', linewidth=3)
    plt.xlabel('Iterations')
    plt.xlim([0,total_iter0])
    # plt.ylim([0,1.5e19])
    plt.ylabel('Integral Mean Curvature [lu]')
    plt.legend()
    
    plt.subplot(2,2,4)
    # iterations = np.linspace(1, total_iter, len(eulerdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, total_iter0+1, x_spacing))
    
    plt.plot(iterations, eulerdata, 'g-', label=r'k = 0.020', linewidth=3)
    plt.plot(iterations1, eulerdata1, 'r-', label=r'k = 0.032', linewidth=3)
    plt.plot(iterations2, eulerdata2, 'b-', label=r'k = 0.048', linewidth=3)
    plt.xlabel('Iterations')
    plt.xlim([0,total_iter0])
    plt.ylim([-60,10])
    plt.ylabel('Euler Charateristic [-]')
    plt.legend()
    
    # plt.subplot(1,4,4)
    # # iterations = np.linspace(1, total_iter, len(eulerdata))
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    # # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, total_iter0+1, x_spacing))
    # plt.plot(iterations, 4*np.pi*areadata/(lengthdata**2), 'g-', label=r'$\pi$ = 0.027', linewidth=2)
    # plt.plot(iterations1, 4*np.pi*areadata1/(lengthdata1**2), 'r-', label=r'$\pi$ = 0.030', linewidth=2)
    # # plt.plot(iterations2, 4*np.pi*areadata2/(lengthdata2**2), 'b-', label=r'$\pi$ = 0.032', linewidth=2)
    # plt.xlabel('Iterations')
    # plt.ylabel('Roundness (-)')
    # plt.legend()
    # plt.ylim([0,1])
    # for i in range(len(smooth_point_chosen_index)):
    #     plt.plot(iterations[rough_point_chosen_index[i]], roughdata[rough_point_chosen_index[i]], 'ko')

    
    # plt.suptitle("The Influence of Disjoining Pressure on Minkowski Parameters During Simulation")
    plt.tight_layout()
    

    # plt.savefig(f'{sim_folder1}/../../figure/3D/Minkowski_kpi.jpg', dpi=500)
    plt.show()

    return


nz = 13
ny = 120
nx = 400

# sim_folder = os.getcwd()
geom_folder = '/Users/cedar/data/lbfoamtest/LBfoam/examples/lbfoam/LBfoam_in_fracture_local/input'
sim_folder1 = '/Users/cedar/Library/CloudStorage/Box-Box/Research_Cedar/LBFOAM/Qingdao_meeting/results/3D/1con_5k'
# tmp_folder_list = [f'{sim_folder}/kpi/sr27/', f'{sim_folder}/kpi/sr30/']
# tmp_folder_list = [f'{sim_folder}/tau/r07/',f'{sim_folder}/tau/r09/',f'{sim_folder}/tau/r11/',f'{sim_folder}/tau/r13/']
# tmp_folder_list =[f'{sim_folder}/taun/7/',f'{sim_folder}/taun/9/',f'{sim_folder}/taun/11/']
# tmp_folder_list = [f'{sim_folder}/force/sr63/',f'{sim_folder}/force/sr67/',f'{sim_folder}/force/sr75/']
# tmp_folder_list = [f'{sim_folder}/sigma/sr2e-3/',f'{sim_folder}/sigma/sr5e-3/',f'{sim_folder}/sigma/sr8e-3/']
# tmp_folder_list = [f'{sim_folder}/T/r07/',f'{sim_folder}/T/r10/',f'{sim_folder}/T/r13/',f'{sim_folder}/T/r15/']
# tmp_folder_list = [f'{sim_folder}/T/r10/',f'{sim_folder}/../smooth/83_r3/']
# tmp_folder_list = [f'{sim_folder1}/rough/',f'{sim_folder1}/smooth/']#, f'{sim_folder}/../smooth/smooth_tau53/',f'{sim_folder}/../smooth/smooth_tau51/',f'{sim_folder}/../smooth/smooth_tau505/' ]
tmp_folder_list = [f'{sim_folder1}/h1e3/force15e5/sigma2e3/',f'{sim_folder1}/h1e3/force15e5/sigma4e3/',f'{sim_folder1}/h1e3/force15e5/sigma8e3/', 
                  f'{sim_folder1}/h1e3/force5e5/sigma2e3/', f'{sim_folder1}/h1e3/force1e4/sigma2e3/',
                  f'{sim_folder1}/tau75_force2e4_ca0/dis20_sigma8e3/',f'{sim_folder1}/tau75_force2e4_ca0/sigma8e3/',f'{sim_folder1}/dis48_ca0/force2e4/sigma8e3/', f'{sim_folder1}/dis64_ca0/force2e4/sigma8e3/',
                  f'{sim_folder1}/ca-1/tau115_sigma2e3/', f'{sim_folder1}/ca-1/tau95_force1e4/sigma2e3/', f'{sim_folder1}/ca-1/tau75_force1e4/sigma2e3/']
# geom_file_list = [f"{geom_folder}/gumbo_fracture_horrizontal_dilate.raw",
#                   f"{geom_folder}/smooth_fracture_2D.raw"]
geom_file_list = [f'{geom_folder}/h13_1constriction_channel.raw', f'{geom_folder}/smooth_fracture_2Dheng.raw', f'{geom_folder}/gumbo_fracture_cal_combined.raw']
# Car = calculate_Ca(nx, ny, tmp_folder_list[0], geom_file_list[0])
# Cas = calculate_Ca(nx, ny, tmp_folder_list[1], geom_file_list[1])
# v,s,c,u = calculate_minkowski_over_sim(nx, ny, nz, tmp_folder_list[0], geom_file_list[0], skip=5)
# v1,s1,c1,u1 = calculate_minkowski_over_sim(nx, ny, nz, tmp_folder_list[1], geom_file_list[0], skip=1)
# v2,s2,c2,u2 = calculate_minkowski_over_sim(nx, ny, nz, tmp_folder_list[2], geom_file_list[0], skip=1)


data = np.load(f'{tmp_folder_list[9]}/data.npz')
data1 = np.load(f'{tmp_folder_list[10]}/data.npz')
data2 = np.load(f'{tmp_folder_list[11]}/data.npz')

plot_simulation_minkowski_tau(data['v'], data1['v'],data2['v'], data['s'], data1['s'],data2['s'], data['c'], data1['c'], data2['c'], data['u'], data1['u'], data2['u'], total_iter0=22700,total_iter1=19500,total_iter2=13000,x_spacing=5000)
# plot_simulation_minkowski_T(area, area1, area2,area3,length, length1, length2,length3, euler, euler1, euler2,euler3, total_iter0=26500,total_iter1=26500,total_iter2=26500, total_iter3=26500,x_spacing=5000)
# plot_simulation_minkowski_kpi(data['v'], data1['v'],data2['v'], data['s'], data1['s'],data2['s'], data['c'], data1['c'], data2['c'], data['u'], data1['u'], data2['u'], total_iter0=10200,total_iter1=10100, total_iter2=9600, x_spacing=5000)
# plot_simulation_minkowski_nabla(data['v'], data1['v'],data2['v'], data['s'], data1['s'],data2['s'], data['c'], data1['c'], data2['c'], data['u'], data1['u'], data2['u'],  total_iter0=6600,total_iter1=19900, total_iter2=17500, x_spacing=500)
# plot_simulation_minkowski_sigma(data['v'], data1['v'], data2['v'], data['s'], data1['s'], data2['s'], data['c'], data1['c'], data2['c'], data['u'], data1['u'],data2['u'], total_iter0=6600,total_iter1=13000,total_iter2=11700, x_spacing=500)
# plot_simulation_minkowski_tauD(area, area1, area2, length, length1, length2, euler, euler1,euler2, total_iter0=28500,total_iter1=28500, total_iter2=28500, x_spacing=500)
# plot_simulation_minkowski_geometry(area, area1, length, length1, euler, euler1, total_iter0=29000,total_iter1=25000, x_spacing=5000)