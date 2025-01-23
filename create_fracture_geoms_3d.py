import pyvista as pv
# import vedo as vd
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hdf5storage import loadmat
import skimage.transform as skit
from skimage import img_as_bool
from scipy.ndimage import distance_transform_edt as edist
from matplotlib.patches import Circle
from scipy.spatial import distance
import poisson_disc as pd
from scipy.stats import qmc


def scale_geometry(geom, rescale_factor, data_type):

    # Rescale geometry
    geom = skit.rescale(geom, rescale_factor, anti_aliasing=False,
                         order=0)  # order=0 means nearest neighbor interpolation (keeps image binary)

    # Ensure image has 0 as pore space and 1 as grains
    geom = edist(geom)
    geom[geom==0] = 0
    geom[geom>0] = 1

    # Change to specified data type
    geom = geom.astype(data_type)

    return geom


def create_rough_fracture(scale):
    grains = loadmat('input/gumbo_fracture_1.mat')['bin']
    nx = 256
    ny = 256
    nz = 256

    rough_fracture_tmp = grains[75:185,:,120:130]
    # plt.figure(figsize=[4,3])
    # plt.imshow(rough_fracture_tmp, cmap='gray')
    # plt.show()

    rough_fracture = scale_geometry(rough_fracture_tmp, scale, data_type='int8')
    # plt.figure(figsize=[4,3])
    # plt.imshow(rough_fracture, cmap='gray')
    # plt.show()

    return rough_fracture


def create_smooth_fracture(scale):

    # smooth_fracture_tmp = np.ones([56, 256, 10])
    # smooth_fracture_tmp[6:50, :,:] = 0
    smooth_fracture_tmp = np.ones([110, 256, 10])
    smooth_fracture_tmp[33:77, :,:] = 0

    # plt.figure(figsize=[4,3])
    # plt.imshow(smooth_fracture_tmp, cmap='gray')
    # plt.show()

    smooth_fracture = scale_geometry(smooth_fracture_tmp, scale, data_type='int8')
    # plt.figure(figsize=[4,3])
    # plt.imshow(smooth_fracture, cmap='gray')
    # plt.show()

    return smooth_fracture


def plot_bubbles(geom, xs, ys, zs, nx, ny, nz, bubble_radius, filen):

    # fig = plt.gcf()
    # ax = fig.gca()

    grid = pv.wrap(geom)
    pl = pv.Plotter()
    _ = pl.add_axes(line_width=5)
    
    c = (geom).reshape(1,-1)
    pl.add_volume(geom*255,scalars=c.astype(np.uint8), cmap='summer_r')
    
    for i in range(len(xs)):
        cs = (ys[i], xs[i], zs[i])
        bubble = pv.Sphere(radius=bubble_radius, center = cs)
        actor = pl.add_mesh(bubble, color='blue')

    pl.show()
    # pl.screenshot(filename=f'./3Dtest/{filen}_Initial_State.png',return_img=True)

    return


def create_bubble_coordinates(geom, bubble_radius, bubble_padding):
    
    nx = geom.shape[1] - 1
    ny = geom.shape[0] - 1
    nz = geom.shape[2] - 1
    # Poisson Disk Sampling (same method as LBfoam)
    poisson_r = np.round(bubble_radius*2 + bubble_padding, decimals=2)  # A minimum distance between samples, add bubble pad so that it's a bit more uniform after removal algorithm
    poisson_k = 60
    radius = poisson_r/220 # Normalize in fracture aperture
    engine = qmc.PoissonDisk(d=3, radius=radius, ncandidates=poisson_k)
    sample = engine.random(500)
    x_points_all = (165+220*sample[:,0]).astype('int')
    y_points_all = (1280*sample[:,1]).astype('int')
    z_points_all = (50 * sample[:,2]).astype('int')
    
    # plt.title('All points')
    # plot_bubbles(geom, x_points_all, y_points_all, z_points_all, nx, ny, nz)


    # Remove points where bubble overlaps or is inside matrix
    geom[geom==1] = 2
    geom[geom==0] = 1
    geom[geom==2] = 0
    geom_edist = edist(geom)

    distance_mask = bubble_radius + bubble_padding
    bubble_index_overlap = np.where(geom_edist[y_points_all, x_points_all, z_points_all] < distance_mask)[0]
    x_points = np.delete(x_points_all, bubble_index_overlap)
    y_points = np.delete(y_points_all, bubble_index_overlap)
    z_points = np.delete(z_points_all, bubble_index_overlap)

    
    # plt.title('Remove bubbles overlapping matrix')
    # plot_bubbles(geom, x_points, y_points, z_points, nx, ny, nz)
    # plt.gca().invert_yaxis()
    

    return x_points, y_points, z_points

def geom_edist_for_foam(geom):

    # Flip pores and grains to get reverse distance map
    geom[geom == 1] = 2
    geom[geom == 0] = 1
    geom[geom == 2] = 0
    geom_edist = edist(geom)
    # plt.figure()
    # plt.imshow(geom_edist)
    # plt.show()

    return geom_edist


def create_bubble_placement_figure(smooth_geom, rough_geom, bubble_radius, bubble_pad):

    # Create points
    smooth_geom = np.rot90(smooth_geom)
    rough_geom = np.rot90(rough_geom)
    
    # bubble_r = int(12)
    # # bubble_pad = 3
    # poisson_radius = np.round(bubble_r*1.6 + bubble_pad, decimals=2)  # A minimum distance between samples, add bubble pad so that it's a bit more uniform after removal algorithm
    # poisson_k = 60  # This is the number of trials of uniform random sampling the code uses to place a point amongst the others without overlapping
    
    smooth_x_points, smooth_y_points, smooth_z_points = \
        create_bubble_coordinates(geom=smooth_geom, bubble_radius=bubble_radius, bubble_padding=bubble_pad)
    rough_x_points, rough_y_points, rough_z_points = \
        create_bubble_coordinates(geom=rough_geom, bubble_radius=bubble_radius, bubble_padding=bubble_pad)

    nx = smooth_geom.shape[1]
    ny = smooth_geom.shape[0]
    nz = smooth_geom.shape[2]

    smooth_edist = geom_edist_for_foam(smooth_geom)
    rough_edist = geom_edist_for_foam(rough_geom)
    

    # Change plot font formatting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })

    fontsize_ticks = 6
    fontsize_title = 8
    fontsize_suptitle = 17

    plt.figure()   # figsize=[4,5]
    plt.subplot(1,2,1)
    strsmooth = 'Smooth_Fracture'
    plt.title('Smooth Fracture,\n Remove Bubbles in Matrix \n k = {poisson_k}, r = {poisson_r} voxels', fontsize=fontsize_title)
    plot_bubbles(smooth_geom, smooth_x_points, smooth_y_points, smooth_z_points, nx, ny, nz, bubble_radius, filen = strsmooth)
    # plt.gca().invert_yaxis()
    
    plt.subplot(1,2,2)
    strrough = 'Rough_Fracture'
    plt.title('Rough Fracture,\n Remove Bubbles in Matrix \n k = {poisson_k}, r = {poisson_r} voxels', fontsize=fontsize_title)
    plot_bubbles(rough_geom, rough_x_points, rough_y_points, rough_z_points, nx, ny, nz, bubble_radius, filen = strrough)
    # plt.gca().invert_yaxis()

    # plt.subplots_adjust(wspace=0, hspace=0.4)
    # plt.tight_layout()
    # plt.savefig('bubble_placement_example.png', dpi=500)

    # plt.show()

    return


bubble_r = int(12)
bubble_pad = 3

smooth_fracture = create_smooth_fracture(scale=5)
smooth_fracture = np.rot90(smooth_fracture)
print(smooth_fracture.shape)
smooth_fracture.tofile(f'input/smooth_fracture_horizontal550.raw')

# Add padding so no bubbles overlap inlet or outlet
smooth_fracture[:,0,:] = 1
smooth_fracture[:,-1,:] = 1
smooth_fracture[0,:,:] = 1
smooth_fracture[-1,:,:] = 1
smooth_fracture[:,:,0] = 1
smooth_fracture[:,:,-1] = 1


smooth_fracture = np.rot90(smooth_fracture)
smooth_x_points, smooth_y_points, smooth_z_points= create_bubble_coordinates(geom=smooth_fracture, bubble_radius=bubble_r, bubble_padding=bubble_pad)
np.savetxt('3Dtest/smooth_nucleation_x_points.txt', smooth_x_points, fmt='%i')
np.savetxt('3Dtest/smooth_nucleation_y_points.txt', smooth_y_points, fmt='%i')
np.savetxt('3Dtest/smooth_nucleation_z_points.txt', smooth_z_points, fmt='%i')


# rough_fracture = create_rough_fracture(scale=5)
# rough_fracture.tofile(f'input/gumbo_fracture_2.raw')

# Add padding so no bubbles overlap inlet or outlet
# rough_fracture[:,0,:] = 1
# rough_fracture[:,-1,:] = 1
# rough_fracture[0,:,:] = 1
# rough_fracture[-1,:,:] = 1
# rough_fracture[:,:,0] = 1
# rough_fracture[:,:,-1] = 1


# rough_fracture = np.rot90(rough_fracture)
# rough_x_points, rough_y_points, rough_z_points = create_bubble_coordinates(geom=rough_fracture, bubble_radius=bubble_r, bubble_padding=bubble_pad)
# np.savetxt('3Dtest/rough_nucleation_x_points.txt', rough_x_points, fmt='%i')
# np.savetxt('3Dtest/rough_nucleation_y_points.txt', rough_y_points, fmt='%i')
# np.savetxt('3Dtest/rough_nucleation_z_points.txt', rough_z_points, fmt='%i')


# create_bubble_placement_figure(smooth_geom=smooth_fracture, rough_geom=rough_fracture,
#                                 bubble_radius=bubble_r, bubble_pad=bubble_pad)