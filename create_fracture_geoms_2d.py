import pyvista as pv
import vedo as vd
import glob
import numpy as np
import matplotlib.pyplot as plt
from hdf5storage import loadmat
import skimage.transform as skit
from skimage import img_as_bool
from scipy.ndimage.morphology import distance_transform_edt as edist
from matplotlib.patches import Circle
from scipy.spatial import distance
import poisson_disc as pd


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

    rough_fracture_tmp = grains[75:185,:,128]
    # plt.figure(figsize=[4,3])
    # plt.imshow(rough_fracture_tmp, cmap='gray')
    # plt.show()

    rough_fracture = scale_geometry(rough_fracture_tmp, scale, data_type='int8')
    # plt.figure(figsize=[4,3])
    # plt.imshow(rough_fracture, cmap='gray')
    # plt.show()

    return rough_fracture


def create_smooth_fracture(scale):

    smooth_fracture_tmp = np.ones([110, 256])
    smooth_fracture_tmp[33:77, :] = 0

    # plt.figure(figsize=[4,3])
    # plt.imshow(smooth_fracture_tmp, cmap='gray')
    # plt.show()

    smooth_fracture = scale_geometry(smooth_fracture_tmp, scale, data_type='int8')
    # plt.figure(figsize=[4,3])
    # plt.imshow(smooth_fracture, cmap='gray')
    # plt.show()

    return smooth_fracture


def plot_bubbles(x_points, y_points, bubble_radius, bubble_color):

    fig = plt.gcf()
    ax = fig.gca()

    for i in range(len(x_points)):
        bubble = Circle((x_points[i], y_points[i]), radius=bubble_radius,
                        facecolor='None', edgecolor=bubble_color, alpha=0.8)
        ax.add_patch(bubble)

    return


def create_bubble_coordinates(geom, bubble_radius, bubble_padding):
    
    nx = geom.shape[1] - 1
    ny = geom.shape[0] - 1

    # Poisson Disk Sampling (same method as LBfoam)
    coords = pd.Bridson_sampling(dims=np.array([nx,ny]), radius=bubble_radius*1.6 + bubble_padding, k=60)
    x_points_all = coords[:,0].astype('int')
    y_points_all = coords[:,1].astype('int')

    plt.figure(figsize=[5,4])
    plt.title('All points')
    plt.imshow(geom, cmap='gray')
    plt.scatter(x_points_all, y_points_all, c='red', s=0.5)
    plot_bubbles(x_points_all, y_points_all, bubble_radius, bubble_color='red')
    plt.xlim([0, nx])
    plt.ylim([0, ny])

    # Remove points where bubble overlaps or is inside matrix
    geom[geom==1] = 2
    geom[geom==0] = 1
    geom[geom==2] = 0
    geom_edist = edist(geom)

    distance_mask = bubble_radius + bubble_padding
    bubble_index_overlap = np.where(geom_edist[y_points_all, x_points_all] < distance_mask)[0]
    x_points = np.delete(x_points_all, bubble_index_overlap)
    y_points = np.delete(y_points_all, bubble_index_overlap)

    plt.figure(figsize=[5,4])
    plt.title('Remove bubbles overlapping matrix')
    plt.imshow(geom_edist, cmap='turbo')
    plt.scatter(x_points, y_points, c='white', s=0.5)
    plot_bubbles(x_points, y_points, bubble_radius, bubble_color='white')
    plt.gca().invert_yaxis()
    plt.xlim([0, nx])
    plt.ylim([0, ny])

    return x_points, y_points, x_points_all, y_points_all

def geom_edist_for_foam(geom):

    # Flip pores and grains to get reverse distance map
    # geom[geom == 1] = 2
    # geom[geom == 0] = 1
    # geom[geom == 2] = 0
    geom_edist = edist(geom)
    # plt.figure()
    # plt.imshow(geom_edist)
    # plt.show()

    return geom_edist


def create_bubble_placement_figure(smooth_geom, rough_geom, bubble_radius, bubble_pad):

    # Create points
    smooth_geom = np.rot90(smooth_geom)
    rough_geom = np.rot90(rough_geom)
    smooth_x_points, smooth_y_points, smooth_x_points_all, smooth_y_points_all = \
        create_bubble_coordinates(geom=smooth_geom, bubble_radius=bubble_radius, bubble_padding=bubble_pad)
    rough_x_points, rough_y_points, rough_x_points_all, rough_y_points_all = \
        create_bubble_coordinates(geom=rough_geom, bubble_radius=bubble_radius, bubble_padding=bubble_pad)

    bubble_r = int(12)
    # bubble_pad = 3
    poisson_r = np.round(bubble_r*1.6 + bubble_pad, decimals=2)  # A minimum distance between samples, add bubble pad so that it's a bit more uniform after removal algorithm
    poisson_k = 60  # This is the number of trials of uniform random sampling the code uses to place a point amongst the others without overlapping

    plt.close('all')

    nx = smooth_geom.shape[1]
    ny = smooth_geom.shape[0]

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
    plt.title(f'Smooth Fracture Poisson Disk Sampling:\n k = {poisson_k}, r = {poisson_r} voxels', fontsize=fontsize_title)
    plt.imshow(smooth_geom, cmap='gray')
    plt.scatter(smooth_x_points_all, smooth_y_points_all, c='red', s=0.5, alpha=0.8)
    plot_bubbles(smooth_x_points_all, smooth_y_points_all, bubble_radius, bubble_color='red')
    plt.xlim([0, nx])
    plt.ylim([0, ny])
    # plt.xticks([])
    # plt.yticks([])
    plt.gca().invert_yaxis()

    plt.subplot(1,2,2)
    plt.title('Smooth Fracture,\n Remove Bubbles in Matrix', fontsize=fontsize_title)
    plt.imshow(smooth_edist, cmap='turbo')
    cbar = plt.colorbar()
    cbar.set_label(r'Euclidean Distance', size=fontsize_ticks)
    cbar.ax.tick_params(labelsize=fontsize_ticks)
    plt.scatter(smooth_x_points, smooth_y_points, c='white', s=0.5, alpha=0.8)
    plot_bubbles(smooth_x_points, smooth_y_points, bubble_radius, bubble_color='white')
    plt.xlim([0, nx])
    plt.ylim([0, ny])
    # plt.xticks([])
    # plt.yticks([])
    plt.gca().invert_yaxis()

    plt.subplots_adjust(wspace=0, hspace=0.4)
    plt.tight_layout()
    plt.savefig('bubble_placement_example_smooth.png', dpi=500)

    plt.figure()
    plt.subplot(1,2,1)
    plt.title(f'Rough Fracture Poisson Disk Sampling:\n k = {poisson_k}, r = {poisson_r} voxels', fontsize=fontsize_title)
    plt.imshow(rough_geom, cmap='gray')
    plt.scatter(rough_x_points_all, rough_y_points_all, c='red', s=0.5, alpha=0.8)
    plot_bubbles(rough_x_points_all, rough_y_points_all, bubble_radius, bubble_color='red')
    plt.xlim([0, nx])
    plt.ylim([0, ny])
    # plt.xticks([])
    # plt.yticks([])
    plt.gca().invert_yaxis()

    plt.subplot(1,2,2)
    plt.title('Rough Fracture,\n Remove Bubbles in Matrix', fontsize=fontsize_title)
    plt.imshow(rough_edist, cmap='turbo')
    cbar = plt.colorbar()
    cbar.set_label(r'Euclidean Distance', size=fontsize_ticks)
    cbar.ax.tick_params(labelsize=fontsize_ticks)
    plt.scatter(rough_x_points, rough_y_points, c='white', s=0.5, alpha=0.8)
    plot_bubbles(rough_x_points, rough_y_points, bubble_radius, bubble_color='white')
    plt.xlim([0, nx])
    plt.ylim([0, ny])
    # plt.xticks([])
    # plt.yticks([])
    plt.gca().invert_yaxis()

    plt.subplots_adjust(wspace=0, hspace=0.4)
    plt.tight_layout()
    plt.savefig('bubble_placement_example_rough.png', dpi=500)

    # plt.show()

    return


bubble_r = int(12)
bubble_pad = 3

smooth_fracture = create_smooth_fracture(scale=5)
# smooth_fracture.tofile(f'input/smooth_fracture.raw')

# Add padding so no bubbles overlap inlet or outlet
smooth_fracture[:,0] = 1
smooth_fracture[:,-1] = 1
smooth_fracture[0,:] = 1
smooth_fracture[-1,:] = 1
# smooth_x_points, smooth_y_points = create_bubble_coordinates(geom=smooth_fracture, bubble_radius=bubble_r, bubble_padding=bubble_pad)
# np.savetxt('tmp_smooth2/nucleation_y_points.txt', smooth_x_points, fmt='%i')
# np.savetxt('tmp_smooth2/nucleation_x_points.txt', smooth_y_points, fmt='%i')
# plt.tight_layout()
# plt.show()

rough_fracture = create_rough_fracture(scale=5)
# rough_fracture.tofile(f'input/gumbo_fracture.raw')

# Add padding so no bubbles overlap inlet or outlet
rough_fracture[:,0] = 1
rough_fracture[:,-1] = 1
rough_fracture[0,:] = 1
rough_fracture[-1,:] = 1
# rough_x_points, rough_y_points = create_bubble_coordinates(geom=rough_fracture, bubble_radius=bubble_r, bubble_padding=bubble_pad)
# np.savetxt('tmp_rough2/nucleation_y_points.txt', rough_x_points, fmt='%i')
# np.savetxt('tmp_rough2/nucleation_x_points.txt', rough_y_points, fmt='%i')
# plt.tight_layout()
# plt.show()

create_bubble_placement_figure(smooth_geom=smooth_fracture, rough_geom=rough_fracture,
                               bubble_radius=bubble_r, bubble_pad=bubble_pad)