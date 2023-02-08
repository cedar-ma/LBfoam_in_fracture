import pyvista as pv
import vedo as vd
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label


####################################
# New stuff that's cool with foam! #
####################################

def find_nearest(array, value):

    array = np.asarray(array)
    ind = (np.abs(array - value)).argmin()

    return array[ind], ind


def calculate_quality_over_sim(nx, ny, tmp_folder, geom_file, data_savename, skip=10):
    # Foam quality calculations
    foam_files_regex = fr'{tmp_folder}volumeData*.vti'
    foam_files = glob.glob(foam_files_regex)

    # Sort lists for correct order
    foam_list = sorted(foam_files)
    print(len(foam_list))

    quality = np.array([])
    # skip = 10
    for i in range(0, len(foam_list), skip):

        foam_mesh = pv.read(foam_list[i])
        print(foam_list[i])

        # print(foam_mesh.array_names)
        # ['velocity', 'pressure', 'adDensity', 'volumeFraction', 'smoothedVolumeFraction', 'bubbleTags', 'disjoiningPressure']
        foam_vof = foam_mesh.get_array('volumeFraction').reshape([ny, nx, 1])

        fracture = np.fromfile(geom_file, dtype=np.int8).reshape([nx, ny, 1])
        fracture = fracture.transpose([1, 0, 2])

        foam_vof_calc = np.where(fracture==0, foam_vof, -1)

        fracture_voxels = len(np.where(foam_vof_calc>=0)[0])
        # lamella_voxels = len(np.where(foam_vof_calc>0)[0])
        gas_voxels = len(np.where(foam_vof_calc==0)[0])
        quality = np.append(quality, gas_voxels/fracture_voxels)
        print(quality)

    np.save(data_savename, quality)

    return


def plot_foam_quality_comparison(nx, ny, nrows, ncols, save_name, tmp_folder_list,
                                 geom_file_list, image_index_list, titles_list, super_title=''):

    # Total number of images to visualize
    nframes = nrows * ncols

    fontsize_ticks = 6
    fontsize_title = 6
    fontsize_suptitle = 17

    plt.figure()  # figsize=[12/1.25, 8/1.25]
    # fig.gridspec_kw={'width_ratios': 3}
    for i in range(nframes):

        foam_files_regex = fr'{tmp_folder_list[i]}volumeData*.vti'
        foam_files = glob.glob(foam_files_regex)

        # Sort lists for correct order
        foam_list = sorted(foam_files)

        # Load foam data
        foam_mesh = pv.read(foam_list[image_index_list[i]])

        # print(foam_mesh.array_names)
        # ['velocity', 'pressure', 'adDensity', 'volumeFraction', 'smoothedVolumeFraction', 'bubbleTags', 'disjoiningPressure']
        foam_vof = foam_mesh.get_array('volumeFraction').reshape([ny, nx, 1])

        # Load geometry
        # geom_file = "input/smooth_fracture.raw"
        # geom_file = "input/gumbo_fracture.raw"
        fracture = np.fromfile(geom_file_list[i], dtype=np.int8).reshape([nx, ny, 1])
        fracture = fracture.transpose([1, 0, 2])

        # Plotting setup
        x = np.arange(0, fracture.shape[1], 1)
        y = np.arange(0, fracture.shape[0], 1)
        X, Y = np.meshgrid(x, y)
        fracture2d = fracture[:, :, 0]

        # Change plot font formatting
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Palatino"],
        })

        # Plotting
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(foam_vof[:, :, 0], cmap='Oranges', alpha=1, vmin=0, vmax=1)
        cbar = plt.colorbar()
        cbar.set_label(r'Liquid Volume Fraction', size=fontsize_ticks)
        cbar.ax.tick_params(labelsize=fontsize_ticks)
        plt.contourf(X, Y, fracture2d, levels=[0.5, 1], alpha=1, colors='gray')
        plt.gca().invert_yaxis()
        plt.title(titles_list[i], fontsize=fontsize_title)
        plt.xticks([])
        plt.yticks([])

    # plt.suptitle(super_title, fontsize=fontsize_suptitle)
    plt.subplots_adjust(wspace=0, hspace=0.4)
    plt.tight_layout()
    plt.savefig(save_name, dpi=500)
    # plt.show()

    return


def plot_simulation_quality(smoothdata, smooth_point_chosen_index, roughdata, rough_point_chosen_index, total_iter, savename, x_spacing):

    # Change plot font formatting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })

    plt.figure(figsize=[12,5])
    iterations = np.linspace(1, total_iter, len(smoothdata))
    plt.subplot(1,2,1)
    from matplotlib.ticker import StrMethodFormatter
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.yticks(np.arange(0, 1, 0.05))
    plt.xticks(np.arange(0, total_iter+1, x_spacing))
    plt.plot(iterations, smoothdata, 'bo-', label='Simulation Data', markersize=2)
    plt.title("Smooth Fracture Foam Quality During Simulation")
    plt.xlabel('Iterations')
    plt.ylabel('Foam Quality')
    plt.grid()
    plt.ylim([0, 1])
    for i in range(len(smooth_point_chosen_index)):
        plt.plot(iterations[smooth_point_chosen_index[i]], smoothdata[smooth_point_chosen_index[i]], 'ko')
    plt.legend(['Simulation Data', 'Points Chosen for Comparison'])

    plt.subplot(1,2,2)
    iterations = np.linspace(1, total_iter, len(roughdata))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.yticks(np.arange(0, 1, 0.05))
    plt.xticks(np.arange(0, total_iter+1, x_spacing))
    plt.plot(iterations, roughdata, 'go-', label='Simulation Data', markersize=2)
    plt.title("Rough Fracture Foam Quality During Simulation")
    plt.xlabel('Iterations')
    plt.ylabel('Foam Quality')
    plt.grid()
    plt.ylim([0,1])
    for i in range(len(smooth_point_chosen_index)):
        plt.plot(iterations[rough_point_chosen_index[i]], roughdata[rough_point_chosen_index[i]], 'ko')

    plt.legend(['Simulation Data', 'Points Chosen for Comparison'])

    plt.tight_layout()

    plt.savefig(savename, dpi=500)

    return


ny = 1280
nx = 550

sim_folder = '/oden/gigliotti/Desktop/ccgo2/storage/gigliotti/LBfoam/LBfoam/examples/lbfoam/foam_in_fractures_2D'

tmp_folder_list = [f'{sim_folder}/tmp_smooth2/',
                   f'{sim_folder}/tmp_rough2/']

geom_file_list = ["input/smooth_fracture.raw",
                  "input/gumbo_fracture.raw"]

# calculate_quality_over_sim(nx, ny, tmp_folder=tmp_folder_list[0], geom_file=geom_file_list[0],
#                            data_savename='smooth_fracture_data_2.npy', skip=1)
# #
# calculate_quality_over_sim(nx, ny, tmp_folder=tmp_folder_list[1], geom_file=geom_file_list[1],
#                            data_savename='rough_fracture_data_2.npy', skip=1)

# Simulation Quality Numerical Plot #
quality_data_smooth = np.load('smooth_fracture_data_2.npy')
quality_data_rough = np.load('rough_fracture_data_2.npy')

smooth_q_60, smooth_ind_60 = find_nearest(quality_data_smooth, 0.6)
smooth_q_70, smooth_ind_70 = find_nearest(quality_data_smooth, 0.7)
smooth_q_80, smooth_ind_80 = find_nearest(quality_data_smooth, 0.8)

rough_q_60, rough_ind_60 = find_nearest(quality_data_rough, 0.6)
rough_q_70, rough_ind_70 = find_nearest(quality_data_rough, 0.7)
rough_q_80, rough_ind_80 = find_nearest(quality_data_rough, 0.8)


plot_simulation_quality(smoothdata=quality_data_smooth,
                        smooth_point_chosen_index=[smooth_ind_60, smooth_ind_70, smooth_ind_80],
                        roughdata=quality_data_rough,
                        rough_point_chosen_index=[rough_ind_60, rough_ind_70, rough_ind_80],
                        total_iter=200000, savename='simulation_quality.png', x_spacing=20000)

# Quality simulation visualization #
nrows = 2
ncols = 4
ndec = 1  # number of decimals in title
iter_interval = 1000
titles_list = [f'Smooth Fracture IC,\n Iterations: {1},\n Quality = {np.round(quality_data_smooth[0]*100, decimals=ndec)}\%',
               f'Smooth Fracture,\n Iterations: {smooth_ind_60*iter_interval},\n Quality = {np.round(smooth_q_60*100, decimals=ndec)}\%',
               f'Smooth Fracture,\n Iterations: {smooth_ind_70*iter_interval},\n Quality = {np.round(smooth_q_70*100, decimals=ndec)}\%',
               f'Smooth Fracture,\n Iterations: {smooth_ind_80*iter_interval},\n Quality = {np.round(smooth_q_80*100, decimals=ndec)}\%',
               f'Rough Fracture IC,\n Iterations: {1},\n Quality = {np.round(quality_data_rough[0]*100, decimals=ndec)}\%',
               f'Rough Fracture,\n Iterations: {rough_ind_60*iter_interval},\n Quality = {np.round(rough_q_60*100, decimals=ndec)}\%',
               f'Rough Fracture,\n Iterations: {rough_ind_70*iter_interval},\n Quality = {np.round(rough_q_70*100, decimals=ndec)}\%',
               f'Rough Fracture,\n Iterations: {rough_ind_80*iter_interval},\n Quality = {np.round(rough_q_80*100, decimals=ndec)}\%']

save_name = 'foam_simulation_quality_geom_comparison.png'

tmp_folder_list = [f'{sim_folder}/tmp_smooth2/', f'{sim_folder}/tmp_smooth2/', f'{sim_folder}/tmp_smooth2/', f'{sim_folder}/tmp_smooth2/',
                   f'{sim_folder}/tmp_rough2/', f'{sim_folder}/tmp_rough2/', f'{sim_folder}/tmp_rough2/', f'{sim_folder}/tmp_rough2/']

geom_file_list = ["input/smooth_fracture.raw", "input/smooth_fracture.raw", "input/smooth_fracture.raw", "input/smooth_fracture.raw",
                  "input/gumbo_fracture.raw", "input/gumbo_fracture.raw", "input/gumbo_fracture.raw", "input/gumbo_fracture.raw"]

image_index_list = [0, smooth_ind_60, smooth_ind_70, smooth_ind_80,
                    0, rough_ind_60, rough_ind_70, rough_ind_80]
plot_foam_quality_comparison(nx, ny, nrows, ncols, save_name, tmp_folder_list,
                             geom_file_list, image_index_list, titles_list,
                             super_title='Fracture Geometry and Foam Quality Comparison')
# plt.show()





# which_sim = 0
# index = -1
# foam_files_regex = fr'{tmp_folder_list[which_sim]}volumeData*.vti'
# foam_files = glob.glob(foam_files_regex)
#
# # Sort lists for correct order
# foam_list = sorted(foam_files)
# foam_mesh = pv.read(foam_list[index])
# print(foam_list[index])
#
# # print(foam_mesh.array_names)
# # ['velocity', 'pressure', 'adDensity', 'volumeFraction', 'smoothedVolumeFraction', 'bubbleTags', 'disjoiningPressure']
# foam_vf = foam_mesh.get_array('disjoiningPressure').reshape([ny, nx, 1])
# plt.figure(figsize=[2,3])
# plt.imshow(foam_vf, cmap='turbo')
# plt.colorbar()
#
# plt.savefig('disjoingPtest.png',dpi=300)

