import pyvista as pv
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
import os

def get_total_iter(tmp_folder_list):
    foam_files_regex = fr'{tmp_folder_list}volumeData*.vti'
    foam_files = glob.glob(foam_files_regex)

    # Sort lists for correct order
    foam_list = sorted(foam_files)
    # print(len(foam_list))
    titles = []
    for i in range(len(foam_list)):
        title_iter = foam_list[i][-12:-4]
        title_iter = title_iter.lstrip('0')
        titles.append(title_iter)
    return titles[-1]

def find_nearest(array, value):

    array = np.asarray(array)
    ind = (np.abs(array - value)).argmin()

    return array[ind], ind

def calculate_quality_over_sim(nx, ny, tmp_folder, geom_file, data_savename, skip=1):
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
        foam_vof = foam_mesh.get_array('volumeFraction').reshape([nz, ny, nx])

        fracture = np.fromfile(geom_file, dtype=np.int8).reshape([nz, ny, nx])
        # fracture = fracture.transpose([2, 0, 1])

        foam_vof_calc = np.where(fracture==0, foam_vof, 100)

        fracture_voxels = len(np.where(foam_vof_calc <= 1)[0])
        # lamella_voxels = len(np.where(foam_vof_calc>0)[0])
        gas_voxels = len(np.where(foam_vof_calc<1)[0])
        porosity = fracture_voxels/np.prod(fracture.shape)
        quality = np.append(quality, gas_voxels/fracture_voxels)
        # print(quality)

    np.save(data_savename, quality)

    plt.figure()
    plt.imshow(foam_vof_calc[0, :, :], vmin=0, vmax=1)
    plt.colorbar(label='Fracture', orientation='horizontal')
    plt.show()
    print(f'{porosity = } ')

    return

def plot_simulation_quality(smoothdata, smooth_point_chosen_index, savename, total_iter, x_spacing):

    # Change plot font formatting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })

    plt.figure(figsize=[6,5])
    iterations = np.linspace(1, total_iter, len(smoothdata))
    # plt.subplot(1,2,1)
    from matplotlib.ticker import StrMethodFormatter
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.yticks(np.arange(0, 1, 0.05))
    plt.xticks(np.arange(0, total_iter+1, x_spacing))
    plt.plot(iterations, smoothdata, 'b-', label='R8 Data', linewidth=2)
    plt.title("Bubble R=8 Foam Quality During Simulation")
    plt.xlabel('Iterations')
    plt.ylabel('Foam Quality')
    plt.grid()
    plt.ylim([0, 1])
    plt.xlim([0, total_iter+1])
    for i in range(len(smooth_point_chosen_index)):
        plt.plot(iterations[smooth_point_chosen_index[i]], smoothdata[smooth_point_chosen_index[i]], 'ko')
    plt.legend(['Simulation Data', 'Points Chosen for Prediction'])

    # plt.subplot(1,2,2)
    # iterations = np.linspace(1, total_iter, len(roughdata))
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    # plt.yticks(np.arange(0, 1, 0.05))
    # plt.xticks(np.arange(0, total_iter+1, x_spacing))
    # plt.plot(iterations, roughdata, 'go-', label='Simulation Data', markersize=2)
    # plt.title("Bubble R=12 Foam Quality During Simulation")
    # plt.xlabel('Iterations')
    # plt.ylabel('Foam Quality')
    # plt.grid()
    # plt.ylim([0,1])
    # plt.xlim([0, total_iter+1])
    # for i in range(len(smooth_point_chosen_index)):
    #     plt.plot(iterations[rough_point_chosen_index[i]], roughdata[rough_point_chosen_index[i]], 'ko')
    #
    # plt.legend(['Simulation Data', 'Points Chosen for Comparison'])
    #
    # plt.tight_layout()

    plt.savefig(savename, dpi=500)
    plt.show()

    return

def plot_foam_quality_comparison(nx, ny, nrows, ncols, save_name, tmp_folder_list,
                                 geom_file_list, image_index_list, titles_list, super_title=''):

    # Total number of images to visualize
    nframes = nrows * ncols

    fontsize_ticks = 6
    fontsize_title = 13
    fontsize_suptitle = 18

    plt.figure(figsize=[15,3])  # figsize=[12/1.25, 8/1.25]
    # fig.gridspec_kw={'width_ratios': 3}
    for i in range(nframes):

        foam_files_regex = fr'{tmp_folder_list}volumeData*.vti'
        foam_files = glob.glob(foam_files_regex)

        # Sort lists for correct order
        foam_list = sorted(foam_files)

        # Load foam data
        foam_mesh = pv.read(foam_list[image_index_list[i]])

        # print(foam_mesh.array_names)
        # ['velocity', 'pressure', 'adDensity', 'volumeFraction', 'smoothedVolumeFraction', 'bubbleTags', 'disjoiningPressure']
        foam_vof = foam_mesh.get_array('volumeFraction').reshape([ny, nx])

        # Load geometry
        # geom_file = "input/smooth_fracture.raw"
        # geom_file = "input/gumbo_fracture.raw"
        fracture = np.fromfile(geom_file_list, dtype=np.int8).reshape([ny, nx])
        # fracture = fracture.transpose([1, 0, 2])

        # Plotting setup
        x = np.arange(0, fracture.shape[1], 1)
        y = np.arange(0, fracture.shape[0], 1)
        X, Y = np.meshgrid(x, y)
        fracture2d = fracture[:, :]

        # Change plot font formatting
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Palatino"],
        })

        # Plotting
        plt.subplot(nrows, ncols, i + 1)
        voi = plt.imshow(foam_vof[:, :], cmap='Oranges', alpha=1, vmin=0, vmax=1)
        # cbar = plt.colorbar(orientation='horizontal')
        # cbar.set_label(r'Liquid Volume Fraction', size=fontsize_ticks)
        # cbar.ax.tick_params(labelsize=fontsize_ticks)
        plt.contourf(X, Y, fracture2d, levels=[0.5, 1], alpha=1, colors='gray')
        plt.gca().invert_yaxis()
        plt.title(titles_list[i], fontsize=fontsize_title)
        plt.xticks([])
        plt.yticks([])

    plt.suptitle(super_title, fontsize=fontsize_suptitle)
    plt.subplots_adjust(wspace=0, hspace=0.4)
    plt.tight_layout()
    plt.savefig(save_name, dpi=500)
    plt.show()

    return


os.chdir('/Users/cedar/Library/CloudStorage/Box-Box/Research_Cedar/LBFOAM/Qingdao_meeting/results')
geom_folder = '/Users/cedar/data/lbfoamtest/LBfoam/examples/lbfoam/LBfoam_in_fracture_local/input'
tmp_folder = [f'3D/pool_c002/',f'beat/r8/q5e5/', f'../double/diff100/pool/r4/', f'smooth/pool/x8/', f'r6/22smooth/pool3/', f'r6/smooth/pool_a/', f'r6/smooth/r6_pool_ori/', f'r6/smooth/pool4/', 'rough/roughpoolload/pool_rough/']
geom_file_list = [f"{geom_folder}/h13_pool_channel.raw", f"{geom_folder}/double_pool.raw", f"{geom_folder}/smooth_pool.raw", f"{geom_folder}/rough_pool.raw", f"{geom_folder}/rough_pool_100.raw"]

nz = 13
ny = 120
nx = 120
total_iter = 112000# int(get_total_iter(tmp_folder[0]))
x_spacing = 2000
# print(type(total_iter))

# Calculate foam quality values
# calculate_quality_over_sim(nx, ny, tmp_folder[0], geom_file_list[0], f'{tmp_folder[0]}/quality.npy', skip=1)

data = np.load(f'{tmp_folder[0]}/quality.npy')
print(data)
#
# smooth_q_60, smooth_ind_60 = find_nearest(data, 0.6)
# smooth_q_70, smooth_ind_70 = find_nearest(data, 0.7)
# smooth_q_80, smooth_ind_80 = find_nearest(data, 0.8)
#
# # Plot foam quality with respect to time steps
# # print(f'{data =}')
# plot_simulation_quality(data, [smooth_ind_60, smooth_ind_70, smooth_ind_80], f'{tmp_folder[0]}/Foam_quality_during_simulation.png', total_iter, x_spacing)
#
# # Show chosen vof figures
# nrows = 1
# ncols = 4
# ndec = 1  # number of decimals in title
# iter_interval = 1000
# save_name = f'{tmp_folder[0]}/Bubble_shape_for_prediction.png'
# image_index_list = [0, smooth_ind_60, smooth_ind_70, smooth_ind_80]
#
# titles_list = [f'IC\n Iterations: {1},\n Quality = {np.round(data[0]*100, decimals=ndec)}\%',
#                f'\n Iterations: {smooth_ind_60*iter_interval},\n Quality = {np.round(smooth_q_60*100, decimals=ndec)}\%',
#                f'\n Iterations: {smooth_ind_70*iter_interval},\n Quality = {np.round(smooth_q_70*100, decimals=ndec)}\%',
#                f'\n Iterations: {smooth_ind_80*iter_interval},\n Quality = {np.round(smooth_q_80*100, decimals=ndec)}\%']
#
# plot_foam_quality_comparison(nx, ny, nrows, ncols, save_name, tmp_folder[0],
#                              geom_file_list[0], image_index_list, titles_list,
#                              super_title='Bubble Shape and Foam Quality')