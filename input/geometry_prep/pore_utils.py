import numpy as np
import matplotlib.pyplot as plt
from skimage import measure


def create_geom_edist(rock, args):
    from scipy.ndimage.morphology import distance_transform_edt as edist
    
    if args.swapXZ:
        rock = rock.transpose([2, 1, 0])
        
    if args.scale_2:
        NotImplementedError('Feature not yet implemented')

    erock = edist(rock)

    # make sure all the BCs have BB nodes
    erock[0,:,:] = 1
    erock[:,0,:] = 1
    #erock[:,:,0] = 1
    
    erock[-1,:,:] = 1
    erock[:,-1,:] = 1
    #erock[:,:,-1] = 1

    import matplotlib.pyplot as plt
    plt.imshow(erock[:,:,0], cmap='turbo')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.show()
    
    # re open the pores
    erock[rock==0] = 0

    # Get the final matrix [0,1,2]
    erock[(erock>0)*(erock<2)] = 1
    erock[erock>1] = 2
    
    if args.add_mesh:
        NotImplementedError('Feature not yet implemented')
        
    if args.num_slices:
        erock = np.pad(erock, [(args.num_slices,args.num_slices), (0,0), (0,0)])

    if args.print_size:
        size = erock.shape
        geom_name = f'{args.name}_{size[0]}_{size[1]}_{size[2]}'
    else:
        geom_name = args.name

    # I don't understand why this works, but it does
    erock = erock.astype(np.int16)
    erock[erock==0] = 2608  # pore space
    erock[erock==1] = 2609  # boundary
    erock[erock==2] = 2610  # grains
    
    return erock, geom_name


def erase_regions(rock):
    # find connected-comps
    blobs_labels = measure.label(rock, background=1, connectivity=1)

    #vols = [np.sum(blobs_labels==label) for label in range(np.max(blobs_labels))]
    # it seems that label 1 is the largest comp, but gotta check

    # delete non-connected regions
    rock[blobs_labels>1] = 0
    
    return rock
