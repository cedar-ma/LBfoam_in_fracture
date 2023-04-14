Here is a quick rundown of how to use the LBFoam code. The workflow takes a few steps but shouldn't be too bad:

0) Compiling
    a) Clone the repo
        - https://github.com/mehdiataei/LBfoam
    b) Inside LBFoam repo directory, copy this folder into examples/lbfoam
    c) cd into the LBfoam_in_fracture directory and run 'make' in the terminal to compile
    d) LBFoam github also does a pretty good job at explaining compilation in the readme but no docs!

1) Geometry Prep
    a) The create_fracture_geoms_2d.py file can be used to create a smooth fracture and/or a fracture from a raw image (reusing MPLBM code for palabos). The bubble placement algorithm is also in there and will create the data needed for LBFoam. At the bottom of the file, all the function calls are there. Some commands are commented out, but you can uncomment/edit as needed.
    b) Outputs will be:
        - geomtry files (saved to 'input' directory)
        - bubble nucleation x and y points (saved as separate txt files under the specified simulation output folder, 'tmp_*' folders)

2) Running a simulation
    a) The .xml files are the LBFoam input files
    b) Just to avoid renaming things, I made one for the rough fracture and one for the smooth fracture.
    c) To run, use this command in the terminal (example for running rough fracture on 20 cores): mpirun -np 20 bucket2D bucket2D_rough.xml
    d) Output files are:
        - screenshots of gas fraction and smoothed VOF in output folder 
        - vti files in output folder containing:
            - 'velocity', 'pressure', 'adDensity', 'volumeFraction', 'smoothedVolumeFraction', 'bubbleTags', 'disjoiningPressure'
        - log files for bubbles

3) Analysis and visualization
    a) This is all contained in foam_quality.py
    b) There are functions to plot foam quality and to visualize simulation the VOF output (like the foam_simulation_quality_geom_comparison.png figure)

    
Other comments:

Inside this folder, I ended up running 5 different simulations. The simulations used for the TIPM and SPE papers are in tmp_rough2 and tmp_smooth2. The others were used for testing different parameters for bubble initialization, so they might be illustrative/helpful; the main idea is that you can use a denser configuration to reach a higher foam quality in fewer iterations. There are also some other python files not mentioned above that might be useful in the future, but they are not necessary to reproduce what we reported in the TIPM and SPE papers.
