Here is a quick rundown of how to use the LBFoam code. The workflow takes a few steps but shouldn't be too bad:

### Compiling

Clone the LBFoam repo
```
git clone https://github.com/mehdiataei/LBfoam.git
```
Inside LBFoam repo directory, copy this current folder into examples/lbfoam
``` 
cd examples/lbfoam/
git clone https://github.com/cedar-ma/LBfoam_in_fracture.git
```
Compile `bucket2D.cpp`
``` 
cd LBfoam_in_fracture
make
```

(Note: To compile the software on MacOS, uncomment the ` -DPLB_MAC_OS_X` compilation flag in the Makefile).`cd LBfoam_in_fracture` directory and run `make` in the terminal to compile
    
### Geometry Prep
- The [create_fracture_geoms_2d.py](create_fracture_geoms_2d.py) file can be used to create a smooth fracture and/or a fracture from a raw image (reusing MPLBM code for palabos). The bubble placement algorithm is also in there and will create the data needed for LBFoam. At the bottom of the file, all the function calls are there. Some commands are commented out, but you can uncomment/edit as needed.
- Outputs will be:
    - geomtry files (saved to `input` directory)
    - bubble nucleation x and y points (saved as separate txt files under the specified simulation output `tmp` folder)

### Running a simulation
Create `tmp` folder to store the outputs (the output folder name must be the same as `outDir` variable in the `bucket2D.xml` file). Run the example using the following command. The `bucket2D.xml` contains the simulation parameters.

``` 
./bucket2D bucket2D.xml
```

To run the example in parallel on TACC using 8 cores for example:

``` 
ibrun -np 8 bucket2D bucket2D.xml
```

Output files are:
- screenshots of gas fraction and smoothed VOF in output folder 
- vti files in output folder containing:
    - 'velocity', 'pressure', 'adDensity', 'volumeFraction', 'smoothedVolumeFraction', 'bubbleTags', 'disjoiningPressure'
- log files for bubbles

### Analysis and visualization

This is all contained in [foam_quality.py](foam_quality.py).
There are functions to plot foam quality and to visualize the VOF simulation output (like the foam_simulation_quality_geom_comparison.png figure)

   
