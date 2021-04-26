# Python code for the simulation
This library provides the models and instructions 
to build differomorphic mappings by the iteration method and
the invertible Neural Network (iNN) method. It includes the 2D 
simulation code of our paper **Motion Mappings for Continuous Bilateral Teleoperation**.
See for our paper: https://ieeexplore.ieee.org/document/9387124

## Installation in Ubuntu
- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or 
[Anaconda](https://docs.anaconda.com/anaconda/install/) if you don't have it.
 - Create a new Conda environment with Python 2.7, and activate it.

    `conda create -n env_name python=2.7`

    `conda activate env_name`
 - Install requirements in `env_name`.
  
   `conda install numpy scipy numba matplotlib tensorflow=1.15`

   `conda install -c conda-forge quaternion notebook`

## Examples
 - `iteration_2D_sim.ipynb` shows an example of motion mapping by the iteration method in 2D example, which generates the Fig. 4 of our paper.

 - `iNN_2D_sim.ipynb` shows the method of iNN in the same 2D example.