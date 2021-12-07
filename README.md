## Bayesian Optimization over Permutation Spaces


This repository contains the source code and the resources related to the paper "[Bayesian Optimization over Permutation Spaces](https://arxiv.org/abs/2112.01049)" published at [AAAI'22](https://aaai.org/Conferences/AAAI-22/) conference. 

### Benchmark simulations

We provided three real-world benchmarks to drive future research on this important problem. They are described below:
 
1. Floorplanning: 
	- The simulator file is in floorplanning directory.
	- The input is given in a permutation file (named 'permutation.txt') as a comma separated values from 0-10 
	- The output is given by running: ./floorplan_simulation b1_floorplan.blk
	- Permutation file will be read by the simulation internally
	- There are two variants: b1_floorplan.blk and b2_floorplan.blk 
 
2. Cell Placement 
	- The simulator file is in cell_placement directory.
	- The input is given in a permutation file (named 'permutation.txt') as a comma separated values from 0-10 
	- The output is given by running: ./cp_simulator ex10_40_2_3.dat
	- Permutation file will be read by the simulation internally

3. Heterogeneous Manycore Design 
	- There is a dataset file named 'hmd_dataset.pkl' containing around 15K points
	- hmd_dataset.pkl contains a dictionary with two keys 'points' (permutations) and 'vals' (objective values) 
  

### Source code
As discussed in the paper, we propose two algorithms: BOPS-T and BOPS-H. 


#### Installation Requirements
- numpy, scipy
- PyTorch, GPyTorch, BoTorch
- [MATLAB engine API for python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) (only for BOPS-T) 

#### Example usage on the floorplanning domain:

The floorplanning directory contains two main files: ``floorplan_kendall.py`` and ``floorplan_mallows.py`` for BOPS-T and BOPS-H respectively. 

In ``floorplan_mallows.py``, ``evaluate_floorplan`` method defines a call to the black-box objective function and ``bo_loop`` is the entry point for the code. 
In ``bo_loop``, ``n_init`` is the number of initial evaluations to initialize the GP surrogate model. The total budget is given by ``n_evals-n_init``. We use the Expected improvement acquisition function which is optimized via local search with multiple restarts. The number of restarts can be changed in line 127.


#### Acknowledgements
BOPS-T utilizes an SDP solver (for acquisition function optimization) implemented [here](https://github.com/fsbravo/csdp).
BOPS-H is built on top of [GPyTorch](https://github.com/cornellius-gp/gpytorch) and [BoTorch](https://github.com/pytorch/botorch) libraries.
We thank the original authors for their code.

