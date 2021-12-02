## Bayesian Optimization over Permutation Spaces


This repository contains the source code and the resources related to the paper "[Bayesian Optimization over Permutation Spaces]()" published at [AAAI'22](https://aaai.org/Conferences/AAAI-22/) conference. 

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
As discussed in the paper, we propose two algorithms: BOPS-T and BOPS-H. A good place to start is the floorplanning directory where the files  'floorplan_kendall.py' and 'floorplan_mallows.py' contains the code for BOPS-T and BOPS-H respectively. 


BOPS-T utilizes an SDP solver (for acquisition function optimization) implemented [here](https://github.com/fsbravo/csdp).
BOPS-H is built on top of [GPyTorch](https://github.com/cornellius-gp/gpytorch) and [BoTorch](https://github.com/pytorch/botorch) libraries.
We thank the original authors for their code.

