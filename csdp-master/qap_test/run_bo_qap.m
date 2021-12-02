addpath ../
clear; clc;
bo_qap = load('../test_QAP.mat')

% define problem
n = 16;
problem = struct;
problem.A = bo_qap.tls_matrix;          % graph 1
problem.B = bo_qap.theta_ls_matrix';          % graph 2
problem.P0 = eye(n);    % initialization
problem.maxits = 200;   % maximum number of iterations
problem.varsize = 2;    % number of graph nodes per variable
problem.nprocs = 2;     % number of parallel processors
problem.mode = 'admm';  


% solve
tic;
results = assign_csdp(problem);