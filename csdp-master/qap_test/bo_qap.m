addpath ../
clear; clc;
bo_qap = load('../test_QAP.mat')

% define problem
problem = struct;
problem.A = bo_qap.A;          % graph 1
problem.B = bo_qap.B;          % graph 2
problem.P0 = eye(n);    % initialization
problem.maxits = 1000;   % maximum number of iterations
problem.varsize = 3;    % number of graph nodes per variable
problem.nprocs = 2;     % number of parallel processors
problem.mode = 'admm';  

fprintf('Solving %s with %d nodes per variable.\n', ...
    keys{i}, problem.varsize);

% solve
tic;
results = assign_csdp(problem);