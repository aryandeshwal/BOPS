% test csdp on a selection of qap problems.
% warning: running this script may require large amounts of memory. Smaller
% instances (n<50), should run in minutes on most modern computers.
%
% workspace is periodically saved to qap_results.mat
% 
% author: Jose Bravo Ferreira
%         josesf@math.princeton.edu
addpath ../
clear; clc;

load ./qaplib

keys = qaplib.keys;
n_keys = length(keys);

objvals = zeros(1,n_keys);          % the lower bound
gaps = zeros(1,n_keys);             % gaps (1-objval/optval);
upper_bounds = zeros(1,n_keys);     % the upper bound
rtimes = zeros(1,n_keys);           % running time in seconds
for i=1:length(keys)
    prob = qaplib(keys{i});
    % A is the matrix of distances, B is the TSP graph
    A = prob(:,:,1);
    B = prob(:,:,2);
    n = size(A,1);
    
    % define problem
    problem = struct;
    problem.A = A;          % graph 1
    problem.B = B;          % graph 2
    problem.P0 = eye(n);    % initialization
    problem.maxits = 750;   % maximum number of iterations
    problem.varsize = 3;    % number of graph nodes per variable
    problem.nprocs = 2;     % number of parallel processors
    problem.mode = 'admm';  
    
    fprintf('Solving %s with %d nodes per variable.\n', ...
        keys{i}, problem.varsize);
    
    % solve
    tic;
    results = assign_csdp(problem);
    
    % record resuls
    rtimes(i) = toc;
    objvals(i) = results.optval;
    gaps(i) = 1-results.optval/qapoptvals(keys{i});
    upper_bounds(i) = results.upper_bound_stochastic;
    
    save qap_results.mat
end
