% test csdp on tsp problems with n <= 150 nodes.
% warning: running this script may require large amounts of memory. Smaller
% instances (n<50), should run in minutes on most modern laptops.
%
% workspace is periodically saved to tsp_results.mat
% 
% author: Jose Bravo Ferreira
%         josesf@math.princeton.edu
addpath ../
clear; clc;

load ./tsplib

keys = tsplib.keys;
n_keys = length(keys);

objvals = zeros(1,n_keys);          % the lower bound
gaps = zeros(1,n_keys);             % gaps (1-objval/optval);
upper_bounds = zeros(1,n_keys);     % the upper bound
rtimes = zeros(1,n_keys);           % running time in seconds
for i=1:length(keys)
    % A is the matrix of distances, B is the TSP graph
    A = tsplib(keys{i});
    n = size(A,1);
    B = diag(ones(1,n-1),1); B(1,n) = 1; B = 0.5*(B+B');
    
    % define problem
    problem = struct;
    problem.A = A;          % graph 1
    problem.B = B;          % graph 2
    problem.P0 = eye(n);    % initialization
    problem.maxits = 750;   % maximum number of iterations
    problem.varsize = 3;    % number of graph nodes per variable
    problem.nprocs = 20;     % number of parallel processors
    problem.mode = 'admm';
    
    fprintf('Solving %s with %d nodes per variable.\n', ...
        keys{i}, problem.varsize);
    
    % solve
    tic;
    results = assign_csdp(problem);
    
    % record resuls
    rtimes(i) = toc;
    objvals(i) = results.optval;
    gaps(i) = 1-results.optval/tspoptvals(keys{i});
    upper_bounds(i) = results.upper_bound_stochastic;
    
    save tsp_results.mat
end
