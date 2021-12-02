% add white noise to a doubly stochastic matrix before using hungarian
% algorithm to obtain better permutation matrix
function [P] = project_to_permutation(D,A,B)
    fprintf('Projecting to a permutation\n');
    n = size(A,1);
    itsperstep = 1000;
    
    max_noise = 0.5;
    n_noise = 10;
    noise_v = linspace(0,max_noise,n_noise+1);
    noise_v = noise_v(2:end);
    
    P = closest_permutation(D);
    best_val = trace(A*P*B*P');
    for i=1:n_noise
        noise = noise_v(i);
        for j=1:itsperstep
            D_n = D + normrnd(0,noise,n,n);
            P_n = closest_permutation(D_n);
            obj = trace(A*P_n*B*P_n');
            if obj<best_val
                best_val = obj;
                P = P_n;
            end
        end
    end
end
