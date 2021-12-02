% MIT License
%
% Copyright (c) 2017 Jose Bravo Ferreira
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

%%% Interior point

function [results] = assign_csdp_ip(problem)
    A = problem.A;
    B = problem.B;
    varsize = problem.varsize;
    
    n = size(A,1);
    varlist = find_variables(B,varsize);
    BLKSZ = size(varlist,2);
    n_vars = size(varlist,1);
    fprintf('Number of nodes: %d\n',n);
    fprintf('Number of variables: %d\n', n_vars);
    fprintf('Variable size: %dn+1\n', varsize);
    
    d = BLKSZ*n+1;
    
    %% constraints
    %%% linear constraints
    [Ae, ~, be] = build_Ae(n,BLKSZ);

    %%% overlap constraints
    BB_col = build_BB(n,BLKSZ);
              
    %% COST
    tempC = kron(B,A); 
    % fix tempC to account for overlapping blocks
    count = zeros(n);
    for i=1:n
        for j=1:n
            for k=1:n_vars
                if (sum(varlist(k,:)==i)==1 && sum(varlist(k,:)==j)==1)
                    count(i,j) = count(i,j)+1;
                end
            end
        end
    end
    count(count==0) = 1;
    tempC = tempC ./ kron(count,ones(n));
    
    
    C = zeros(d^2,n_vars);
    for k=1:n_vars
        idx = [];
        for i=varlist(k,:)
            idx = [idx, (i-1)*n+1:i*n];
        end
        c = [tempC(idx,idx) zeros(d-1,1); zeros(1,d)];
        c = 0.5*(c+c');
        C(:,k) = c(:);
    end
    
    % constraint auxiliary matrices
    J = ones(n);
    
    %% SOLVE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    cvx_begin
        cvx_solver mosek
        cvx_precision low
        
        variable X(d^2,n_vars)
        variable L(n,n)
        
        minimize sum(sum(X.*C))
        subject to
            for i=1:n_vars
                cX = reshape(X(:,i),d,d);
                % psdness %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                cX == hermitian_semidefinite(d);
                % linear constraints %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                cX(d,d) == 1;
                for j=1:BLKSZ
                    % diagonal constraints
                    idx_j = (j-1)*n+1:j*n;
                    trace(cX(idx_j,idx_j)) == 1;
                    trace(cX(idx_j,idx_j)*J) == 1;
                    % offdiagonal constraints
                    for k=j+1:BLKSZ
                        idx_k = (k-1)*n+1:k*n;
                        trace(cX(idx_j,idx_k)) == 0;
                        trace(cX(idx_j,idx_k)*J) == 1;
                    end
                end
                % squared constraints
                for j=1:BLKSZ*n
                    cX(j,j) == cX(j,d);
                end
                % overlap constraints %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                for j=1:BLKSZ
                    cX((j-1)*n+1:j*n,d) == L(:,varlist(i,j));
                end
            end
            % positiveness %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            X(:) >= 0;
            % stochasticity %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            sum(L') == 1;
    cvx_end
    P = closest_permutation(L);
    if problem.stoch
        P_s = project_to_permutation(D,A*nA,B*nB);
    else
        P_s = P;
    end
    
    results = struct;
    results.P = P;
    results.P_stochastic = P_s;
    results.D = L;
    results.optval = cvx_optval;
    results.upper_bound = trace(A*P*B*P');
    results.upper_bound_stochastic = trace(A*P_s*B*P_s');
end
