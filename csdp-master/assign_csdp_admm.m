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

%%% ADMM for C-SDP (dual)

function [results] = assign_csdp_admm(problem)
    A = problem.A;
    B = problem.B;
    P0 = problem.P0;
    MAX_ITS = problem.maxits;
    varsize = problem.varsize;
    
    nA = norm(A(:));
    A = A/nA;
    nB = norm(B(:));
    B = B/nB;
    
    n = size(A,1);
    varlist = find_variables(B,varsize);
    neighbours = find_neighbours(varlist,n);
    BLKSZ = size(varlist,2);
    n_vars = size(varlist,1);
    fprintf('Number of nodes: %d\n',n);
    fprintf('Number of variables: %d\n', n_vars);
    fprintf('Variable size: %dn+1\n', varsize);
    
    d = BLKSZ*n+1;
    
    %% constraints
    %%% linear constraints
    [Ae, invAeAeT, be] = build_Ae(n,BLKSZ);
    n_cons = size(Ae,1);

    %%% overlap constraints
    BB_col = build_BB(n,BLKSZ);

    %%% positivity
    DD = build_DD(n,BLKSZ);
              
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
            
    %% INITIALIZATION
    % dual variables
    X = zeros(d^2,n_vars);
    p = P0(:);
    for k=1:n_vars
        idx = [];
        for i=varlist(k,:)
            idx = [idx, (i-1)*n+1:i*n];
        end
        p_temp = [p(idx); 1];
        X(:,k) = reshape(p_temp*p_temp',d^2,1);
    end
    S = zeros(d^2,n_vars);
    Z = zeros((BLKSZ*n)^2,n_vars);
    Y = zeros(n_cons,n_vars);
    G = zeros(n,n);
    
    %%% initialization    
    alpha = 1.5;
    beta = 1.0;
    c_tilde_w = cell(1,n);
    for i=1:n
        nb = [];
        for k=1:BLKSZ
            nb = [nb, neighbours{k}{i}];
        end
        p = length(nb);
        x = 1/(alpha-beta);
        y = -beta/(alpha-beta)/(alpha-beta+p*beta);
        c_tilde_w{i} = x*eye(p)+y*ones(p);
    end
        
    nBe = sqrt(norm(be)^2*n_vars);
    nC = norm(C,'fro');
    
    %% SOLVE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% parameters
    rho = 1/n;
    RHO_MIN = 0.00001;
    RHO_MAX = 10000;
    alpha = 1.05;
    tau = 1.568;

    etas = zeros(6,1);
    ETA_MAX = 10^-6;
    eta = 1;
    eta_vec = [];
    cur_it = 0;

    W_col = cell(1,BLKSZ);
    for i=1:BLKSZ
        W_col{i} = zeros(n,n_vars);
    end
    BBtW = col_multiply(BB_col,W_col);
        
    while cur_it<MAX_ITS && eta>ETA_MAX
            
        % 1. (s,t)
        for i=1:n_vars
            S(:,i) = project_to_psd_cone(C(:,i)-...
                DD'*Z(:,i)-...
                Ae'*Y(:,i)-...
                BBtW(:,i)-...
                1/rho*X(:,i));
        end
        T = 1/n*(sum(cell2mat(W_col),2)-1/rho/n*sum(G,2))+1/rho/n*ones(n,1);
        % 2. (y)
        Y = invAeAeT*(Ae*(C-S-DD'*Z-BBtW-1/rho*X)+1/rho*repmat(be,1,n_vars));
        % 3. (z,w) and (g)
        Z = project_to_pos_cone(DD*(C-S-Ae'*Y-1/rho*X));
        dMat = C-S-Ae'*Y-1/rho*X;
        for i=1:n
            vec = zeros(1,BLKSZ);
            for k=1:BLKSZ
                vec(k) = length(neighbours{k}{i});
            end
            nb_size = sum(vec);
            c_tilde = zeros(n,nb_size);
            count = 1;
            for k=1:BLKSZ
                c_tilde(:,count:count+vec(k)-1) = BB_col{k}*dMat(:,neighbours{k}{i});
                count = count+vec(k);
            end
            c_tilde = c_tilde + repmat(T+1/rho*G(:,i),1,nb_size);
            res = c_tilde*c_tilde_w{i};
            part = matpart(res,2,vec);
            for k=1:BLKSZ
                W_col{k}(:,neighbours{k}{i}) = part{k};
            end
            G(:,i) = G(:,i) + tau*rho*(T-sum(res,2));
        end
        BBtW = col_multiply(BB_col,W_col);
        % 2. (y)
        Y = invAeAeT*(Ae*(C-S-DD'*Z-BBtW-1/rho*X)+1/rho*repmat(be,1,n_vars));
        % (x)
        X = X + tau*rho*(-C+S+DD'*Z+Ae'*Y+BBtW);

        % control variables
        nX = norm(X,'fro');
        nZ = norm(Z,'fro');
        nS = norm(S,'fro');
        etas(1) = abs(trace(X'*S))/(1+nX+nS);
        etas(2) = abs(trace(X'*(DD'*Z)))/(1+nX+nZ);
        etas(3) = norm(Ae*X-repmat(be,1,n_vars),'fro')/(1+nBe);
        etas(4) = norm(-C+Ae'*Y+S+DD'*Z+BBtW,'fro')/(1+nC);
        etas(5) = norm(X-project_to_pos_cone(X),'fro')/(1+nX);
        etas(6) = norm(Z-project_to_pos_cone(Z),'fro')/(1+nZ);

        eta = max(etas);
        if max([etas(1) etas(2)])>10*max([etas(3) etas(4)])
            rho = rho/alpha;
            rho = max(rho,RHO_MIN);
        elseif max([etas(1) etas(2)])<0.1*max([etas(3) etas(4)])
            rho = rho*alpha;
            rho = min(rho,RHO_MAX);
        end
        cur_it = cur_it + 1;
        if mod(cur_it, 100)==0
            fprintf('Iteration %d :: rho = %f :: eta = %f :: eta_p = %f :: eta_d = %f :: eta_c1 = %f :: eta_c2 = %f :: eta_ps1 = %f :: eta_ps2 = %f\n',...
                cur_it,rho,eta,etas(3),etas(4),etas(1),etas(2),etas(5),etas(6));
        end
        eta_vec = [eta_vec, eta];
    end
    
    %% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    D = zeros(n);
    seen = [];
    varsize = size(varlist,2);
    for idx=1:n_vars
        temp = reshape(X(:,idx),d,d);
        for i=1:varsize
            var = varlist(idx,i);
            if sum(seen==var)==0
                D(:,var) = temp(d,(i-1)*n+1:i*n);
            end
            seen = [seen, var];
        end
    end
    P = closest_permutation(D);
    if problem.stoch
        P_s = project_to_permutation(D,A*nA,B*nB);
    else
        P_s = P;
    end
    
    results = struct;
    results.P = P;
    results.P_stochastic = P_s;
    results.D = D;
    results.optval = sum(sum(C.*X))*nA*nB;
    results.upper_bound = trace(A*P*B*P')*nA*nB;
    results.upper_bound_stochastic = trace(A*P_s*B*P_s')*nA*nB;
    results.eta_vec = eta_vec;
end

function [x_n] = project_to_psd_cone(x)
    l = round(sqrt(length(x)));
    [U,S] = eig(reshape(x,l,l));
    S(S<0) = 0;
    x_n = reshape(U*S*U',l^2,1);
end

function [x_n] = project_to_pos_cone(x)
    x_n = x; 
    x_n(x<0) = 0;
end

function [A] = col_multiply(A_col,B_col)
    A = A_col{1}'*B_col{1};
    for i=2:length(A_col)
        A = A + A_col{i}'*B_col{i};
    end
end
