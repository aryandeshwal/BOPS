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

function [results] = assign_csdp_parallel(problem)
    A = problem.A;
    B = problem.B;
    P0 = problem.P0;
    MAX_ITS = problem.maxits;
    nprocs = problem.nprocs;
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
    
    %% LAUNCH PARALLEL POOL AND INITIALIZE LOCAL VARIABLES %%%%%%%%%%%%%%%%
    mypool = gcp('nocreate');
    if isempty(mypool)
        mypool = parpool(nprocs);
    end
    
    spmd
        %%% linear constraints
        [Ae, invAeAeT, be] = build_Ae(n,BLKSZ);
        n_cons = size(Ae,1);
        
        %%% overlap constraints
        BB_col = build_BB(n,BLKSZ);
        
        %%% positivity
        DD = build_DD(n,BLKSZ);
    end
              
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
    spmd
        codist = codistributor('1d',2);
        X = codistributed(X,codist);
        C = codistributed(C,codist);
        S = zeros(d^2,n_vars,codist);
        Z = zeros((BLKSZ*n)^2,n_vars,codist);
        Y = zeros(n_cons,n_vars,codist);
        dMat = zeros(d^2,n_vars,codist);

        codist_w = codistributor('1d',2,ones(1,nprocs));
        etas = zeros(9,nprocs,codist_w);
        codist_g = codistributor('1d',2);
        G = zeros(n,n,codist_g);
    end
    T = zeros(n,1);
    
    %%% initialization  
    spmd
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
    end
        
    spmd
        %%% setup local variables
        nC = norm(C,'fro');
        nBe = sqrt(norm(be)^2*n_vars);
        lC = getLocalPart(C);
        lS = getLocalPart(S);
        lZ = getLocalPart(Z);
        lY = getLocalPart(Y);
        lX = getLocalPart(X);
    end
    
    %% SOLVE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    spmd
        %%% parameters
        rho = 1/n;
        RHO_MIN = 0.00001;
        RHO_MAX = 10000;
        alpha = 1.05;
        tau = 1.568;
        
        ETA_MAX = 10^-6;
        eta = 1;
        cur_it = 0;
        
        Gl = gather(G);
        W_col = cell(1,BLKSZ);
        for i=1:BLKSZ
            W_col{i} = zeros(n,n_vars);
        end
        lW_col = cell(1,BLKSZ);
        [st,ed] = globalIndices(C,2);
        for i=1:BLKSZ
            lW_col{i} = W_col{i}(:,st:ed);
        end
        BBtW = col_multiply(BB_col,lW_col);
        
        while cur_it<MAX_ITS && eta>ETA_MAX
            [st,ed] = globalIndices(C,2);
            for i=1:BLKSZ
                lW_col{i} = W_col{i}(:,st:ed);
            end
            
            % 1. (s,t)
            for i=1:size(lC,2)
                lS(:,i) = project_to_psd_cone(lC(:,i)-...
                    DD'*lZ(:,i)-...
                    Ae'*lY(:,i)-...
                    BBtW(:,i)-...
                    1/rho*lX(:,i));
            end
            T = 1/n*(sum(cell2mat(W_col),2)-1/rho/n*sum(Gl,2))+1/rho/n*ones(n,1);
            labBarrier;
            % 2. (y)
            lY = invAeAeT*(Ae*(lC-lS-DD'*lZ-BBtW-1/rho*lX)+1/rho*repmat(be,1,size(lY,2)));
            labBarrier; 
            % 3. (z,w) and (g)
            lZ = project_to_pos_cone(DD*(lC-lS-Ae'*lY-1/rho*lX));
            ldMat = lC-lS-Ae'*lY-1/rho*lX;
            dMat = codistributed.build(ldMat,getCodistributor(dMat));
            dMatl = gather(dMat);
            for i=1:n
                vec = zeros(1,BLKSZ);
                for k=1:BLKSZ
                    vec(k) = length(neighbours{k}{i});
                end
                nb_size = sum(vec);
                c_tilde = zeros(n,nb_size);
                count = 1;
                for k=1:BLKSZ
                    c_tilde(:,count:count+vec(k)-1) = BB_col{k}*dMatl(:,neighbours{k}{i});
                    count = count+vec(k);
                end
                c_tilde = c_tilde + repmat(T+1/rho*Gl(:,i),1,nb_size);
                res = c_tilde*c_tilde_w{i};
                part = matpart(res,2,vec);
                for k=1:BLKSZ
                    W_col{k}(:,neighbours{k}{i}) = part{k};
                end
                Gl(:,i) = Gl(:,i) + tau*rho*(T-sum(res,2));
            end
            for i=1:BLKSZ
                lW_col{i} = W_col{i}(:,st:ed);
            end
            labBarrier;
            BBtW = col_multiply(BB_col,lW_col);
            % 2. (y)
            lY = invAeAeT*(Ae*(lC-lS-DD'*lZ-BBtW-1/rho*lX)+1/rho*repmat(be,1,size(lY,2)));
            labBarrier;
            % (x)
            lX = lX + tau*rho*(-lC+lS+DD'*lZ+Ae'*lY+BBtW);
            
            % control variables
            etas(1,labindex) = abs(trace(lX'*lS));
            etas(2,labindex) = abs(trace(lX'*(DD'*lZ)));
            etas(3,labindex) = norm(Ae*lX-repmat(be,1,size(lX,2)),'fro');
            etas(4,labindex) = norm(-lC+Ae'*lY+lS+DD'*lZ+BBtW,'fro');
            etas(5,labindex) = norm(lX-project_to_pos_cone(lX),'fro');
            etas(6,labindex) = norm(lZ-project_to_pos_cone(lZ),'fro');
            etas(7,labindex) = norm(lX,'fro');
            etas(8,labindex) = norm(lZ,'fro');
            etas(9,labindex) = norm(lS,'fro');
            
            etas_l = gather(etas);
            etas_vec = [sum(etas_l(1:2,:),2); sqrt(sum(etas_l(3:end,:).^2,2))];
            nX = etas_vec(7); nZ = etas_vec(8); nS = etas_vec(9);
            etas_den = [1+nX+nS;1+nX+nZ;1+nBe;1+nC;1+nX;1+nZ];
            etas_vec = etas_vec(1:6)./etas_den;
            eta = max(etas_vec);
            if max([etas_vec(1) etas_vec(2)])>10*max([etas_vec(3) etas_vec(4)])
                rho = rho/alpha;
                rho = max(rho,RHO_MIN);
            elseif max([etas_vec(1) etas_vec(2)])<0.1*max([etas_vec(3) etas_vec(4)])
                rho = rho*alpha;
                rho = min(rho,RHO_MAX);
            end
            cur_it = cur_it + 1;
            if labindex == 1
                fprintf('Iteration %d :: rho = %f :: eta = %f :: eta_p = %f :: eta_d = %f :: eta_c1 = %f :: eta_c2 = %f :: eta_ps1 = %f :: eta_ps2 = %f\n',...
                    cur_it,rho,eta,etas_vec(3),etas_vec(4),etas_vec(1),etas_vec(2),etas_vec(5),etas_vec(6));
            end
        end
        X = codistributed.build(lX,getCodistributor(X));
    end
    X = gather(X);
    C = gather(C);
    
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
    delete(gcp('nocreate'));
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
