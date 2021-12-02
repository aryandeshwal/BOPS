import torch
import numpy as np
import sys
import scipy.io
import os, subprocess
import matlab.engine

def fastmvg(Phi, alpha, D):
    # fastmvg sampler (code from BOCS) https://github.com/baptistar/BOCS
    # Fast sampler for multivariate Gaussian distributions (large p, p > n) of
    #  the form N(mu, S), where
    #       mu = S Phi' y
    #       S  = inv(Phi'Phi + inv(D))
    # Reference:
    #   Fast sampling with Gaussian scale-mixture priors in high-dimensional
    #   regression, A. Bhattacharya, A. Chakraborty and B. K. Mallick
    #   arXiv:1506.04778
    n, p = Phi.shape

    d = np.diag(D)
    u = np.random.randn(p) * np.sqrt(d)
    delta = np.random.randn(n)
    v = np.dot(Phi,u) + delta
    mult_vector = np.vectorize(np.multiply)
    Dpt = mult_vector(Phi.T, d[:,np.newaxis])
    w = np.linalg.solve(np.matmul(Phi,Dpt) + np.eye(n), alpha - v)
    x = u + np.dot(Dpt,w)

    return x

def featurize(x):
    featurized_x = []
    for nums in range(x.size(0)):
        vec = []
        for i in range(x.size(1)):
            for j in range(i+1, x.size(1)):
                vec.append(1 if x[nums][i] > x[nums][j] else -1)
        featurized_x.append(vec)
    normalizer = np.sqrt(x.size(1)*(x.size(1) - 1)/2)
    return (torch.tensor(featurized_x/normalizer).float())


def evaluate_single(x, dim):
    if x.dim() == 2:
        x = x.squeeze(0)
    print(f"x {x}")
    with open("permutation.txt", "w") as f:
        for i in range(len(x)):
            print(x[i].item(), end=',', file=f)
    FNULL = open(os.devnull, 'w')
    subprocess.call(['./floorplan_simulation', 'b1_floorplan.blk'], stdout=FNULL, stderr=subprocess.STDOUT)
    with open("output_floorplan.txt", "r") as f:
        fl = f.readlines()
    output = float(fl[0])
    print(f"result: {output}")
    return output


def main():
    n_init = 20
    n_evals = 200
    dim = 10 # int(sys.argv[1]) 
    for nruns in range(20):
        torch.manual_seed(nruns)
        np.random.seed(nruns)
        train_x = torch.from_numpy(np.array([np.random.permutation(np.arange(dim)) for _ in range(n_init)]))
        outputs = []
        for i in range(n_init):
            outputs.append(evaluate_single(train_x[i], dim))
        train_y = torch.tensor(outputs)
        # print(train_x)
        # print(train_y)
                covar_module = KendallKernel()
        for num_iters in range(n_init, n_evals):
            X = featurize(train_x).numpy()
            for i in range(X.shape[1]):
                X[:, i] = (X[:, i] - np.mean(X[:, i]))/(np.std(X[:, i]))
            y = train_y.numpy()
            y = (y-np.mean(y))/np.std(y)
            theta = fastmvg(X, y, np.eye(X.shape[1]))
            theta_ls_matrix = np.zeros((dim, dim))
            theta_ls_matrix[np.triu_indices(dim, k=1)] = theta 
            tls_matrix = np.zeros((dim, dim))
            tls_matrix[np.triu_indices(dim, k=1)] = 1/np.sqrt(dim*(dim - 1)/2)
            tls_matrix.T[np.triu_indices(dim, k=1)] = -1/np.sqrt(dim*(dim - 1)/2)
            scipy.io.savemat('bo_FP.mat', {'theta_ls_matrix':tls_matrix, 'tls_matrix':theta_ls_matrix, 'theta':theta, 'dim':dim})
            eng = matlab.engine.start_matlab()
            eng.run_sdp_fp(nargout=0)
            # print(f"Best upper bound value found by SDP {eng.workspace['results']['upper_bound_stochastic']}") 
            # print(f"Best lower bound value found by SDP {eng.workspace['results']['optval']}") 
            best_point = torch.from_numpy(np.argwhere(np.asarray(eng.workspace['results']['P_stochastic']) == 1)[:, 1]) 
            if not torch.all(best_point.unsqueeze(0) == train_x, axis=1).any():
                best_next_input = best_point.unsqueeze(0)
            else:
                print(f"Point already existing. Generating randomly!")
                best_next_input = torch.from_numpy(np.random.permutation(train_x[0].numpy())).unsqueeze(0)
            next_val = evaluate_single(best_next_input, dim)
            train_x = torch.cat([train_x, best_next_input])
            train_y = torch.cat([train_y, torch.tensor([next_val])])
            print(f"Best value found till now: {train_y.min().item()}")
            torch.save({'inputs_selected':train_x, 'outputs':train_y}, 'floorplan_kendall_ts_sdp_'+str(dim)+'_'+str(nruns)+'.pkl')


if __name__ == '__main__':
    main()
