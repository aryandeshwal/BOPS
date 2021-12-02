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



def evaluate_single(x, benchmark_index, dim):
    if x.dim() == 2:
        x = x.squeeze(0)
    x = x.numpy()
    A = np.asarray(scipy.io.loadmat('QAP_LIB_A'+str(benchmark_index+1)+'.mat')['A'])
    B = np.asarray(scipy.io.loadmat('QAP_LIB_'+str(benchmark_index+1)+'.mat')['B'])
    E = np.eye(dim)
    permutation = np.array([np.arange(dim), x])

    P = np.zeros([dim, dim]) #initialize the permutation matrix

    for i in range(dim):
        P[:, i] = E[:, permutation[1][i]]
    result = (np.trace(P.dot(B).dot(P.T).dot(A.T)))
    print(f'QAP objective value: {result}')
    return result


def main():
    n_init = 20
    n_evals = 200
    benchmark_index = 3 # int(sys.argv[1]) # set to 3 
    for nruns in range(20):
        torch.manual_seed(nruns)
        np.random.seed(nruns)
        dim = np.asarray(scipy.io.loadmat('QAP_LIB_A'+str(benchmark_index+1)+'.mat')['A']).shape[0]
        train_x = torch.from_numpy(np.array([np.random.permutation(np.arange(dim)) for _ in range(n_init)]))
        outputs = []
        for i in range(n_init):
            outputs.append(evaluate_single(train_x[i], benchmark_index, dim))
        train_y = torch.tensor(outputs)
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
            scipy.io.savemat('bo_qap.mat', {'theta_ls_matrix':tls_matrix, 'tls_matrix':theta_ls_matrix, 'theta':theta, 'dim':dim})
            eng = matlab.engine.start_matlab()
            eng.run_sdp_qap(nargout=0)
            best_point = torch.from_numpy(np.argwhere(np.asarray(eng.workspace['results']['P_stochastic']) == 1)[:, 1]) 
            if not torch.all(best_point.unsqueeze(0) == train_x, axis=1).any():
                best_next_input = best_point.unsqueeze(0)
            else:
                print(f"Point already existing. Generating randomly!")
                best_next_input = torch.from_numpy(np.random.permutation(train_x[0].numpy())).unsqueeze(0)
            next_val = evaluate_single(best_next_input, benchmark_index, dim)
            train_x = torch.cat([train_x, best_next_input])
            train_y = torch.cat([train_y, torch.tensor([next_val])])
            torch.save({'inputs_selected':train_x, 'outputs':train_y}, 'qap_kendall_ts_sdp_benchmark_index_'+str(benchmark_index)+'_'+str(nruns)+'.pkl')


if __name__ == '__main__':
    main()
