import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os, subprocess
import argparse

import botorch
from botorch import fit_gpytorch_model
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.optim import module_to_array
from botorch.acquisition import ExpectedImprovement
from gpytorch.constraints import Interval, Positive
from gpytorch.priors import Prior
from gpytorch.kernels import Kernel


class MallowsKernel(Kernel):
    has_lengthscale = True
    def forward(self, X, X2, **params):
        check_extra_dim = 0
        if len(X.shape) > 2:
            check_extra_dim = X.shape[0]
            X = X.squeeze(1)
        if len(X2.shape) > 2:
            check_extra_dim = X2.shape[0]
            X2 = X2[0]
        kernel_mat = torch.sum((X[:, None, :] - X2)**2, axis=-1)
        if check_extra_dim > 0:
            kernel_mat = kernel_mat.unsqueeze(1)
        return torch.exp(-self.lengthscale * kernel_mat)


def featurize(x):
    featurized_x = []
    for nums in range(x.size(0)):
        vec = []
        for i in range(x.size(1)):
            for j in range(i+1, x.size(1)):
                vec.append(1 if x[nums][i] > x[nums][j] else -1)
        featurized_x.append(vec)
    normalizer = np.sqrt(x.size(1)*(x.size(1) - 1)/2) 
    return torch.tensor(featurized_x/normalizer).float()


def evaluate_floorplan(x, dim):
    if x.dim() == 2:
        x = x.squeeze(0)
    # print(f"x {x}")
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


def initialize_model(train_x, train_obj, covar_module=None, state_dict=None):
    # define models for objective and constraint
    if covar_module is not None:
        model = SingleTaskGP(train_x, train_obj, covar_module=covar_module)
    else:
        model = SingleTaskGP(train_x, train_obj)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()#noise_constraint=gpytorch.constraints.Positive())
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def EI_local_search(AF, x):
    best_val = AF(featurize(x.unsqueeze(0)).unsqueeze(1).detach())
    best_point = x.numpy()
    for num_steps in range(100):
        # print(f"best AF value : {best_val} at best_point = {best_point}")
        all_vals = []
        all_points = []
        for i in range(len(best_point)):
            for j in range(i+1, len(best_point)):
                x_new = best_point.copy()
                x_new[i], x_new[j] = x_new[j], x_new[i]
                all_vals.append(AF(featurize(torch.from_numpy(x_new).unsqueeze(0)).unsqueeze(1)).detach())
                all_points.append(x_new)
        idx = np.argmax(all_vals)
        if all_vals[idx] > best_val:
            best_point = all_points[idx]
            best_val = all_vals[idx]
        else:
              break
    print(f"best AF value : {best_val.item()} at best_point = {best_point}")
    return torch.from_numpy(best_point), best_val


def bo_loop(kernel_type):
    n_init = 20
    n_evals = 200
    for nruns in range(20):
        torch.manual_seed(nruns)
        np.random.seed(nruns)
        dim = 30 # np.asarray(scipy.io.loadmat('../COMBO/qap_test/QAP_LIB_A'+str(benchmark_index+1)+'.mat')['A']).shape[0]
        print(f'Input dimension {dim}')
        train_x = torch.from_numpy(np.array([np.random.permutation(np.arange(dim)) for _ in range(n_init)])) 
        outputs = []
        for i in range(n_init):
            outputs.append(evaluate_floorplan(train_x[i], dim))
            # outputs.append(evaluate_qap(train_x[i], benchmark_index, dim))
        train_y = -1*torch.tensor(outputs)
        for num_iters in range(n_init, n_evals):
            inputs = featurize(train_x)
            if kernel_type == 'mallows':
                covar_module = MallowsKernel()
            train_y = (train_y - torch.mean(train_y))/(torch.std(train_y)).float()
            mll_bt, model_bt = initialize_model(inputs, train_y.unsqueeze(1), covar_module)
            model_bt.likelihood.noise_covar.noise = torch.tensor(0.0001).float()
            mll_bt.model.likelihood.noise_covar.raw_noise.requires_grad = False
            fit_gpytorch_model(mll_bt)
            # print(train_y.dtype)
            print(f'\n -- NLL: {mll_bt(model_bt(inputs), train_y.float())}')
            EI = ExpectedImprovement(model_bt, best_f = train_y.max().item())
            # Multiple random restarts
            best_point, ls_val = EI_local_search(EI, torch.from_numpy(np.random.permutation(np.arange(dim))))
            for _ in range(10):
                new_point, new_val = EI_local_search(EI, torch.from_numpy(np.random.permutation(np.arange(dim)))) 
                if new_val > ls_val:
                    best_point = new_point
                    ls_val = new_val
            print(f"Best Local search value: {ls_val}")
            if not torch.all(best_point.unsqueeze(0) == train_x, axis=1).any():
                best_next_input = best_point.unsqueeze(0)
            else:
                print(f"Generating randomly !!!!!!!!!!!")
                best_next_input = torch.from_numpy(np.random.permutation(np.arange(dim))).unsqueeze(0)
            # print(best_next_input)
            next_val = evaluate_floorplan(best_next_input, dim)
            # next_val = evaluate_qap(best_next_input, benchmark_index, dim)
            train_x = torch.cat([train_x, best_next_input])
            outputs.append(next_val)
            train_y = -1*torch.tensor(outputs)
            # train_y = torch.cat([train_y, torch.tensor([next_val])])
            print(f"\n\n Iteration {num_iters} with value: {outputs[-1]}")
            print(f"Best value found till now: {np.min(outputs)}")
            torch.save({'inputs_selected':train_x, 'outputs':outputs, 'train_y':train_y}, 'floorplan_botorch_'+kernel_type+'_EI_30_nrun_'+str(nruns)+'.pkl')


if __name__ == '__main__':
    parser_ = argparse.ArgumentParser(
        description='Bayesian optimization over permutations (QAP)')
    # parser_.add_argument('--benchmark_index', dest='benchmark_index', type=int, default=3)
    parser_.add_argument('--kernel_type', dest='kernel_type', type=str, default='mallows')
    args_ = parser_.parse_args()
    kwag_ = vars(args_)
    bo_loop(kwag_['kernel_type'])

