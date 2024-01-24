""""
Total cost ratio comparison with ProxSkip and VR_ProxSkip
"""
# !/usr/bin/env python
# coding: utf-8
# Fundamental environment setup

import matplotlib
import numpy as np
import psutil
import ray
import seaborn as sns
import copy
import os, argparse, time

import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.datasets import load_svmlight_file, fetch_rcv1

from optmethods.datasets import get_dataset
from optmethods.first_order import Adgd, Gd, Nesterov, RestNest
from optmethods.loss import LogisticRegression
from optmethods.utils import get_trace, relative_round
from optmethods.optimizer import StochasticOptimizer

# Arguments
parser = argparse.ArgumentParser(description='Variance-Reduced ProxSkip.')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--nworkers', type=int, default=10, help='number of workers')
parser.add_argument('--it_local', type=int, default=20)
parser.add_argument('--it_max', type=int, default=10001)
parser.add_argument('--cohort_size', type=int, default=10)
parser.add_argument('--dataset', type=str, default='w8a')
parser.add_argument('--choose_p', type=str, default='kappa', help='Choose from local and kappa')
parser.add_argument('--cc', type=int, default=1, help='communication cost')
parser.add_argument('--lc', type=int, default=1, help='local step cost')
parser.add_argument('--cerr', type=float, default=1, help='error condition choosen from [1e-6, 1e-8]')
parser.add_argument('--regul', type=float, default='1e-3', help='regularizer [1e-2, 1e-3, 1e-4, 1e-5]')
args = parser.parse_args()

batch_size = args.batch_size
n_workers = args.nworkers
cohort_size = args.cohort_size
com_cost = args.cc
# local_cost = args.lc
it_local = args.it_local
it_max = args.it_max
dataset = args.dataset
choose_p = args.choose_p
cerr = args.cerr
regul = args.regul

if cohort_size != n_workers:
    print("You are doing partial participation!")

num_cpus = psutil.cpu_count(logical=False)
ray.init(num_cpus=num_cpus)

###########################################################################################
# Get data and define problem
###########################################################################################
import sklearn.datasets
import urllib.request

data_url = f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{dataset}"
data_path = f"./{dataset}"
print(data_path)
f = urllib.request.urlretrieve(data_url, data_path)
A, b = sklearn.datasets.load_svmlight_file(data_path)

n, dim = A.shape  # \# of data and dimension for each data: 49749, 300
print(n, dim)
if n % num_cpus != 0:
    A = A[:n - (n % num_cpus)]
    b = b[:n - (n % num_cpus)]
b_unique = np.unique(b)
if (b_unique == [1, 2]).all():
    # Transform labels {1, 2} to {0, 1}
    b = b - 1
elif (b_unique == [-1, 1]).all():
    # Transform labels {-1, 1} to {0, 1}
    b = (b+1) / 2
else:
    # replace class labels with 0's and 1's
    b = 1. * (b == b[0])
# A = A.toarray()
l1 = 0
loss = LogisticRegression(A, b, l1=l1, l2=0)
n, dim = A.shape
if n <= 20000 or dim <= 20000:
    print('Computing the smoothness constant via SVD, it may take a few minutes...')
# L = loss.smoothness
oL = loss.smoothness
L = loss.batch_smoothness(batch_size)
# l2 = 0
l2 = regul * L       # here we choose the regularizer to be 1e-5, in paper 1e-4
loss.l2 = l2
# x0 = csc_matrix((dim, 1))
x0 = np.zeros(dim)
n_epoch = 1000
# it_max = (n_epoch * n) // batch_size
trace_len = it_max
# print(L, l2)

###########################################################################################
# Solve problem by Nesterov's method
###########################################################################################
rest = RestNest(loss=loss, doubling=True)
rest_tr = rest.run(x0=x0, it_max=50001)
rest_tr.compute_loss_of_iterates()
# rest.trace.plot_losses()
# plt.yscale('log')
# plt.show()
# plt.close()

###########################################################################################
# Non-iid
###########################################################################################
# cohort_size = cohort_size # CHANGE IF YOU WANT PARTIAL PARTICIPATION
# n_workers = 20
n_seeds = 1
# batch_size = None
# batch_size = 1

# permutation = A[:, 0].A.squeeze().argsort()
permutation = b.squeeze().argsort()

losses = []
idx = [0] + [(n * i) // n_workers for i in range(1, n_workers)] + [n]
for i in range(n_workers):
    idx_i = permutation[idx[i] : idx[i+1]]
    # idx_i = range(idx[i], idx[i + 1])
    loss_i = LogisticRegression(A[idx_i].A, b[idx_i], l1=0, l2=l2)
    loss_i.computed_grads = 0
    losses.append(loss_i)

grad_norms = [np.linalg.norm(loss_i.gradient(loss.x_opt))**2 for loss_i in losses]
print(np.mean(grad_norms))


@ray.remote
class Worker:
    def __init__(self, method, loss=None, it_local=None, batch_size=1):
        self.loss = loss
        self.prox_skip, self.sprox_skip, self.vr_prox_skip = False, False, False

        if method == 'prox_skip':
            self.prox_skip = True
        elif method == 'sprox_skip':
            self.sprox_skip = True
        elif method == 'vr_prox_skip':
            self.vr_prox_skip = True
        else:
            raise ValueError(f'Unknown method {method}!')

        self.it_local = it_local
        self.batch_size = batch_size
        self.c = None
        self.h = None
        self.rng_skip = np.random.default_rng(42)  # random number generator for random synchronizations
        self.rng_vr_skip = np.random.default_rng(50)

    def run_local(self, x, lr):
        self.x = x * 1.
        if self.prox_skip:
            lsteps = self.run_prox_skip(lr)
        elif self.sprox_skip:
            lsteps = self.run_sprox_skip(lr)
        elif self.vr_prox_skip:
            lsteps = self.run_vr_prox_skip(lr)
        return self.x, lsteps

    def run_prox_skip(self, lr):

        if choose_p == 'local':
            p = 1 / self.it_local
        elif choose_p == 'kappa':
            kappa = oL / l2
            p = 1 / np.sqrt(kappa)
        else:
            raise ValueError(f'Unrecognized chosen p {choose_p}!')

        if self.h is None:
            # first iteration
            self.h = self.x * 0.  # initialize zero vector of the same dimension
        else:
            # update the gradient estimate
            self.h += p / lr * (self.x - self.x_before_averaing)

        # since all workers use the same random seed, this number is the same for all of them
        it_local = self.rng_skip.geometric(p=p)

        for i in range(it_local):
            g = self.loss.gradient(self.x)
            self.x -= lr * (g - self.h)
        self.x_before_averaing = self.x * 1.

        return it_local

    def run_vr_prox_skip(self, lr):
        """ Variance reduced ProxSkip.
        lr: learning rate
        """
        lr = lr / 6
        kappa = L / l2
        q = 4 / kappa

        if choose_p == 'local':
            p = 1 / self.it_local
        elif choose_p == 'kappa':
            p = 1 / np.sqrt(kappa)
        else:
            raise ValueError(f'Unrecognized chosen p {choose_p}!')

        if self.h is None:
            # first iteration
            self.h = self.x * 0.  # initialize zero vector of the same dimension
            self.y = copy.deepcopy(self.x)
        else:
            # update the gradient estimate
            self.h += p / lr * (self.x - self.x_before_averaing)
            mchoice = np.random.choice(2, 1, p=[q, 1 - q])
            if not mchoice:
                self.y = copy.deepcopy(self.x)

        it_local = self.rng_skip.geometric(
            p=p)  # since all workers use the same random seed, this number is the same for all of them

        self.vr_proxskip_it_local = it_local

        full_g_y = self.loss.gradient(self.y)

        for i in range(it_local):
            g_x, idx = self.loss.stochastic_gradient(self.x, batch_size=self.batch_size, rng=self.rng_vr_skip, return_idx=True)
            g_y = self.loss.stochastic_gradient(self.y, batch_size=self.batch_size, rng=self.rng_vr_skip, idx=idx)
            g = g_x - g_y + full_g_y
            self.x -= lr * (g - self.h)
        self.x_before_averaing = self.x * 1.

        return it_local

    def get_control_var(self):
        return self.c


class VR_ProxSkip(StochasticOptimizer):
    """
    Stochastic gradient descent with decreasing or constant learning rate.

    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
        lr_decay_coef (float, optional): the coefficient in front of the number of finished iterations
            in the denominator of step-size. For strongly convex problems, a good value
            is mu/2, where mu is the strong convexity constant
        lr_decay_power (float, optional): the power to exponentiate the number of finished iterations
            in the denominator of step-size. For strongly convex problems, a good value is 1 (default: 1)
        it_start_decay (int, optional): how many iterations the step-size is kept constant
            By default, will be set to have about 2.5% of iterations with the step-size equal to lr0
        batch_size (int, optional): the number of samples from the function to be used at each iteration
    """

    def __init__(self, it_local, n_workers=None, cohort_size=None, iid=False, lr0=None, lr_max=np.inf, lr_decay_coef=0,
                 lr_decay_power=1, it_start_decay=None, batch_size=1, losses=None, *args, **kwargs):
        super(VR_ProxSkip, self).__init__(*args, **kwargs)
        self.it_local = it_local
        if n_workers is None:
            n_workers = psutil.cpu_count(logical=False)
        if cohort_size is None:
            cohort_size = n_workers
        self.n_workers = n_workers
        self.cohort_size = cohort_size
        self.iid = iid
        self.lr0 = lr0
        self.lr_max = lr_max
        self.lr_decay_coef = lr_decay_coef
        self.lr_decay_power = lr_decay_power
        self.it_start_decay = it_start_decay
        self.batch_size = batch_size
        self.losses = losses
        self.vr_proxskip_it_locals = []

    def step(self):
        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.it_local * self.lr_decay_coef * max(0,
                                                                                 self.it - self.it_start_decay) ** self.lr_decay_power)
        if lr_decayed < 0:
            lr_decayed = np.inf
        self.lr = min(lr_decayed, self.lr_max)
        x_id = ray.put(self.x)

        if self.cohort_size == self.n_workers:
            self.x, vr_proxskip_it_local = np.mean(ray.get([worker.run_local.remote(x_id, self.lr) for worker in self.workers]), axis=0)
            self.vr_proxskip_it_locals.append(vr_proxskip_it_local)
        else:
            cohort = np.random.choice(self.n_workers, self.cohort_size, replace=False)
            self.x, vr_proxskip_it_local = np.mean(ray.get([self.workers[i].run_local.remote(x_id, self.lr) for i in cohort]), axis=0)
            self.vr_proxskip_it_locals.append(vr_proxskip_it_local)

    def init_run(self, *args, **kwargs):
        super(VR_ProxSkip, self).init_run(*args, **kwargs)
        if self.it_start_decay is None and np.isfinite(self.it_max):
            self.it_start_decay = self.it_max // 40 if np.isfinite(self.it_max) else 0
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.batch_smoothness(batch_size)
        if self.iid:
            loss_id = ray.put(self.loss)
            self.workers = [
                Worker.remote(method='vr_prox_skip', loss=loss_id, it_local=self.it_local, batch_size=self.batch_size)
                for _ in range(self.n_workers)]
        else:
            loss_ids = [ray.put(self.losses[i]) for i in range(self.n_workers)]
            self.workers = [
                Worker.remote(method='vr_prox_skip', loss=loss, it_local=self.it_local, batch_size=self.batch_size) for
                loss in loss_ids]

    def update_trace(self, first_iterations=10):
        super(VR_ProxSkip, self).update_trace()

    def terminate_workers(self):
        for worker in self.workers:
            ray.kill(worker)


class ProxSkip(StochasticOptimizer):
    """
    Stochastic gradient descent with decreasing or constant learning rate.

    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
        lr_decay_coef (float, optional): the coefficient in front of the number of finished iterations
            in the denominator of step-size. For strongly convex problems, a good value
            is mu/2, where mu is the strong convexity constant
        lr_decay_power (float, optional): the power to exponentiate the number of finished iterations
            in the denominator of step-size. For strongly convex problems, a good value is 1 (default: 1)
        it_start_decay (int, optional): how many iterations the step-size is kept constant
            By default, will be set to have about 2.5% of iterations with the step-size equal to lr0
        batch_size (int, optional): the number of samples from the function to be used at each iteration
    """

    def __init__(self, it_local, n_workers=None, cohort_size=None, iid=False, lr0=None, lr_max=np.inf, lr_decay_coef=0,
                 lr_decay_power=1, it_start_decay=None, batch_size=1, losses=None, *args, **kwargs):
        super(ProxSkip, self).__init__(*args, **kwargs)
        self.it_local = it_local
        if n_workers is None:
            n_workers = psutil.cpu_count(logical=False)
        if cohort_size is None:
            cohort_size = n_workers
        self.n_workers = n_workers
        self.cohort_size = cohort_size
        self.iid = iid
        self.lr0 = lr0
        self.lr_max = lr_max
        self.lr_decay_coef = lr_decay_coef
        self.lr_decay_power = lr_decay_power
        self.it_start_decay = it_start_decay
        self.batch_size = batch_size
        self.losses = losses
        self.proxskip_it_locals = []

    def step(self):
        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.it_local * self.lr_decay_coef * max(0,
                                                                                 self.it - self.it_start_decay) ** self.lr_decay_power)
        if lr_decayed < 0:
            lr_decayed = np.inf
        self.lr = min(lr_decayed, self.lr_max)
        x_id = ray.put(self.x)

        if self.cohort_size == self.n_workers:
            self.x, proxskip_it_local = np.mean(ray.get([worker.run_local.remote(x_id, self.lr) for worker in self.workers]), axis=0)
            self.proxskip_it_locals.append(proxskip_it_local)
        else:
            cohort = np.random.choice(self.n_workers, self.cohort_size, replace=False)
            self.x, proxskip_it_local = np.mean(ray.get([self.workers[i].run_local.remote(x_id, self.lr) for i in cohort]), axis=0)
            self.proxskip_it_locals.append(proxskip_it_local)

    def init_run(self, *args, **kwargs):
        super(ProxSkip, self).init_run(*args, **kwargs)
        if self.it_start_decay is None and np.isfinite(self.it_max):
            self.it_start_decay = self.it_max // 40 if np.isfinite(self.it_max) else 0
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.batch_smoothness(batch_size)
        if self.iid:
            loss_id = ray.put(self.loss)
            self.workers = [
                Worker.remote(method='prox_skip', loss=loss_id, it_local=self.it_local, batch_size=self.batch_size) for
                _ in range(self.n_workers)]
        else:
            loss_ids = [ray.put(self.losses[i]) for i in range(self.n_workers)]
            self.workers = [
                Worker.remote(method='prox_skip', loss=loss, it_local=self.it_local, batch_size=self.batch_size) for
                loss in loss_ids]

    def update_trace(self, first_iterations=10):
        super(ProxSkip, self).update_trace()

    def terminate_workers(self):
        for worker in self.workers:
            ray.kill(worker)


###########################################################################################
# Running methods
###########################################################################################

skip_lr0 = 1 / oL
skip_decay_coef = 0. # With full gradients, we don't need to decrease the stepsize
skip_lr_max = skip_lr0
skip = ProxSkip(loss=loss, n_workers=n_workers, cohort_size=cohort_size, it_local=it_local,
               lr_max=skip_lr_max, lr0=skip_lr0, lr_decay_coef=skip_decay_coef,
               it_start_decay=0, n_seeds=n_seeds, batch_size=batch_size, trace_len=trace_len, iid=False, losses=losses)
skip.run(x0=x0, it_max=it_max)
proxskip_it_locals = skip.proxskip_it_locals
# sgd.trace.convert_its_to_epochs(batch_size=batch_size*it_local)
proxskip_loss_vals = skip.trace.compute_loss_of_iterates(return_loss_vals=True)
skip.terminate_workers()

 
vr_skip_lr0 = 1 / L
vr_skip_decay_coef = 0. # With full gradients, we don't need to decrease the stepsize
vr_skip_lr_max = vr_skip_lr0
vr_skip = VR_ProxSkip(loss=loss, n_workers=n_workers, cohort_size=cohort_size, it_local=it_local,
               lr_max=vr_skip_lr_max, lr0=vr_skip_lr0, lr_decay_coef=vr_skip_decay_coef,
               it_start_decay=0, n_seeds=n_seeds, batch_size=batch_size, trace_len=trace_len, iid=False, losses=losses)
vr_skip.run(x0=x0, it_max=it_max)
vr_proxskip_it_locals = vr_skip.vr_proxskip_it_locals
vr_proxskip_loss_vals = vr_skip.trace.compute_loss_of_iterates(return_loss_vals=True)
vr_skip.terminate_workers()

costs = [1e-4, 1e-3, 3e-3, 6e-3, 1e-2, 3e-2, 6e-2, 1e-1, 3e-1, 6e-1, 1, 3, 6, 1e1, 1e2, 1e3]
it_stop_cost_proxskip = it_stop_cost_vr_proxskip = 0
for i in range(it_max):
    curErr = proxskip_loss_vals[i] - loss.f_opt
    if curErr <= cerr or i == trace_len - 1:
        it_stop_cost_proxskip = i
        break
for j in range(it_max):
    curErr = vr_proxskip_loss_vals[j] - loss.f_opt
    if curErr <= cerr or j == trace_len - 1:
        it_stop_cost_vr_proxskip = j
        break
if it_stop_cost_proxskip == 0: it_stop_cost_proxskip = it_max - 1
if it_stop_cost_vr_proxskip == 0: it_stop_cost_vr_proxskip = it_max - 1

size = 30
# marker_size = 10
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'FreeSerif'
# plt.rcParams['lines.linewidth'] = 3
# plt.rcParams['lines.markersize'] = 10
plt.rcParams['xtick.labelsize'] = 20  # 40
plt.rcParams['ytick.labelsize'] = 20  # 40
plt.rcParams['legend.fontsize'] = 20  # 30
plt.rcParams['axes.titlesize'] = 22  # 40
plt.rcParams['axes.labelsize'] = 22  # 40
plt.rcParams["figure.figsize"] = [13, 9]
markevery = max(1, it_max//100)
plt.yscale('log')
plt.ylabel(r'$f(x)-f_*$')
plt.xlabel('Communication rounds')
plt.grid()

save_name2 = f"0019_{dataset}_n{n_workers}_bs{batch_size}_cosize{cohort_size}_cc{com_cost}_{choose_p}_kappa{L / l2}_error_{cerr}"
saved_log2_nm = f'./logs/{save_name2}.txt'
with open(f'{saved_log2_nm}', 'w') as output:
    output.write(str(it_stop_cost_proxskip) + ',' + str(it_stop_cost_vr_proxskip) + '\n')
    for element in proxskip_it_locals:
        output.write(str(element) + ',')
    output.write('\n')
    for element in vr_proxskip_it_locals:
        output.write(str(element) + ',')
    output.write('\n')

expected_lstepss = [0, 1]
for expected_lsteps in expected_lstepss:
    my = []
    for local_cost in costs:
        kappa1 = L / l2
        kappa2 = oL / l2
        q = 4 / kappa1
        p1 = 1 / np.sqrt(kappa1)
        p2 = 1 / np.sqrt(kappa2)
        m = int(n / n_workers)
        if expected_lsteps:
            skip_cost = (com_cost + local_cost * m / p2) * it_stop_cost_proxskip * p2
            vr_skip_cost = (com_cost + local_cost * (q * m + (2 - q) * batch_size) / p1) * it_stop_cost_vr_proxskip * p1
        else:
            skip_cost = com_cost * it_stop_cost_proxskip + \
                        np.sum(proxskip_it_locals[:it_stop_cost_proxskip]) * local_cost * int(n / n_workers)
            vr_skip_cost = com_cost * it_stop_cost_vr_proxskip + \
                           np.sum(vr_proxskip_it_locals[:it_stop_cost_vr_proxskip]) * local_cost * batch_size

        my.append(float(skip_cost/vr_skip_cost))

    save_name = f"0009_{dataset}_n{n_workers}_bs{batch_size}_cosize{cohort_size}_cc{com_cost}_{choose_p}_kappa{L/l2}_error_{cerr}_exp{expected_lsteps}"
    saved_log_nm = f'./logs/{save_name}.txt'
    saved_pdf_nm = f'./outputs/{save_name}'

    with open(f'{saved_log_nm}', 'w') as output:
        for element in costs:
            # if costs.index(element) == len(costs) - 1:
            #     output.write(str(element))
            # else:
            output.write(str(element) + ',')
        output.write('\n')
    with open(f'{saved_log_nm}', 'a+') as output:
        for element in my:
            # if costs.index(element) == len(costs) - 1:
            #     output.write(str(element))
            # else:
            output.write(str(element) + ',')


    # plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Computation cost per sample')
    plt.ylabel('Total cost ratio')  # proxskip / vr_proxskip
    # plt.legend()
    plt.grid()
    plt.plot(costs, my, marker='x')
    plt.savefig(f'{saved_pdf_nm}.pdf')
    plt.show()
    plt.close()