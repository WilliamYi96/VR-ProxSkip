""""
ProxSkip-QLSVRG method.
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
import math

import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.datasets import load_svmlight_file, fetch_rcv1

from optmethods.datasets import get_dataset
from optmethods.first_order import Adgd, Gd, Nesterov, RestNest
from optmethods.loss import LogisticRegression
from optmethods.utils import get_trace, relative_round
from optmethods.optimizer import StochasticOptimizer

# from method import ProxSkip, SProxSkip, VR_ProxSkip

# Arguments
parser = argparse.ArgumentParser(description='Variance-Reduced ProxSkip.')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--nworkers', type=int, default=10, help='number of workers')
parser.add_argument('--it_local', type=int, default=20)
parser.add_argument('--it_max', type=int, default=20001)
parser.add_argument('--cohort_size', type=int, default=10)
parser.add_argument('--dataset', type=str, default='w8a')
parser.add_argument('--choose_p', type=str, default='local', help='Choose from local and kappa')
parser.add_argument('--k', type=int, default=1, help='choose k in rand-k for DIANA, w8a: 300, a9a: 123')
parser.add_argument('--kappa', type=int, default=2000)
parser.add_argument('--ratio', type=float, default=1, help='learning rate scaling ratio')
args = parser.parse_args()

batch_size = args.batch_size
n_workers = args.nworkers
cohort_size = args.cohort_size
it_local = args.it_local
it_max = args.it_max
dataset = args.dataset
choose_p = args.choose_p
kappa = args.kappa
k = args.k
ratio = args.ratio

if cohort_size != n_workers:
    print("You're doing partial participation!")

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
    b = (b + 1) / 2
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
L_max = loss.max_smoothness
# l2 = 0
l2 = L / kappa  # here we choose the regularizer to be 1e-5, in paper 1e-4
loss.l2 = l2
# x0 = csc_matrix((dim, 1))
x0 = np.zeros(dim)
n_epoch = 1000
# it_max = (n_epoch * n) // batch_size
trace_len = 200
# print(L, l2)

###########################################################################################
# Solve problem by Nesterov's method
###########################################################################################
rest = RestNest(loss=loss, doubling=True)
rest_tr = rest.run(x0=x0, it_max=10001)
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
    idx_i = permutation[idx[i]: idx[i + 1]]
    # idx_i = range(idx[i], idx[i + 1])
    loss_i = LogisticRegression(A[idx_i].A, b[idx_i], l1=0, l2=l2)
    loss_i.computed_grads = 0
    losses.append(loss_i)

grad_norms = [np.linalg.norm(loss_i.gradient(loss.x_opt)) ** 2 for loss_i in losses]
print(np.mean(grad_norms))


@ray.remote
class Worker:
    def __init__(self, method=None, loss=None, it_local=None, batch_size=1):
        self.loss = loss
        self.prox_skip, self.sprox_skip, self.vr_prox_skip = False, False, False
        self.s_local_svrg, self.local_sgd, self.fedlin, self.scaffold, self.shuffle = \
            False, False, False, False, False
        self.lsvrg, self.qlsvrg = False, False

        if method == 'lsvrg':
            self.lsvrg = True
        elif method == 'qlsvrg':
            self.qlsvrg = True
        else:
            raise ValueError(f'Unknown method {method}!')

        self.it_local = it_local
        self.batch_size = batch_size
        self.c = None
        self.h = None
        self.y = None
        self.rng_skip = np.random.default_rng(42)  # random number generator for random synchronizations
        self.rng_mskip = np.random.default_rng(45)
        self.rng_vr_skip = np.random.default_rng(50)

    def run_local(self, x, lr):
        self.x = x * 1.
        if self.lsvrg:
            self.run_lsvrg(lr)
        elif self.qlsvrg:
            self.run_qlsvrg(lr)
        return self.x

    def run_vr_prox_skip(self, lr):
        """ Variance reduced ProxSkip.
        lr: learning rate
        """
        lr = lr / 6
        kappa = L / l2
        # q = 2 / kappa
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

        full_g_y = self.loss.gradient(self.y)

        for i in range(it_local):
            g_x, idx = self.loss.stochastic_gradient(self.x, batch_size=self.batch_size, rng=self.rng_vr_skip,
                                                     return_idx=True)
            g_y = self.loss.stochastic_gradient(self.y, batch_size=self.batch_size, rng=self.rng_vr_skip, idx=idx)
            g = g_x - g_y + full_g_y
            self.x -= lr * (g - self.h)
        self.x_before_averaing = self.x * 1.

    def run_lsvrg(self, lr):
        """ Variance reduced ProxSkip.
        lr: learning rate
        """
        lr = lr / 6
        kappa = L / l2
        # q = 2 / kappa
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

        full_g_y = self.loss.gradient(self.y)

        for i in range(it_local):
            g_x, idx = self.loss.stochastic_gradient(self.x, batch_size=self.batch_size, rng=self.rng_vr_skip,
                                                     return_idx=True)
            g_y = self.loss.stochastic_gradient(self.y, batch_size=self.batch_size, rng=self.rng_vr_skip, idx=idx)
            g = g_x - g_y + full_g_y
            self.x -= lr * (g - self.h)
        self.x_before_averaing = self.x * 1.

    def run_qlsvrg(self, lr):
        """ Variance reduced ProxSkip.
        lr: learning rate
        """
        kappa = L / l2
        # q = 2 / kappa
        q = 4 / kappa

        omega = float(dim / k) - 1
        A = 4 * (L + omega / batch_size * L_max)
        B = 4 * (1 + omega / batch_size)
        C = 0
        tau_A = q * L_max
        tau_B = 1 - q
        tau_C = 0
        W = 1 * B / (1 - tau_B)
        lr = np.minimum(1 / l2, 1 / (A + W * tau_A))

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

        full_g_y = self.loss.gradient(self.y)

        for i in range(it_local):
            _, midx = self.loss.stochastic_gradient(
                self.x, batch_size=self.batch_size, rng=self.rng_vr_skip, return_idx=True)
            g_x = np.array([self.loss.stochastic_gradient(
                self.x, rng=self.rng_vr_skip, batch_size=1, idx=midx[i]) for i in range(len(midx))])
            g_y = np.array([self.loss.stochastic_gradient(
                self.y, rng=self.rng_vr_skip, batch_size=1, idx=midx[i]) for i in range(len(midx))])
            diff_xy = g_x - g_y
            compressed_grad = self.rand_k_matrix(diff_xy, k=k, ind=True)  # bs * d
            # print(diff_xy, compressed_grad)
            sum_comp_grad = np.mean(compressed_grad, 0)
            # sum_comp_grad = np.mean(diff_xy, 0)
            g = sum_comp_grad + full_g_y
            self.x -= lr * (g - self.h)
        self.x_before_averaing = self.x * 1.

    def get_control_var(self):
        return self.c

    def rand_k_matrix(self, X, k, ind=False):
        # print(X.shape)  # (bs, d)
        output = np.zeros(X.shape)

        dim = X.shape[1]
        idxs = None if ind else np.random.choice(dim, k, replace=False)

        for i in range(X.shape[0]):
            output[i] = self.rand_k_compressor(X[i], k, ind, idxs)

        return output

    def rand_k_compressor(self, x, k, ind, idxs=None):
        # RandK compressor
        output = np.zeros(x.shape)
        dim = x.shape[0]
        # omega = float(dim / k) - 1

        if ind: idxs = np.random.choice(dim, k, replace=False)

        output[idxs] = x[idxs] * float(dim / k)
        return output


class LSVRG(StochasticOptimizer):
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
        super(LSVRG, self).__init__(*args, **kwargs)
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

    def step(self):
        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.it_local * self.lr_decay_coef * max(0,
                                                                                 self.it - self.it_start_decay) ** self.lr_decay_power)
        if lr_decayed < 0:
            lr_decayed = np.inf
        self.lr = min(lr_decayed, self.lr_max)
        x_id = ray.put(self.x)

        if self.cohort_size == self.n_workers:
            self.x = np.mean(ray.get([worker.run_local.remote(x_id, self.lr) for worker in self.workers]), axis=0)
        else:
            cohort = np.random.choice(self.n_workers, self.cohort_size, replace=False)
            self.x = np.mean(ray.get([self.workers[i].run_local.remote(x_id, self.lr) for i in cohort]), axis=0)

    def init_run(self, *args, **kwargs):
        super(LSVRG, self).init_run(*args, **kwargs)
        if self.it_start_decay is None and np.isfinite(self.it_max):
            self.it_start_decay = self.it_max // 40 if np.isfinite(self.it_max) else 0
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.batch_smoothness(batch_size)
        if self.iid:
            loss_id = ray.put(self.loss)
            self.workers = [
                Worker.remote(method='lsvrg', loss=loss_id, it_local=self.it_local, batch_size=self.batch_size)
                for _ in range(self.n_workers)]
        else:
            loss_ids = [ray.put(self.losses[i]) for i in range(self.n_workers)]
            self.workers = [
                Worker.remote(method='lsvrg', loss=loss, it_local=self.it_local, batch_size=self.batch_size) for
                loss in loss_ids]

    def update_trace(self, first_iterations=10):
        super(LSVRG, self).update_trace()

    def terminate_workers(self):
        for worker in self.workers:
            ray.kill(worker)


class QLSVRG(StochasticOptimizer):
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
        super(QLSVRG, self).__init__(*args, **kwargs)
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

    def step(self):
        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.it_local * self.lr_decay_coef * max(0,
                                                                                 self.it - self.it_start_decay) ** self.lr_decay_power)
        if lr_decayed < 0:
            lr_decayed = np.inf
        self.lr = min(lr_decayed, self.lr_max)
        x_id = ray.put(self.x)

        if self.cohort_size == self.n_workers:
            self.x = np.mean(ray.get([worker.run_local.remote(x_id, self.lr) for worker in self.workers]), axis=0)
        else:
            cohort = np.random.choice(self.n_workers, self.cohort_size, replace=False)
            self.x = np.mean(ray.get([self.workers[i].run_local.remote(x_id, self.lr) for i in cohort]), axis=0)

    def init_run(self, *args, **kwargs):
        super(QLSVRG, self).init_run(*args, **kwargs)
        if self.it_start_decay is None and np.isfinite(self.it_max):
            self.it_start_decay = self.it_max // 40 if np.isfinite(self.it_max) else 0
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.batch_smoothness(batch_size)
        if self.iid:
            loss_id = ray.put(self.loss)
            self.workers = [
                Worker.remote(method='qlsvrg', loss=loss_id, it_local=self.it_local, batch_size=self.batch_size)
                for _ in range(self.n_workers)]
        else:
            loss_ids = [ray.put(self.losses[i]) for i in range(self.n_workers)]
            self.workers = [
                Worker.remote(method='qlsvrg', loss=loss, it_local=self.it_local, batch_size=self.batch_size) for
                loss in loss_ids]

    def update_trace(self, first_iterations=10):
        super(QLSVRG, self).update_trace()

    def terminate_workers(self):
        for worker in self.workers:
            ray.kill(worker)


#################################################################
# LSVRG_ProxSkip
#################################################################
lsvrg_lr0 = 1 / L
lsvrg_decay_coef = 0.  # With full gradients, we don't need to decrease the stepsize
lsvrg_lr_max = lsvrg_lr0
lsvrg = LSVRG(loss=loss, n_workers=n_workers, cohort_size=cohort_size, it_local=it_local,
              lr_max=lsvrg_lr_max, lr0=lsvrg_lr0, lr_decay_coef=lsvrg_decay_coef,
              it_start_decay=0, n_seeds=n_seeds, batch_size=batch_size, trace_len=trace_len, iid=False, losses=losses)
lsvrg.run(x0=x0, it_max=it_max)
# sgd.trace.convert_its_to_epochs(batch_size=batch_size*it_local)
lsvrg_loss_vals = lsvrg.trace.compute_loss_of_iterates(return_loss_vals=True)
lsvrg.terminate_workers()

#################################################################
# QLSVRG_ProxSkip
#################################################################
qlsvrg_lr0 = 1 / L
kappa = L / l2
q = 4 / kappa
omega = float(dim / k) - 1
A = 4 * (L + omega / batch_size * L_max)
B = 4 * (1 + omega / batch_size)
C = 0
tau_A = q * L_max
tau_B = 1 - q
tau_C = 0
W = 1 * B / (1 - tau_B)
qlsvrg_lr = np.minimum(1 / l2, 1 / (A + W * tau_A))
# Fine-tuning the learning rate
lr = ratio * qlsvrg_lr
if ratio > 1:
    qlsvrg_decay_coef = l2 / 2
else:
    qlsvrg_decay_coef = 0.  # With full gradients, we don't need to decrease the stepsize
qlsvrg_lr_max = lr
qlsvrg = QLSVRG(loss=loss, n_workers=n_workers, cohort_size=cohort_size, it_local=it_local,
                lr_max=qlsvrg_lr_max, lr0=qlsvrg_lr0, lr_decay_coef=qlsvrg_decay_coef,
                it_start_decay=0, n_seeds=n_seeds, batch_size=batch_size, trace_len=trace_len, iid=False, losses=losses)
qlsvrg.run(x0=x0, it_max=it_max)
# sgd.trace.convert_its_to_epochs(batch_size=batch_size*it_local)
qlsvrg_loss_vals = qlsvrg.trace.compute_loss_of_iterates(return_loss_vals=True)
qlsvrg.terminate_workers()

kappa = int(L / l2)
save_name = f"1000_{dataset}_n{n_workers}_bs{batch_size}_cosize{cohort_size}_lsteps{it_local}_{choose_p}_{kappa}_{it_max}_{k}_{ratio}"
saved_log_nm = f'./logs/{save_name}.txt'
saved_pdf_nm = f'./outputs/{save_name}'

mcases = [lsvrg_loss_vals - loss.f_opt, qlsvrg_loss_vals - loss.f_opt]
# mcases = [s_local_svrg_loss_vals - loss.f_opt]

with open(f'{saved_log_nm}', 'w') as output:
    for element in lsvrg.trace.its:
        output.write(str(element) + ',')
    output.write('\n')
with open(f'{saved_log_nm}', 'a+') as output:
    for mcase in mcases:
        for element in mcase:
            output.write(str(element) + ',')
        output.write('\n')

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
markevery = max(1, len(mcases[0]) // 20) * 10
plt.yscale('log')
plt.ylabel(r'$f(x)-f_*$')
plt.xlabel('Communication rounds')
plt.grid()
plt.plot(lsvrg.trace.its, mcases[0], label='ProxSkip-L-SVRG', marker='p', markevery=markevery, color='#e377c2')
plt.plot(qlsvrg.trace.its, mcases[1], label='ProxSkip-QLSVRG', marker='*', markevery=markevery, color='red')
plt.legend()
plt.savefig(f'{saved_pdf_nm}.pdf')
plt.show()
plt.close()
