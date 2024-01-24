""""
Convergence Analysis with different methods
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
parser.add_argument('--k', type=int, default=1, help='choose k in rand-k for DIANA')
parser.add_argument('--kappa', type=int, default=10000)
args = parser.parse_args()

batch_size = args.batch_size
n_workers = args.nworkers
cohort_size = args.cohort_size
it_local = args.it_local
it_max = args.it_max
dataset = args.dataset
choose_p = args.choose_p
kappa = args.kappa

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

n, dim = A.shape 
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
L_max = loss.max_smoothness
# l2 = 0
l2 = L / kappa    # here we choose the regularizer to be 1e-5, in paper 1e-4
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
    def __init__(self, method=None, loss=None, it_local=None, batch_size=1):
        self.loss = loss
        self.prox_skip, self.sprox_skip, self.vr_prox_skip = False, False, False
        self.s_local_svrg, self.local_sgd, self.fedlin, self.scaffold, self.shuffle = \
            False, False, False, False, False

        if method == 'prox_skip':
            self.prox_skip = True
        elif method == 'sprox_skip':
            self.sprox_skip = True
        elif method == 'vr_prox_skip':
            self.vr_prox_skip = True
        # elif method == 'diana_prox_skip':
        #     self.diana_prox_skip = True
        elif method == 's_local_svrg':
            self.s_local_svrg = True
        elif method == 'local_sgd':
            self.local_sgd = True
        elif method == 'fedlin':
            self.fedlin = True
        elif method == 'scaffold':
            self.scaffold = True
        elif method == 'shuffle':
            self.shuffle = True
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

    def run_local(self, x, lr, whole=0, return_whole=False):
        self.x = x * 1.
        if self.prox_skip:
            self.run_prox_skip(lr)
        elif self.sprox_skip:
            self.run_sprox_skip(lr)
        elif self.vr_prox_skip:
            self.run_vr_prox_skip(lr)
        # elif self.diana_prox_skip:
        #     self.run_diana_prox_skip(lr)
        elif self.s_local_svrg:
            if return_whole:
                return self.run_s_local_svrg(lr, return_whole=True)
            else:
                self.run_s_local_svrg(lr, whole_y=whole)
        elif self.shuffle:
            self.run_local_shuffle(lr)
        elif self.local_sgd:
            self.run_local_sgd(lr)
        return self.x

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

    def run_sprox_skip(self, lr):
        if choose_p == 'local':
            p = 1 / self.it_local
        elif choose_p == 'kappa':
            kappa = L / l2
            p = 1 / np.sqrt(kappa)
        else:
            raise ValueError(f'Unrecognized chosen p {choose_p}!')

        if self.h is None:
            # first iteration
            self.h = self.x * 0.  # initialize zero vector of the same dimension
        else:
            # update the gradient estimate
            self.h += p / lr * (self.x - self.x_before_averaing)
        it_local = self.rng_skip.geometric(p=p)

        for i in range(it_local):
            g = self.loss.stochastic_gradient(self.x, batch_size=self.batch_size)
            self.x -= lr * (g - self.h)
        self.x_before_averaing = self.x * 1.

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
            g_x, idx = self.loss.stochastic_gradient(self.x, batch_size=self.batch_size, rng=self.rng_vr_skip, return_idx=True)
            g_y = self.loss.stochastic_gradient(self.y, batch_size=self.batch_size, rng=self.rng_vr_skip, idx=idx)
            g = g_x - g_y + full_g_y
            self.x -= lr * (g - self.h)
        self.x_before_averaing = self.x * 1.

    def run_s_local_svrg(self, lr, whole_y=0, return_whole=False):
        if return_whole:
            if self.y is None:
                self.y = copy.deepcopy(self.x)
            return self.loss.gradient(self.y)

        kappa = L / l2
        q = 1 / int(n / n_workers)
        p = 1 / np.sqrt(kappa)
        t1 = 56 * L_max / (3 * n) + 4 * oL + 32 * oL / (3 * n)
        t2 = 32 * np.sqrt(2 * L * (1 - p) * (oL * (2 + p) + p * L_max + 4 * (oL + L_max) * (1 + p) / (1 - q)))
        lr = np.minimum(1 / t1, np.sqrt(3) * p / t2)

        it_local = self.rng_skip.geometric(p=p)

        for i in range(it_local):
            g_x, idx = self.loss.stochastic_gradient(self.x, batch_size=self.batch_size, rng=self.rng_vr_skip, return_idx=True)
            g_y = self.loss.stochastic_gradient(self.y, batch_size=self.batch_size, rng=self.rng_vr_skip, idx=idx)
            g = g_x - g_y + whole_y
            self.x -= lr * g
        self.x_before_averaing = self.x * 1.

        if self.y is None:
            self.y = copy.deepcopy(self.x)
        else:
            mchoice = np.random.choice(2, 1, p=[q, 1 - q])
            if mchoice:
                self.y = copy.deepcopy(self.x)

    def run_scaffold(self, x, lr, c):
        # as in the original scaffold paper, we use their Option II
        self.x = x * 1.
        if self.c is None:
            self.c = self.x * 0. #initialize zero vector of the same dimension
        for i in range(self.it_local):
            if self.batch_size is None:
                g = self.loss.gradient(self.x)
            else:
                g = self.loss.stochastic_gradient(self.x, batch_size=self.batch_size)
            self.x -= lr * (g - self.c + c)
        self.c += 1 / (self.it_local * lr) * (x - self.x) - c
        return self.x

    def run_local_sgd(self, lr):
        for i in range(self.it_local):
            if self.batch_size is None:
                self.x -= lr * self.loss.gradient(self.x)
            else:
                self.x -= lr * self.loss.stochastic_gradient(self.x, batch_size=self.batch_size)

    def run_local_shuffle(self, lr):
        permutation = np.random.permutation(self.loss.n)
        i = 0
        while i < self.loss.n:
            i_max = min(self.loss.n, i + self.batch_size)
            idx = permutation[i:i_max]
            self.x -= lr * self.loss.stochastic_gradient(self.x, idx=idx)
            i += self.batch_size

    def run_fedlin(self, x, lr, g):
        self.x = x * 1.
        for i in range(self.it_local):
            if self.batch_size is None:
                grad = self.loss.gradient(self.x)
            else:
                grad = self.loss.stochastic_gradient(self.x, batch_size=self.batch_size)
            self.x -= lr * (grad - self.g + g)
        return self.x

    def get_control_var(self):
        return self.c

    def rand_k_compressor(self, x, k, ind, idxs=None):
        # RandK compressor with scaling
        output = np.zeros(x.shape)
        dim = x.shape[0]
        # omega = float(dim / k) - 1

        if ind: idxs = np.random.choice(dim, k)

        output[idxs] = x[idxs] * float(dim / k)
        return output

    def get_fedlin_grad(self, x):
        if self.batch_size is None:
            self.g = self.loss.gradient(x)
        else:
            self.g = self.loss.stochastic_gradient(x, batch_size=self.batch_size)
        return self.g


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


class S_Local_SVRG(StochasticOptimizer):
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
        super(S_Local_SVRG, self).__init__(*args, **kwargs)
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
            self.whole_y = np.mean(ray.get([worker.run_local.remote(x_id, self.lr, return_whole=True) for worker in self.workers]), axis=0)
            # print(np.mean(self.whole_y))
            self.x = np.mean(ray.get([worker.run_local.remote(x_id, self.lr, whole=self.whole_y) for worker in self.workers]), axis=0)
        else:
            cohort = np.random.choice(self.n_workers, self.cohort_size, replace=False)
            self.whole_y = np.mean(ray.get([self.workers[i].run_local.remote(x_id, self.lr, return_whole=True) for i in cohort]), axis=0)
            self.x = np.mean(ray.get([self.workers[i].run_local.remote(x_id, self.lr, whole=self.whole_y) for i in cohort]), axis=0)

    def init_run(self, *args, **kwargs):
        super(S_Local_SVRG, self).init_run(*args, **kwargs)
        if self.it_start_decay is None and np.isfinite(self.it_max):
            self.it_start_decay = self.it_max // 40 if np.isfinite(self.it_max) else 0
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.batch_smoothness(batch_size)
        if self.iid:
            loss_id = ray.put(self.loss)
            self.workers = [
                Worker.remote(method='s_local_svrg', loss=loss_id, it_local=self.it_local, batch_size=self.batch_size) for
                _ in range(self.n_workers)]
        else:
            loss_ids = [ray.put(self.losses[i]) for i in range(self.n_workers)]
            self.workers = [
                Worker.remote(method='s_local_svrg', loss=loss, it_local=self.it_local, batch_size=self.batch_size) for
                loss in loss_ids]

    def update_trace(self, first_iterations=10):
        super(S_Local_SVRG, self).update_trace()

    def terminate_workers(self):
        for worker in self.workers:
            ray.kill(worker)


class SProxSkip(StochasticOptimizer):
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
        super(SProxSkip, self).__init__(*args, **kwargs)
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
        super(SProxSkip, self).init_run(*args, **kwargs)
        if self.it_start_decay is None and np.isfinite(self.it_max):
            self.it_start_decay = self.it_max // 40 if np.isfinite(self.it_max) else 0
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.batch_smoothness(batch_size)
        if self.iid:
            loss_id = ray.put(self.loss)
            self.workers = [
                Worker.remote(method='sprox_skip', loss=loss_id, it_local=self.it_local, batch_size=self.batch_size) for
                _ in range(self.n_workers)]
        else:
            loss_ids = [ray.put(self.losses[i]) for i in range(self.n_workers)]
            self.workers = [
                Worker.remote(method='sprox_skip', loss=loss, it_local=self.it_local, batch_size=self.batch_size) for
                loss in loss_ids]

    def update_trace(self, first_iterations=10):
        super(SProxSkip, self).update_trace()

    def terminate_workers(self):
        for worker in self.workers:
            ray.kill(worker)


class Fedlin(StochasticOptimizer):
    """
    Fedlin (local SGD with variance control and linear convergence).

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

    def __init__(self, it_local, n_workers=None, iid=False, lr0=None, lr_max=np.inf,
                 lr_decay_coef=0, lr_decay_power=1, it_start_decay=None,
                 batch_size=1, losses=None, global_lr=1., cohort_size=None, *args, **kwargs):
        super(Fedlin, self).__init__(*args, **kwargs)
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
        self.global_lr = global_lr

    def step(self):
        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.it_local * self.lr_decay_coef * max(0,
                                                                                 self.it - self.it_start_decay) ** self.lr_decay_power)
        if lr_decayed < 0:
            lr_decayed = np.inf
        self.lr = min(lr_decayed, self.lr_max)
        x_id = ray.put(self.x)
        if self.cohort_size != self.n_workers:
            raise ValueError(
                "There is no theory for FedLin with partial participation. This feature is not implemented.")
            cohort = np.random.choice(self.n_workers, self.cohort_size, replace=False)
        g = np.mean(ray.get([worker.get_fedlin_grad.remote(x_id) for worker in self.workers]), axis=0)
        g_id = ray.put(g)
        if self.cohort_size == self.n_workers:
            self.x = np.mean(ray.get([worker.run_fedlin.remote(x_id, self.lr, g_id) for worker in self.workers]),
                             axis=0)
        else:
            self.x = np.mean(ray.get([self.workers[i].run_fedlin.remote(x_id, self.lr, c_id) for i in cohort]), axis=0)

    def init_run(self, *args, **kwargs):
        super(Fedlin, self).init_run(*args, **kwargs)
        if self.it_start_decay is None and np.isfinite(self.it_max):
            self.it_start_decay = self.it_max // 40 if np.isfinite(self.it_max) else 0
        if self.iid:
            loss_id = ray.put(self.loss)
            self.workers = [
                Worker.remote(method='fedlin', loss=loss_id, it_local=self.it_local, batch_size=self.batch_size) for _ in
                range(self.n_workers)]
        else:
            loss_ids = [ray.put(self.losses[i]) for i in range(self.n_workers)]
            self.workers = [Worker.remote(method='fedlin', loss=loss, it_local=self.it_local, batch_size=self.batch_size)
                            for loss in loss_ids]

    def terminate_workers(self):
        for worker in self.workers:
            ray.kill(worker)


class Scaffold(StochasticOptimizer):
    """
    Scaffold (local SGD with variance control).

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

    def __init__(self, it_local, n_workers=None, iid=False, lr0=None, lr_max=np.inf,
                 lr_decay_coef=0, lr_decay_power=1, it_start_decay=None,
                 batch_size=1, losses=None, global_lr=1., cohort_size=None, *args, **kwargs):
        super(Scaffold, self).__init__(*args, **kwargs)
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
        self.global_lr = global_lr

    def step(self):
        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.it_local * self.lr_decay_coef * max(0,
                                                                                 self.it - self.it_start_decay) ** self.lr_decay_power)
        if lr_decayed < 0:
            lr_decayed = np.inf
        self.lr = min(lr_decayed, self.lr_max)
        x_id = ray.put(self.x)
        c_id = ray.put(self.c)
        if self.cohort_size == self.n_workers:
            x_new = np.mean(ray.get([worker.run_scaffold.remote(x_id, self.lr, c_id) for worker in self.workers]),
                            axis=0)
            c_new = np.mean(ray.get([worker.get_control_var.remote() for worker in self.workers]), axis=0)
        else:
            cohort = np.random.choice(self.n_workers, self.cohort_size, replace=False)
            x_new = np.mean(ray.get([self.workers[i].run_scaffold.remote(x_id, self.lr, c_id) for i in cohort]), axis=0)
            c_new = np.mean(ray.get([self.workers[i].get_control_var.remote() for i in cohort]), axis=0)
        if self.global_lr == 1:
            self.x = x_new
        else:
            self.x += self.global_lr * (x_new - self.x)
        self.c += self.cohort_size / self.n_workers * (c_new - self.c)

    def init_run(self, *args, **kwargs):
        super(Scaffold, self).init_run(*args, **kwargs)
        if self.it_start_decay is None and np.isfinite(self.it_max):
            self.it_start_decay = self.it_max // 40 if np.isfinite(self.it_max) else 0
        self.c = self.x * 0
        if self.iid:
            loss_id = ray.put(self.loss)
            self.workers = [
                Worker.remote(method = 'scaffold', loss=loss_id, it_local=self.it_local, batch_size=self.batch_size) for _ in
                range(self.n_workers)]
        else:
            loss_ids = [ray.put(self.losses[i]) for i in range(self.n_workers)]
            self.workers = [Worker.remote(method = 'scaffold', loss=loss, it_local=self.it_local, batch_size=self.batch_size)
                            for loss in loss_ids]

    def terminate_workers(self):
        for worker in self.workers:
            ray.kill(worker)


class LocalSgd(StochasticOptimizer):
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
        super(LocalSgd, self).__init__(*args, **kwargs)
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
        super(LocalSgd, self).init_run(*args, **kwargs)
        if self.it_start_decay is None and np.isfinite(self.it_max):
            self.it_start_decay = self.it_max // 40 if np.isfinite(self.it_max) else 0
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.batch_smoothness(batch_size)
        if self.iid:
            loss_id = ray.put(self.loss)
            self.workers = [
                Worker.remote(method = 'local_sgd', loss=loss_id, it_local=self.it_local, batch_size=self.batch_size) for _ in
                range(self.n_workers)]
        else:
            loss_ids = [ray.put(self.losses[i]) for i in range(self.n_workers)]
            self.workers = [Worker.remote(method = 'local_sgd', loss=loss, it_local=self.it_local, batch_size=self.batch_size)
                            for loss in loss_ids]

    def update_trace(self, first_iterations=10):
        super(LocalSgd, self).update_trace()

    def terminate_workers(self):
        for worker in self.workers:
            ray.kill(worker)


#################################################################
# SProxSkip
#################################################################
sskip_lr0 = 1 / L
sskip_decay_coef = 0. # With full gradients, we don't need to decrease the stepsize
sskip_lr_max = sskip_lr0
sskip = SProxSkip(loss=loss, n_workers=n_workers, cohort_size=cohort_size, it_local=it_local,
               lr_max=sskip_lr_max, lr0=sskip_lr0, lr_decay_coef=sskip_decay_coef,
               it_start_decay=0, n_seeds=n_seeds, batch_size=batch_size, trace_len=trace_len, iid=False, losses=losses)
sskip.run(x0=x0, it_max=it_max)
# sgd.trace.convert_its_to_epochs(batch_size=batch_size*it_local)
sproxskip_loss_vals = sskip.trace.compute_loss_of_iterates(return_loss_vals=True)
sskip.terminate_workers()


#################################################################
# VR_ProxSkip
#################################################################
vr_skip_lr0 = 1 / L
vr_skip_decay_coef = 0. # With full gradients, we don't need to decrease the stepsize
vr_skip_lr_max = vr_skip_lr0
vr_skip = VR_ProxSkip(loss=loss, n_workers=n_workers, cohort_size=cohort_size, it_local=it_local,
               lr_max=vr_skip_lr_max, lr0=vr_skip_lr0, lr_decay_coef=vr_skip_decay_coef,
               it_start_decay=0, n_seeds=n_seeds, batch_size=batch_size, trace_len=trace_len, iid=False, losses=losses)
vr_skip.run(x0=x0, it_max=it_max)
# sgd.trace.convert_its_to_epochs(batch_size=batch_size*it_local)
vr_proxskip_loss_vals = vr_skip.trace.compute_loss_of_iterates(return_loss_vals=True)
vr_skip.terminate_workers()


#################################################################
# S_Local_SVRG
#################################################################
s_local_svrg_lr0 = 1 / L
s_local_svrg_decay_coef = 0. # With full gradients, we don't need to decrease the stepsize
s_local_svrg_lr_max = s_local_svrg_lr0
s_local_svrg = S_Local_SVRG(loss=loss, n_workers=n_workers, cohort_size=cohort_size, it_local=it_local,
               lr_max=s_local_svrg_lr_max, lr0=s_local_svrg_lr0, lr_decay_coef=s_local_svrg_decay_coef,
               it_start_decay=0, n_seeds=n_seeds, batch_size=batch_size, trace_len=trace_len, iid=False, losses=losses)
s_local_svrg.run(x0=x0, it_max=it_max)
# sgd.trace.convert_its_to_epochs(batch_size=batch_size*it_local)
s_local_svrg_loss_vals = s_local_svrg.trace.compute_loss_of_iterates(return_loss_vals=True)
s_local_svrg.terminate_workers()


#################################################################
# Local SGD
#################################################################
sgd_decay_coef = l2 / 2
sgd_lr0 = 1 / loss.smoothness
sgd_lr_max = sgd_lr0
sgd = LocalSgd(loss=loss, n_workers=n_workers, cohort_size=cohort_size, it_local=it_local,
               lr_max=sgd_lr_max, lr0=sgd_lr0, lr_decay_coef=sgd_decay_coef,
               it_start_decay=0, n_seeds=n_seeds, batch_size=batch_size, trace_len=trace_len, iid=False, losses=losses)
sgd.run(x0=x0, it_max=it_max)
# sgd.trace.convert_its_to_epochs(batch_size=batch_size*it_local)
sgd_loss_vals = sgd.trace.compute_loss_of_iterates(return_loss_vals=True)
sgd.terminate_workers()


#################################################################
# FedLin
#################################################################
fedlin_lr0 = 1 / loss.smoothness
fedlin_lr_max = fedlin_lr0
fedlin_decay_coef = 0
fedlin = Fedlin(loss=loss, n_workers=n_workers, it_local=it_local, lr_max=fedlin_lr_max,
                lr0=fedlin_lr0, lr_decay_coef=fedlin_decay_coef, it_start_decay=0,
                n_seeds=n_seeds, batch_size=batch_size, trace_len=trace_len, iid=False, losses=losses)
fedlin.run(x0=x0, it_max=it_max)
fedlin_loss_vals = fedlin.trace.compute_loss_of_iterates(return_loss_vals=True)
fedlin.terminate_workers()


#################################################################
# Scaffold
#################################################################
scah_lr0 = 1 / loss.smoothness / it_local
scah_decay_coef = l2 / 2
scah_lr_max = scah_lr0
scah = Scaffold(loss=loss, n_workers=n_workers, it_local=it_local, lr_max=scah_lr_max, lr0=scah_lr0, lr_decay_coef=scah_decay_coef,
               it_start_decay=0, n_seeds=n_seeds, batch_size=batch_size, trace_len=trace_len, iid=True)
scah.run(x0=x0, it_max=it_max)
scah_loss_vals = scah.trace.compute_loss_of_iterates(return_loss_vals=True)
scah.terminate_workers()

kappa = int(L / l2)
save_name = f"0007_{dataset}_n{n_workers}_bs{batch_size}_cosize{cohort_size}_lsteps{it_local}_{choose_p}_{kappa}"
saved_log_nm = f'./logs/{save_name}.txt'
saved_pdf_nm = f'./outputs/{save_name}'


mcases = [sproxskip_loss_vals - loss.f_opt,
          vr_proxskip_loss_vals - loss.f_opt,
          s_local_svrg_loss_vals - loss.f_opt,
          sgd_loss_vals - loss.f_opt,
          scah_loss_vals - loss.f_opt,
          fedlin_loss_vals - loss.f_opt]
# mcases = [s_local_svrg_loss_vals - loss.f_opt]

with open(f'{saved_log_nm}', 'w') as output:
    for element in sskip.trace.its:
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
markevery = max(1, len(mcases[2])//20) * 10
plt.yscale('log')
plt.ylabel(r'$f(x)-f_*$')
plt.xlabel('Communication rounds')
plt.grid()
# # sskip.trace.plot_losses(label='SProxSkip', marker='+')
# # skip.trace.plot_losses(label='ProxSkip', marker='x')
# # vr_skip.trace.plot_losses(label='VR_ProxSkip', marker='*')
plt.plot(sgd.trace.its, mcases[3], label='Local SGD', marker='o', markevery=markevery)
plt.plot(scah.trace.its, mcases[4], label='Scaffold', marker='v', markevery=markevery)
plt.plot(fedlin.trace.its, mcases[5], label='FedLin', marker='8', markevery=markevery)
plt.plot(s_local_svrg.trace.its, mcases[2], label='S_Local_SVRG', marker='s', markevery=markevery, color='#7f7f7f')
plt.plot(sskip.trace.its, mcases[0], label='SProxSkip', marker='p', markevery=markevery, color='#e377c2')
plt.plot(vr_skip.trace.its, mcases[1], label='VR_ProxSkip', marker='*', markevery=markevery, color='red')
# plt.plot(s_local_svrg.trace.its, mcases[3], label='', markevery=markevery)
plt.legend()
plt.savefig(f'{saved_pdf_nm}_{it_max}.pdf')
plt.show()
plt.close()



