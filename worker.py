import ray
import numpy as np

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
        self.rng_mskip = np.random.default_rng(45)
        self.rng_vr_skip = np.random.default_rng(50)

    def run_local(self, x, lr):
        self.x = x * 1.
        if self.prox_skip:
            self.run_prox_skip(lr)
        elif self.sprox_skip:
            self.run_sprox_skip(lr)
        elif self.vr_prox_skip:
            self.run_vr_prox_skip(lr)
        return self.x

    def run_prox_skip(self, lr):
        p = 1 / self.it_local
        kappa = L / l2
        #         p = 1 / np.sqrt(kappa)
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
        p = 1 / self.it_local
        kappa = L / l2
        #         p = 1 / np.sqrt(kappa)
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
        #         p = 1 / np.sqrt(kappa)
        q = 2 / kappa
        p = 1 / self.it_local

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

        for i in range(it_local):
            g_x = self.loss.stochastic_gradient(self.x, batch_size=self.batch_size)
            g_y = self.loss.stochastic_gradient(self.y, batch_size=self.batch_size)
            full_g_y = self.loss.gradient(self.y)
            g = g_x - g_y + full_g_y
            self.x -= lr * (g - self.h)
        self.x_before_averaing = self.x * 1.

    def get_control_var(self):
        return self.c