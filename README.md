# VR-ProxSkip
This is the official code repository for NeurIPS 2022 paper: [Variance Reduced ProxSkip: Algorithm, Theory and Application to Federated Learning](https://arxiv.org/abs/2207.04338)

## Requirement
We run each experiment with 64G CPU and Nvidia A100-80G GPU. The estimated running time for each experiments is 30min ~ 6hs (mostly relying on the number of maximum communication rounds). 
 
We are required to install the following packages

```
python=3.8
optmethods
ray
psutil
numpy
matplotlib
copy
argparse
scikit-learn
urllib
other common package
```
 
After that, we need to modify one of the intalled optmethods functions in python package, Main change is to modify this [function](https://github.com/konstmish/opt_methods/blob/master/optmethods/opt_trace.py#L142) with the following code (should be in anaconda3/envs/$PROJECT_NAME$/lib/python3.8/site-packages/optmethods):

```
def compute_loss_of_iterates(self, return_loss_vals=False):
    for seed, loss_vals in self.loss_vals_all.items():
        if loss_vals is None:
            self.loss_vals_all[seed] = np.asarray([self.loss.value(x) for x in self.xs_all[seed]])
        else:
            warnings.warn("""Loss values for seed {} have already been computed. 
                Set .loss_vals_all[{}] = [] to recompute.""".format(seed, seed))
    self.loss_is_computed = True
    if return_loss_vals:
        return self.loss_vals_all[seed]
```

## Convergence Analysis
We provide the code to compare VR-ProxSkip with baselines in `main_0002.py'. One optional script to run it is:

```
python main_0002.py --batch_size 16 --dataset 'a9a' --it_local 20 --choose_p 'kappa' --it_max 10001
```

## Total Cost Ratio
The example script to obtain the total cost ratio:

```
python main_0004.py --batch_size 16 --cerr 1e-8 --regul 5e-4 --it_max 8000 --dataset 'a9a'
```

## ProxSkip-QLSVRG
The example script to run ProxSkip-QLSVRG:

```
python main_0007.py --batch_size 16 --dataset 'a9a' --it_local 20 --choose_p 'kappa' --it_max 15001 --k 11
```

## Citation
@article{vrproxskip2022,
  title={Variance reduced proxskip: Algorithm, theory and application to federated learning},
  author={Malinovsky, Grigory and Yi, Kai and Richt{\'a}rik, Peter},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={15176--15189},
  year={2022}
}

## Acknowledgement
This repo contains VR-ProxSkip related implementations building on top of [opt_methods](https://github.com/konstmish/opt_methods).

