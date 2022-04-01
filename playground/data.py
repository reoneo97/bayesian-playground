from sklearn.datasets import make_regression
import random

def regression_dataset(n_samples=100):
    bias = random.uniform(-3,3)
    return make_regression(n_samples, n_features = 1,n_informative=1,coef=True,
                           bias=bias, noise=1.5)
