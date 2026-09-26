"""The two reported simulation designs, with original seed mappings."""
import numpy as np

def make_one_triplet(config):
    rng = np.random.default_rng(config['data_seed'])
    p,rho = config['p'],config['rho']
    chosen = int(rng.integers(3))
    permutation = rng.permutation(p)
    reverse = np.argsort(permutation)
    block = reverse[:3]
    def design(n):
        raw = rng.normal(size=(n,p))
        common = rng.normal(size=n)
        raw[:,:3] = np.sqrt(rho)*common[:,None]+np.sqrt(1-rho)*raw[:,:3]
        return raw[:,permutation]
    raw = design(config['n'])
    center,scale = raw.mean(0),raw.std(0)
    x = (raw-center)/scale
    original_beta = np.zeros(p)
    original_beta[[chosen,3,4,5,6]] = config['signal']*np.array([1,-1,1,-1,1])
    beta = original_beta[permutation]
    y = x@beta+config['sigma']*rng.normal(size=config['n'])
    test = (design(1000)-center)/scale
    return x,y,beta,test,block


def make_two_triplets(c):
    rng = np.random.default_rng(c['data_seed'])
    chosen = rng.integers(3, size=2)
    permutation = rng.permutation(c['p'])
    inverse = np.argsort(permutation)
    groups = inverse[:6].reshape(2, 3)
    def design(n):
        raw = rng.normal(size=(n, c['p']))
        common = rng.normal(size=(n, 2))
        for g in range(2):
            raw[:, 3*g:3*g+3] = (np.sqrt(c['rho'])*common[:, g, None]
                                       + np.sqrt(1-c['rho'])*raw[:, 3*g:3*g+3])
        return raw[:, permutation]
    raw = design(c['n'])
    center, scale = raw.mean(0), raw.std(0)
    x = (raw-center)/scale
    original_beta = np.zeros(c['p'])
    original_beta[[chosen[0], 3+chosen[1], 6, 7, 8]] = c['signal']*np.array([1,-1,1,-1,1])
    truth = original_beta[permutation]
    y = x@truth+c['sigma']*rng.normal(size=c['n'])
    test = (design(1000)-center)/scale
    return dict(x=x, y=y, truth=truth, test=test, groups=groups,
                center=center, scale=scale, permutation=permutation)
