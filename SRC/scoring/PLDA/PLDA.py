import numpy as np
from numpy.random import randn
import h5py as h5
import scipy.linalg as linalg
# from my_nn.my_functions import GaussianDistribution as Gauss
from core.utils import GaussianDistribution as Gauss
from scipy.linalg import eigvalsh, inv
import pickle


class PLDA:
    def __init__(self, transforms, dev_files=None,n_fac=150, n_iter=20, print_llh=False):
        self.dev_files = dev_files if type(dev_files) is list else [dev_files]
        self.n_fac = n_fac
        self.n_iter = n_iter
        self.llh_lst = [-np.inf]
        self.transforms = transforms
        self.print_llh = print_llh

    @property
    def prec(self):
        return inv(self.sigma)

    def fit(self, X=None, spk_ids=None):
        if X is None:
            X, spk_ids = self.load_data()
        for transform in self.transforms:
            if hasattr(transform, 'train_flag'):
                X = transform.transform(X)
                print('warning one transform is not fitted by internal data')
            else:
                X = transform.fit_transform(X, spk_ids)
        self.W= randn(self.n_fac, X.shape[-1])
        self.sigma = abs(randn()) * np.eye(X.shape[-1])
        xstat = suff_xstats(X, spk_ids)
        for _ in range(self.n_iter):
            zstat = self.e_step(xstat)
            if self.print_llh:
                self.comp_llh(xstat, zstat, mode='elbo')
            self.m_step(xstat, zstat)
        return self

    def load_data(self):
        X, spk_ids = [], []
        for file in self.dev_files:
            print('load dev data from {}'.format(file))
            with h5.File(file, 'r') as f:
                X.append(f['X'][:])
                spk_ids.append(f['spk_ids'][:])
        X = np.concatenate(X, axis=0)
        spk_ids = np.concatenate(spk_ids, axis=0)
        return X, spk_ids

    def e_step(self, x):
        WtP = self.prec @ self.W.T
        WtPW = self.W @ WtP
        n_id = len(x['ns_obs'])
        mu_post = np.zeros((n_id, self.n_fac))
        sigma_post = np.zeros((n_id, self.n_fac, self.n_fac))
        for i_id, (X_homo_sum, n_ob) in enumerate(zip(x['homo_sums'], x['ns_obs'])):
            sigma_post[i_id] = inv(np.eye(self.n_fac) + n_ob * WtPW)
            mu_post[i_id] = X_homo_sum @ WtP @ sigma_post[i_id]
        mu_mom2s = np.einsum('Bi,Bj->Bij', mu_post, mu_post) + sigma_post
        return {'mom1s': mu_post, 'mom2s': mu_mom2s}

    def m_step(self, x, z):
        z_mom2s_sum = np.einsum('B,Bij->ij', x['ns_obs'], z['mom2s'])
        xz_cmom = z['mom1s'].T @ x['homo_sums']
        self.W = inv(z_mom2s_sum) @ xz_cmom
        self.sigma = (x['mom2'] - xz_cmom.T @ self.W) / x['ns_obs'].sum()

    def comp_llh(self, xstat, zstat, mode='elbo', dev=None, spk_ids=None,):
        if mode == 'elbo':
            llh = self.elbo(xstat, zstat)
        else:
            llh = exact_marginal_llh(
                dev=dev, idens=spk_ids, W=self.W, sigma=self.sigma,)
        self._display_llh(llh)

    def elbo(self, xstat, zstat):
        WtPW = self.W @ self.prec @ self.W.T
        return - _ce_cond_xs(xstat, zstat, self.W, self.prec) \
               - _ce_prior(zstat) \
               + _entropy_q(xstat['ns_obs'], WtPW)

    def _display_llh(self, llh):
        self.llh_lst.append(llh)
        if self.llh_lst[-2] == -np.inf:
            print('llh = {:.4f} increased inf\n'.format(llh))
        else:
            margin = self.llh_lst[-1] - self.llh_lst[-2]
            change_percent = 100 * np.abs(margin / self.llh_lst[-2])
            print('llh = {:.4f} {} {:.4f}%\n'.format(
                llh, 'increased' if margin > 0 else 'decreased', change_percent,))

    def comp_pq(self):
        sig_ac = self.W.T @ self.W
        sig_tot = sig_ac + self.sigma
        prec_tot = inv(sig_tot)
        aux = inv(sig_tot - sig_ac @ prec_tot @ sig_ac)
        B0 = np.zeros_like(self.sigma)
        M1 = np.block([[sig_tot, sig_ac], [sig_ac, sig_tot]])
        M2 = np.block([[sig_tot, B0], [B0, sig_tot]])
        P = aux @ sig_ac @ prec_tot
        Q = prec_tot - aux
        const = 0.5 * (-log_det4psd(M1) + log_det4psd(M2))
        return {'P': P, 'Q': Q, 'const': const}

    def comp_pq_using_within(self, within, without):
        sig_ac = within
        sig_tot = within + without
        prec_tot = inv(sig_tot)
        aux = inv(sig_tot - sig_ac @ prec_tot @ sig_ac)
        B0 = np.zeros_like(sig_tot)
        M1 = np.block([[sig_tot, sig_ac], [sig_ac, sig_tot]])
        M2 = np.block([[sig_tot, B0], [B0, sig_tot]])
        P = aux @ sig_ac @ prec_tot
        Q = prec_tot - aux
        const = 0.5 * (-log_det4psd(M1) + log_det4psd(M2))
        return {'P': P, 'Q': Q, 'const': const}

    def save_model(self, save_file_to):
        model = self.comp_pq()
        model_dict = {
            'model': model,
            'transform_lst': self.transforms,
        }
        with open(save_file_to, 'wb') as f:
            pickle.dump(model_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('save model to {}'.format(save_file_to))


def suff_xstats(X, spk_ids):
    # X -= X.mean(0)
    mom2 = X.T @ X
    unique_ids, ns_obs = np.unique(spk_ids, return_counts=True)
    homo_sums = np.zeros((unique_ids.shape[0], X.shape[-1]))
    for i_id, unique_id in enumerate(unique_ids):
        homo_sums[i_id] = np.sum(X[spk_ids == unique_id], axis=0)
    return {'mom2': mom2, 'homo_sums': homo_sums, 'ns_obs': ns_obs}


_LOG_2PI = np.log(2 * np.pi)


def _ce_cond_xs(x, z, W, prec):
    dim = prec.shape[-1]
    N = x['ns_obs'].sum()
    xy_cmom = W.T @ z['mom1s'].T @ x['homo_sums']
    z_mom2s_wsum = np.einsum('B,Bij->ij', x['ns_obs'], z['mom2s'])
    dev_mom2 = x['mom2'] - xy_cmom - xy_cmom.T + W.T @ z_mom2s_wsum @ W
    return 0.5 * (N * dim * _LOG_2PI
                  - N * log_det4psd(prec)
                  + ravel_dot(dev_mom2, prec))


def _ce_prior(z):
    n_ids, dim = z['mom1s'].shape
    return 0.5 * (n_ids * dim * _LOG_2PI
                  + np.einsum('Bii->', z['mom2s']))


def _entropy_q(ns_obs, WtPW):
    n_ids = len(ns_obs)
    zdim = WtPW.shape[0]
    # due to the special form of posterior co logdet can be greatly simplified
    eigvals = np.outer(ns_obs, eigvalsh(WtPW)) + 1
    log_det_sum = np.sum(np.log(1 / eigvals))
    return 0.5 * (n_ids * zdim * _LOG_2PI
                  + log_det_sum
                  + n_ids * zdim)


def log_det4psd(sigma):
    return 2 * np.sum(np.log(np.diag(linalg.cholesky(sigma))))


def ravel_dot(X, Y):
    return X.ravel() @ Y.ravel()


def exact_marginal_llh(dev, idens, W, sigma):
    # this is very computation intensive op should only be used to
    # check whether low-bound is correct on toy data
    # stake mu is 0, diag of cov is sigma + WWt, off-diag is WWt
    llh = 0.0
    unique_ids = np.unique(idens)
    for unique_id in unique_ids:
        dev_homo = dev[idens == unique_id]
        cov = _construct_marginal_cov(W.T @ W, sigma, dev_homo.shape[0])
        llh += Gauss(cov=cov).log_p(dev_homo.ravel())
    return llh


def _construct_marginal_cov(heter_cov, noise_cov, n_obs):
    cov = np.tile(heter_cov, (n_obs, n_obs))
    rr, cc = noise_cov.shape
    r, c = 0, 0
    for _ in range(n_obs):
        cov[r:r+rr, c:c+cc] += noise_cov
        r += rr
        c += cc
    return cov
