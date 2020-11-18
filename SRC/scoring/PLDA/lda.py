import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from scipy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDA:
    def __init__(self, ndim):
        self.ndim = ndim

    def fit(self, X, spk_ids):
        self.lda_mat = _LDA(X, spk_ids).get_lda_matrix(self.ndim)
        # self.lda_mat = _ClassBlanceLDA(X, spk_ids).get_lda_matrix(self.ndim)
        return self

    def transform(self, X):
        return X.dot(self.lda_mat)

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)


class _LDA:
    def __init__(self, X, y):
        y = LabelEncoder().fit_transform(y)
        self.stat = defaultdict(dict)
        for uni in np.unique(y):
            self.stat[uni]['X'] = X[y == uni]
            self.stat[uni]['means'] = self.stat[uni]['X'].mean(0)
            self.stat[uni]['N'] = len(X)

        self.feat_dim = X.shape[-1]
        self.n_cls = len(self.stat)
        self.global_mean = X.mean(0)
        Xc = X - self.global_mean
        self.total_cov = (Xc.T @ Xc) / Xc.shape[0]

    def get_lda_matrix(self, n_dim):
        Sw = self.compute_within_group_cov()
        Sb = self.total_cov - Sw

        evals, evecs = linalg.eigh(Sb, Sw)
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
        evecs /= np.linalg.norm(evecs, axis=0)
        return evecs[:, :n_dim]

    def compute_within_group_cov(self):
        cov_within = np.zeros((self.feat_dim, self.feat_dim))
        for stat in self.stat.values():
            X = stat['X'] - stat['means']
            cov_within += (X.T @ X) / X.shape[0]
        return cov_within / self.n_cls


class _ClassBlanceLDA:
    def __init__(self, X, y):
        print('using class balance LDA')
        y = LabelEncoder().fit_transform(y)
        self.stat = defaultdict(dict)
        for uni in np.unique(y):
            self.stat[uni]['X'] = X[y == uni]
            self.stat[uni]['means'] = self.stat[uni]['X'].mean(0)
            self.stat[uni]['N'] = len(X)

        self.feat_dim = X.shape[-1]
        self.n_cls = len(self.stat)
        self.global_mean = X.mean(0)

        # Xc = X - self.global_mean
        # self.total_cov = (Xc.T @ Xc) / Xc.shape[0]

        cov_tot_sum = np.zeros((X.shape[-1], X.shape[-1]))
        uni_y = np.unique(y)
        for uni in uni_y:
            Xc = X[y == uni] - self.global_mean
            cov_tot_sum += (Xc.T @ Xc) / Xc.shape[0]

        self.total_cov = cov_tot_sum / len(uni_y)

    def get_lda_matrix(self, n_dim):
        Sw = self.compute_within_group_cov()
        Sb = self.total_cov - Sw

        evals, evecs = linalg.eigh(Sb, Sw)
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
        evecs /= np.linalg.norm(evecs, axis=0)
        return evecs[:, :n_dim]

    def compute_within_group_cov(self):
        cov_within = np.zeros((self.feat_dim, self.feat_dim))
        for stat in self.stat.values():
            X = stat['X'] - stat['means']
            cov_within += (X.T @ X) / X.shape[0]
        return cov_within / self.n_cls

