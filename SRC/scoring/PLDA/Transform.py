import numpy as np
from core.utils import lennorm
from scipy.linalg import inv
from scipy import linalg as la
# from sklearn.decomposition import PCA
from scipy.linalg import sqrtm
from scipy.linalg import eig
# from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import math


class Coral:
    def __init__(self):
        pass

    def fit_transform(self, X_src, X_tgt, mode='both'):
        S_inv_sqrt = sqrtm(inv(np.cov(X_src.T)))
        T_sqrt = sqrtm(np.cov(X_tgt.T))
        transfrom = S_inv_sqrt.dot(T_sqrt)
        X_src = X_src.dot(transfrom)
        return X_src

# class Whiten:
#     def __init__(self):
#         pass
#
#     def fit(self, X_tgt):
#         self.matrix = sqrtm(inv(np.cov(X_tgt.T)))
#         return self
#
#     def transform(self, X_src):
#         return X_src.dot(self.matrix)


def get_whiten_matrix(X):
    Xcov = np.cov(X.T)
    d, V = np.linalg.eigh(Xcov)
    D = np.diag(1. / np.sqrt(d))
    return np.dot(V, D)


class Whiten:
    def __init__(self):
        pass

    def fit(self, X):
        Xcov = np.dot(X.T, X) / X.shape[0]

        d, V = np.linalg.eigh(Xcov)

        D = np.diag(1. / np.sqrt(d))
        self.matrix = np.dot(V, D)

        return self

    def transform(self, X_src):
        return X_src.dot(self.matrix)


# class Whiten_diag:
#     def __init__(self):
#         pass
#
#     def fit(self, X):
#         Xcov = np.diag(np.diag(np.dot(X.T, X) / X.shape[0]))
#
#         d, V = np.linalg.eigh(Xcov)
#
#         D = np.diag(1. / np.sqrt(d))
#         self.matrix = np.dot(V, D)
#
#         return self
#
#     def transform(self, X_src):
#         return X_src.dot(self.matrix)


# class Whiten_diag_two:
#     def __init__(self):
#         pass
#
#     def fit(self, X, Y):
#         Xcov = np.diag(np.diag(np.dot(X.T, X) / X.shape[0]))
#         Ycov = np.diag(np.diag(np.dot(Y.T, Y) / Y.shape[0]))
#         cov = (Xcov + Ycov) / 2
#
#         d, V = np.linalg.eigh(cov)
#
#         D = np.diag(1. / np.sqrt(d))
#         self.matrix = np.dot(V, D)
#
#         return self
#
#     def transform(self, X_src):
#         return X_src.dot(self.matrix)

class Whiten_multiple:
    def __init__(self):
        pass

    def fit(self, Xs):
        cov = []
        for X in Xs:
            cov.append(np.dot(X.T, X) / X.shape[0])
        cov = np.stack(cov).mean(0)
        # Xcov = np.dot(X.T, X) / X.shape[0]
        # Ycov = np.dot(Y.T, Y) / Y.shape[0]
        # cov = (Xcov + Ycov) / 2

        d, V = np.linalg.eigh(cov)

        D = np.diag(1. / np.sqrt(d))
        self.matrix = np.dot(V, D)

        return self

    def transform(self, X_src):
        return X_src.dot(self.matrix)


class Whiten_two:
    def __init__(self):
        pass

    def fit(self, X, Y):
        Xcov = np.dot(X.T, X) / X.shape[0]
        Ycov = np.dot(Y.T, Y) / Y.shape[0]
        cov = (Xcov + Ycov) / 2

        d, V = np.linalg.eigh(cov)

        D = np.diag(1. / np.sqrt(d))
        self.matrix = np.dot(V, D)

        return self

    def transform(self, X_src):
        return X_src.dot(self.matrix)


def whiten_by_target_data(X_source, X_target):
    # X is a NxM data matrix, where N is the
    # number of examples, M is the number of features

    # compute covarince matrix of target data
    X_target_cov = np.dot(X_target.T, X_target) / X_target.shape[0]

    # compute eigen of the covariance matrix
    # w is the eignvalues, V is the eighvectors matrix
    w, V = np.linalg.eigh(X_target_cov)

    # compute inverse sqare root of eigen value
    D = np.diag(1. / np.sqrt(w))

    # get whiten matrix
    whiten_matrix = np.dot(V, D)

    # whiten source data by whiten_matrix
    return X_source.dot(whiten_matrix)


def whiten_by_target_data_diag(X_source, X_target):
    # X is a NxM data matrix, where N is the
    # number of examples, M is the number of features

    # compute covarince matrix of target data
    X_target_cov = np.diag(np.diag(np.dot(X_target.T, X_target) / X_target.shape[0]))

    # compute eigen of the covariance matrix
    # w is the eignvalues, V is the eighvectors matrix
    w, V = np.linalg.eigh(X_target_cov)

    # compute inverse sqare root of eigen value
    D = np.diag(1. / np.sqrt(w))

    # get whiten matrix
    whiten_matrix = np.dot(V, D)

    # whiten source data by whiten_matrix
    return X_source.dot(whiten_matrix)

# def whiten_by_target_data(X_source, X_target):
#     # X is a NxM data matrix, where N is the
#     # number of examples, M is the number of features
#
#     # compute covarince matrix of target data
#     X_target_center = X_target - X_target.mean(0)
#     X_target_cov = np.dot(X_target_center.T, X_target_center) / X_target_center.shape[0]
#
#     # compute eigen of the covariance matrix
#     # w is the eignvalues, V is the eighvectors matrix
#     w, V = np.linalg.eigh(X_target_cov)
#
#     # compute inverse sqare root of eigen value
#     D = np.diag(1. / np.sqrt(w))
#
#     # get whiten matrix
#     whiten_matrix = np.dot(V, D)
#
#     # whiten source data by whiten_matrix
#     return X_source.dot(whiten_matrix)


# class MomentsMatcher:
#     def __init__(self):
#         pass
#
#     def fit_transform(self, X_src, X_tgt, mode='both'):
#         S_inv_sqrt = sqrtm(inv(np.cov(X_src.T)))
#         T_sqrt = sqrtm(np.cov(X_tgt.T))
#         transfrom = S_inv_sqrt.dot(T_sqrt)
#         X_src = X_src.dot(transfrom)
#         if mode == 'both':
#             return X_src - X_src.mean(0) + X_tgt.mean(0)
#         elif mode == 'second':
#             return X_src
#         else:
#             raise NotImplemented

        # self.target_mean = X_tgt.mean(0)
        # self.train_flag = True
        # return self

    # def fit(self, source, target):
    #     S_inv_sqrt = sqrtm(inv(np.cov(source.T)))
    #     T_sqrt = sqrtm(np.cov(target.T))
    #     self.transfrom = S_inv_sqrt.dot(T_sqrt)
    #
    #     self.target_mean = target.mean(0)
    #     self.train_flag = True
    #     return self
    #
    # def transform(self, source):
    #     return source.dot(self.transfrom)
    #
    # def fit_transform(self, source, target):
    #     self.fit(source, target)
    #     return self.transform(source)


# class CORAL:
#     def __init__(self):
#         pass
#
#     def fit(self, source, target):
#         S_inv_sqrt = sqrtm(inv(np.cov(source.T)))
#         T_sqrt = sqrtm(np.cov(target.T))
#         self.transfrom = S_inv_sqrt.dot(T_sqrt)
#         self.train_flag = True
#         return self
#
#     def transform(self, source):
#         return source.dot(self.transfrom)
#
#     def fit_transform(self, source, target):
#         self.fit(source, target)
#         return self.transform(source)


# class IDVC:
#     def __init__(self, n_components=None):
#         self.n_components = n_components
#
#     def fit(self, X, y):
#         means = []
#         for y_uni in np.unique(y):
#             means.append(X[y == y_uni].mean(0))
#         means = np.stack(means)
#         means = means - means.mean(0)
#         _, _, Vh = la.svd(means, full_matrices=False)
#         self.proj = np.eye(X.shape[-1]) - Vh.T @ Vh
#         if self.n_components is not None:
#             self.pca = PCA().fit(X @ self.proj)
#         return self
#
#     def transform(self, X):
#         if self.n_components is not None:
#             return self.pca.transform(X @ self.proj)
#         else:
#             return X @ self.proj
#
#     def fit_transform(self, X, y):
#         self.fit(X, y)
#         return self.transform(X)


class ClassBalanceDemean:
    def __init__(self):
        pass

    def fit(self, X, y):
        means = []
        for uni in np.unique(y):
            means.append(X[y == uni].mean(0))
        self.mean = np.stack(means).mean(0)
        return self

    def transform(self, X):
        return X - self.mean

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class Demean:
    def __init__(self):
        pass

    def fit(self, X):
        self.mean = np.mean(X, 0)
        return self

    def transform(self, X):
        return X - self.mean

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class Lennorm:
    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, X):
        return lennorm(X)

    def fit_transform(self, X, y=None):
        self.fit()
        return self.transform(X)


class NewLennorm:
    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, X):
        return lennorm(X) / math.sqrt(X.shape[-1])

    def fit_transform(self, X, y=None):
        self.fit()
        return self.transform(X)

