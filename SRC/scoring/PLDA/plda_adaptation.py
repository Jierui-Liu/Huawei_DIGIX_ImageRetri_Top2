import scipy.linalg as la
import numpy as np
from core.utils import log_det4psd


def plda_adaptation(favor_indomain_within, favor_indomain_between, plda_in, plda_out):
    sig_spk_in, sig_noise_in = get_two_cov(plda_in)
    sig_spk_out, sig_noise_out = get_two_cov(plda_out)
    sig_spk = favor_indomain_within * sig_spk_in + (1 - favor_indomain_within) * sig_spk_out
    sig_tot = favor_indomain_between * sig_noise_in + (1 - favor_indomain_between) * sig_noise_out
    return comp_pq(sig_spk, sig_tot)


def get_two_cov(plda):
    sig_spk = plda.W.T @ plda.W
    sig_noise = plda.sigma
    return sig_spk, sig_noise


def comp_pq(sig_spk, sig_tot):
    prec_tot = la.inv(sig_tot)
    aux = la.inv(sig_tot - sig_spk @ prec_tot @ sig_spk)
    B0 = np.zeros_like(sig_tot)
    M1 = np.block([[sig_tot, sig_spk], [sig_spk, sig_tot]])
    M2 = np.block([[sig_tot, B0], [B0, sig_tot]])
    P = aux @ sig_spk @ prec_tot
    Q = prec_tot - aux
    const = 0.5 * (-log_det4psd(M1) + log_det4psd(M2))
    return {'P': P, 'Q': Q, 'const': const}
