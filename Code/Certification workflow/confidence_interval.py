import numpy as np
from scipy.special import beta, betainc
from scipy.stats import norm


def F(x, d1, d2):
    return betainc(d1 * x / (d1 * x + d2), d1 / 2, d2 / 2) / beta(d1 / 2, d2 / 2)


def compute_series(outcomes):
    s1, s2, s3, s4, s5 = 0, 0, 0, 0, 0
    for outcome in outcomes:
        qi = outcome["Security ensured probability"]
        zi = np.random.rand() < qi
        pi = qi if zi == 1 else 1 - qi
        s1 = s1 + zi
        s2 = s2 + (1 - pi)
        s3 = s3 + 2 * pi - 1
        s4 = s4 + (2 * pi - 1) ** 2
        s5 = s5 + pi * (1 - pi)
    Z_n = (s1 - s2) / s3
    V_n = s3 ** 2 / s4
    sigma2_n = s5 / s4
    return Z_n, V_n, sigma2_n


def compute_confidence_interval_wald(confidence_interval_level, outcomes):
    q                   = norm.ppf(1 - (1 - confidence_interval_level) / 2)
    Z_n, V_n, sigma2_n  = compute_series(outcomes)
    lower_bound         = Z_n - q * np.sqrt(Z_n * (1 - Z_n) / V_n + sigma2_n / V_n)
    upper_bound         = Z_n + q * np.sqrt(Z_n * (1 - Z_n) / V_n + sigma2_n / V_n)
    confidence_interval = {"Lower bound" : lower_bound, "Upper bound" : upper_bound}
    return confidence_interval


def compute_confidence_interval_wilson(confidence_interval_level, outcomes):
    q                   = norm.ppf(1 - (1 - confidence_interval_level) / 2)
    Z_n, V_n, sigma2_n  = compute_series(outcomes)
    lower_bound         = 1 / (1 + q**2 / V_n) * (Z_n + 0.5 * q**2 / V_n - q * np.sqrt(Z_n * (1 - Z_n) / V_n + sigma2_n / V_n * (1 + q**2 / V_n) + 0.25 * q**2 / V_n**2))
    upper_bound         = 1 / (1 + q**2 / V_n) * (Z_n + 0.5 * q**2 / V_n + q * np.sqrt(Z_n * (1 - Z_n) / V_n + sigma2_n / V_n * (1 + q**2 / V_n) + 0.25 * q**2 / V_n**2))
    confidence_interval = {"Lower bound" : lower_bound, "Upper bound" : upper_bound}
    return confidence_interval


def compute_confidence_interval_continuity_correction(confidence_interval_level, outcomes):
    q                   = norm.ppf(1 - (1 - confidence_interval_level) / 2)
    Z_n, V_n, sigma2_n  = compute_series(outcomes)
    lower_bound         = max(0, 1 / (1 + q**2 / V_n) * (Z_n + 0.5 * q**2 / V_n) - (1 + q * np.sqrt(Z_n * (1 - Z_n) / V_n + sigma2_n / V_n * (1 + q**2 / V_n) + 0.25 * q**2 / V_n**2 + (4 * Z_n - 2 - 1 / V_n) / (4 * V_n))) / (1 + q**2 / V_n))
    upper_bound         = min(1, 1 / (1 + q**2 / V_n) * (Z_n + 0.5 * q**2 / V_n) + (1 + q * np.sqrt(Z_n * (1 - Z_n) / V_n + sigma2_n / V_n * (1 + q**2 / V_n) + 0.25 * q**2 / V_n**2 + (4 * Z_n - 2 - 1 / V_n) / (4 * V_n))) / (1 + q**2 / V_n))
    confidence_interval = {"Lower bound" : lower_bound, "Upper bound" : upper_bound}
    return confidence_interval


def compute_confidence_interval_clopper_pearson(confidence_interval_level, outcomes):
    Z_n, V_n, _  = compute_series(outcomes)
    lower_bound         = 1 / (1 + (V_n - Z_n * V_n + 1) / ( Z_n * V_n      * F(    (1 - confidence_interval_level) / 2, 2 *  Z_n * V_n     , 2 * (V_n - Z_n * V_n + 1))))
    upper_bound         = 1 / (1 + (V_n - Z_n * V_n    ) / ((Z_n * V_n + 1) * F(1 - (1 - confidence_interval_level) / 2, 2 * (Z_n * V_n + 1), 2 * (V_n - Z_n * V_n    ))))
    confidence_interval = {"Lower bound" : lower_bound, "Upper bound" : upper_bound}
    return confidence_interval


def compute_confidence_interval_jeffreys(confidence_interval_level, outcomes):
    Z_n, V_n, _  = compute_series(outcomes)
    lower_bound = betainc(    (1 - confidence_interval_level) / 2, Z_n * V_n + 0.5, V_n - Z_n * V_n + 0.5)
    upper_bound = betainc(1 - (1 - confidence_interval_level) / 2, Z_n * V_n + 0.5, V_n - Z_n * V_n + 0.5)
    confidence_interval = {"Lower bound" : lower_bound, "Upper bound" : upper_bound}
    return confidence_interval
