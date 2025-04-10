
import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import entropy

def jensen_shannon_divergence(raw_data, sample_data):
    """
        Calculate the Jensen-Shannon divergence between 
        two probability distributions
    """
    original_hist, _ = np.histogram(raw_data, bins=10, density=True)
    sampled_hist, _ = np.histogram(sample_data, bins=10, density=True)
    m = 0.5 * (original_hist + sampled_hist)
    return 0.5 * (entropy(original_hist, m) + entropy(sampled_hist, m))



def ks_test(raw_data, sample_data, alpha = 0.05):
    """apply ks_test for two sets of data
    https://www.cnblogs.com/arkenstone/p/5496761.html

    Args:
        raw_data : the original data
        sample_data: sampled data from raw_data
        alpha : Threshold. Defaults to 0.05.

    Returns:
        ks_statistic: KS test statistic.
        from_same_dist: Determine whether two sets of data 
            come from the same distribution
    """
    sd=False
    ks_statistic, ks_p_value = ks_2samp(raw_data, sample_data)
    if ks_p_value < alpha:
        sd=False
    else:
        sd=True
    return ks_statistic, sd