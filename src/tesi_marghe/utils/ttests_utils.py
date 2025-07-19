import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import polars as pl
from scipy.stats import norm

# ----------  helper functions  ----------
def cohens_d(v1, v2):
    n1, n2   = len(v1), len(v2)
    s1, s2   = v1.std(ddof=1), v2.std(ddof=1)
    s_pool   = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2))
    return (v1.mean() - v2.mean()) / s_pool

def hedges_g(d, n1, n2):
    J = 1 - 3 / (4*(n1+n2-2) - 1)
    return d * J, J

def ci_from_d(d, n1, n2, alpha=0.05):
    z  = norm.ppf(1 - alpha/2)
    se = np.sqrt((n1 + n2)/(n1*n2) + d**2 / (2*(n1 + n2 - 2)))
    return d - z*se, d + z*se

def magnitude_label(g):
    a = abs(g)
    return ("negligible" if a < .20 else
            "small"       if a < .50 else
            "medium"      if a < .80 else
            "large")

def compute_group_comparison_table(data):
    """
    Computes t-test and effect size statistics for all numeric variables in data,
    comparing 'San Raffaele' vs 'Casilino' for each variable except 'centro_studi'.
    Returns a pandas DataFrame with the results.
    """
    # Extract variable names, excluding 'centro_studi'
    variabili = [col for col in data.columns if col != 'centro_studi']

    rows = []
    for var in variabili:
        v1 = (data.filter(pl.col.centro_studi == 'San Raffaele')
                  .select(var).drop_nulls().to_numpy().flatten())
        v2 = (data.filter(pl.col.centro_studi == 'Casilino')
                  .select(var).drop_nulls().to_numpy().flatten())

        # Skip if either group is empty or not numeric
        if len(v1) == 0 or len(v2) == 0:
            continue
        try:
            t, p     = ttest_ind(v1, v2, equal_var=True)
            d        = cohens_d(v1, v2)
            g, _     = hedges_g(d, len(v1), len(v2))
            ci_low, ci_high = ci_from_d(g, len(v1), len(v2))
            magnitude = magnitude_label(g)
        except Exception:
            # Skip non-numeric or problematic columns
            continue

        rows.append(dict(variable   = var,
                         t          = t,
                         p          = p,
                         cohen_d    = d,
                         hedges_g   = g,
                         ci_low     = ci_low,
                         ci_high    = ci_high,
                         magnitude  = magnitude))

    table = pd.DataFrame(rows).set_index('variable')
    pd.options.display.float_format = lambda x: f'{x:.3g}'  # 6.09e-11, 0.776, 0.023 …
    return pl.from_pandas(table.reset_index()).sort('magnitude')