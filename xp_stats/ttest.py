from typing import Union

import numpy as np
from dataclasses import dataclass
from scipy.stats import t

from xp_stats import constants as C
from scipy.stats import ttest_ind


@dataclass
class TTestResult:
    """
    Represents the result of a T-test.

    Attributes:
        lift: The observed lift or difference in means between the two groups.
        pvalue: The p-value resulting from the T-test, indicating the significance of the results.
        ci_lower: The lower bound of the confidence interval for the effect size.
        ci_upper: The upper bound of the confidence interval for the effect size.
    """
    lift: float
    pvalue: float
    ci_lower: float
    ci_upper: float


def ttest_from_stats(
        mean1: Union[float, np.array],
        std1: Union[float, np.array],
        nobs1:  Union[int, np.array],
        mean2:  Union[float, np.array],
        std2: Union[float, np.array],
        nobs2:  Union[int, np.array],
        alpha: float = 0.05,
        alternative: str = C.TWO_SIDED_ALTERNATIVE,
        relative: bool = False
):
    vn1 = std1 ** 2 / nobs1
    vn2 = std2 ** 2 / nobs2

    if not relative:
        lift = mean1 - mean2
        std_err = np.sqrt(vn1 + vn2)
    else:
        lift = mean2 / mean1 - 1
        std_err = np.sqrt((vn1 + vn2) / mean1**2 + vn1*(mean2 - mean1)**2 / mean1**4 + 2*vn1*(mean2 - mean1) / mean1**3)

    df = (vn1 + vn2)**2 / (vn1**2/(nobs1 - 1) + vn2**2/(nobs2 - 1))
    t_stat = lift / std_err

    # P-value
    if alternative == C.TWO_SIDED_ALTERNATIVE:
        pvalue = t.cdf(x=-np.abs(t_stat), df=df) * 2
    elif alternative == C.GREATER_ALTERNATIVE:
        pvalue = 1 - t.cdf(x=t_stat, df=df)
    elif alternative == C.LESS_ALTERNATIVE:
        pvalue = t.cdf(x=t_stat, df=df)

    # Confidence Interval
    ci_lower, ci_upper = np.nan, np.nan
    if alternative == C.TWO_SIDED_ALTERNATIVE:
        t_crit = t.ppf(q=1-alpha/2, df=df)
        ci_lower = lift - t_crit * std_err
        ci_upper = lift + t_crit * std_err
    elif alternative == C.GREATER_ALTERNATIVE:
        t_crit = t.ppf(q=1-alpha, df=df)
        ci_lower = lift - t_crit * std_err
    elif alternative == C.LESS_ALTERNATIVE:
        t_crit = t.ppf(q=1-alpha, df=df)
        ci_upper = lift + t_crit * std_err

    return TTestResult(
        lift=lift,
        pvalue=pvalue,
        ci_lower=ci_lower,
        ci_upper=ci_upper
    )


def ttest(
        sample1: np.array,
        sample2: np.array,
        alternative: str = C.TWO_SIDED_ALTERNATIVE,
        alpha: float = 0.05,
        relative: bool = False
) -> TTestResult:
    mean1, std1, nobs1 = np.mean(sample1), np.std(sample1, ddof=1), len(sample1)
    mean2, std2, nobs2 = np.mean(sample2), np.std(sample2, ddof=1), len(sample2)

    return ttest_from_stats(
        mean1=mean1,
        std1=std1,
        nobs1=nobs1,
        mean2=mean2,
        std2=std2,
        nobs2=nobs2,
        alternative=alternative,
        alpha=alpha,
        relative=relative
    )


if __name__ == '__main__':
    control1 = np.random.normal(loc=1, scale=1, size=1000)
    control2 = np.random.normal(loc=1.2, scale=1, size=1000)

    print(ttest_ind(a=control1, b=control2, equal_var=False, alternative='two-sided'))
    print(ttest(control1, control2, alternative='two-sided', relative=True))
