from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np
from scipy.stats import t

from xp_stats import constants as C


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

    def __eq__(self, other):
        return self.lift == other.lift and self.pvalue == other.pvalue and self.ci_lower == other.ci_lower


def _calculate_pvalue(t_stat: float, df: float, alternative: str) -> float:
    """
    Calculate the p-value based on the t-statistic and degrees of freedom.

    Args:
        t_stat: The calculated t-statistic.
        df: Degrees of freedom.
        alternative: The alternative hypothesis for the test.

    Returns:
        float: The calculated p-value.
    """
    if alternative == C.TWO_SIDED_ALTERNATIVE:
        pvalue = t.cdf(x=-np.abs(t_stat), df=df) * 2
    elif alternative == C.GREATER_ALTERNATIVE:
        pvalue = 1 - t.cdf(x=t_stat, df=df)
    elif alternative == C.LESS_ALTERNATIVE:
        pvalue = t.cdf(x=t_stat, df=df)
    else:
        raise ValueError(f'Invalid alternative: {alternative}. Can be "two-sided", "greater", or "less".')

    return pvalue


def _calculate_confidence_interval(
        lift: float,
        std_err: float,
        df: float,
        alpha: float,
        alternative: str
) -> Tuple[float, float]:
    """
    Calculate the confidence interval based on lift, standard error, degrees of freedom, and alpha.

    Args:
        lift: The calculated lift.
        std_err: The calculated standard error.
        df: Degrees of freedom.
        alpha: Significance level.
        alternative: The alternative hypothesis for the test.

    Returns:
        tuple: A tuple containing the lower and upper bounds of the confidence interval.
    """
    if alternative == C.TWO_SIDED_ALTERNATIVE:
        t_crit = t.ppf(q=1-alpha/2, df=df)
        ci_lower = lift - t_crit * std_err
        ci_upper = lift + t_crit * std_err
    elif alternative == C.GREATER_ALTERNATIVE:
        t_crit = t.ppf(q=1-alpha, df=df)
        ci_lower = lift - t_crit * std_err
        ci_upper = np.nan  # No upper bound for greater alternative
    elif alternative == C.LESS_ALTERNATIVE:
        t_crit = t.ppf(q=1-alpha, df=df)
        ci_lower = np.nan  # No lower bound for less alternative
        ci_upper = lift + t_crit * std_err
    else:
        raise ValueError(f'Invalid alternative: {alternative}. Can be "two-sided", "greater", or "less".')

    return ci_lower, ci_upper


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
    """
    Perform a Welch's t-test based on summary statistics.

    Args:
        mean1: Mean of the first sample.
        std1: Standard deviation of the first sample.
        nobs1: Number of observations in the first sample.
        mean2: Mean of the second sample.
        std2: Standard deviation of the second sample.
        nobs2: Number of observations in the second sample.
        alpha: Significance level for the test. Defaults to 0.05.
        alternative: The alternative hypothesis for the test. Can be 'two-sided',
            'greater', or 'less'. Defaults to 'two-sided'.
        relative: If True, calculate the relative lift between means. Defaults to False.

    Returns:
        TTestResult: An object containing the t-test results including lift, p-value, and confidence interval.

    Raises:
        ValueError: If `alternative` is not one of 'two-sided', 'greater', or 'less'.

    Notes:
        - This function supports both absolute and relative lift calculations.
        - The confidence interval is calculated based on the specified `alpha` level.

    Example:
        result = ttest_from_stats(mean1=10, std1=2, nobs1=100, mean2=12, std2=2.5, nobs2=120)
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f'Invalid alpha level: {alpha}. Can be from 0 to 1.')

    if alternative not in C.ALTERNATIVES:
        raise ValueError(f'Invalid alternative: {alternative}. Can be "two-sided", "greater", or "less".')

    vn1 = std1 ** 2 / nobs1
    vn2 = std2 ** 2 / nobs2

    if not relative:
        lift = mean1 - mean2
        std_err = np.sqrt(vn1 + vn2)
    else:
        lift = mean2 / mean1 - 1
        # Estimating standard error using Delta Method.
        std_err = np.sqrt((vn1 + vn2) / mean1**2 + vn1*(mean2 - mean1)**2 / mean1**4 + 2*vn1*(mean2 - mean1) / mean1**3)

    df = (vn1 + vn2)**2 / (vn1**2/(nobs1 - 1) + vn2**2/(nobs2 - 1))
    t_stat = lift / std_err

    pvalue = _calculate_pvalue(t_stat=t_stat, df=df, alternative=alternative)
    ci_lower, ci_upper = _calculate_confidence_interval(
        lift=lift, std_err=std_err, df=df, alpha=alpha, alternative=alternative
    )

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
    """
    Perform a Welch's t-test between two samples.

    Args:
        sample1: The first sample data as a NumPy array.
        sample2: The second sample data as a NumPy array.
        alternative: The alternative hypothesis for the test. Can be 'two-sided',
            'greater', or 'less'. Defaults to 'two-sided'.
        alpha: Significance level for the test. Defaults to 0.05.
        relative: If True, calculate the relative lift between means. Defaults to False.

    Returns:
        TTestResult: An object containing the t-test results including lift, p-value, and confidence interval.

    Raises:
        ValueError: If `alternative` is not one of 'two-sided', 'greater', or 'less'.

    Notes:
        - This function calculates the sample means, standard deviations, and number of observations for the two samples.
        - The t-test is then performed using the `ttest_from_stats` function.

    Example:
        sample1 = np.array([10, 12, 15, 8, 9])
        sample2 = np.array([18, 20, 22, 16, 19])
        result = ttest(sample1=sample1, sample2=sample2)
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f'Invalid alpha level: {alpha}. Can be from 0 to 1.')

    if alternative not in C.ALTERNATIVES:
        raise ValueError(f'Invalid alternative: {alternative}. Can be "two-sided", "greater", or "less".')

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
