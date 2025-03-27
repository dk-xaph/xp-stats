import pytest

from xp_stats.ttest import ttest_from_stats, ttest, TTestResult


def test_ttest():
    sample1 = [1, 2, 3, 4, 5]
    sample2 = [1, 2, 10, 4, 5]

    res = ttest(sample1=sample1, sample2=sample2)
    expected = TTestResult(
        lift=-1.4000000000000004,
        pvalue=0.4492231080571084,
        ci_lower=-5.6916088222823324,
        ci_upper=2.8916088222823317
    )

    assert res == expected