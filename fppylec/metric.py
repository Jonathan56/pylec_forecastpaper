"""
Metric functions are kept simple (no data manipulation), when possible metrics use existing function from scikit-learn or scipy.

We choose to implement this module such that time series are passed as a Dataframe, and relevant columns are specified. This way we have a uniform data format (no lists or arrays), the Jupyter side is specific enough without adding too much verbose. The function are kept from accessing other variables, and have similar signatures.
"""
import pandas
import math
import cmath
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pytest

"""
Quality metrics
"""

def mae(df, true, pred):
    """$\text{MAE} = \frac{1}{T} \sum^T_{t=1} |e_t|$
    Unit: [kW]
    """
    mae = mean_absolute_error(df[true].values, df[pred].values)
    return np.round(mae, 2)

def rmse(df, true, pred):
    """$\text{RMSE} = \sqrt{\frac{1}{T} \sum^T_{t=1} e_t^2}$
    """
    mse = mean_squared_error(df[true].values, df[pred].values)
    rmse = math.sqrt(mse)
    return np.round(rmse, 2)

def mape(df, true, pred, threshold=0.1):
    """{MAPE} = frac{1}{T} sum^T_{t=1}left(frac{|e_t|}{y_{t}} times 100right)
    Unit: [%]
    """
    # Remove values below the threshold
    mask = df[true].abs() >= threshold
    pe = ((df[true][mask].values - df[pred][mask].values) * 100
         / df[true][mask].values)
    mape = np.mean(np.abs(pe))
    return np.round(mape, 2)

def mape_hod(df, true, pred, threshold=0.1, start='7:00', end='20:00'):
    """MAPE between start and end time of the day. Start and end included, for details see DataFrame.between_time
    Unit: [%]
    """
    # Remove values below the threshold
    mask = df[true].abs() >= threshold

    # Remove value outside of the time window
    hod = df[mask].between_time(
        start_time=start, end_time=end).copy()

    pe = ((hod[true].values - hod[pred].values) * 100
           / hod[true].values)
    mape = np.mean(np.abs(pe))
    return np.round(mape, 2)

def mase(df, true, pred, only_mase=True):
    """$\text{MASE} = \frac{1}{T} \sum^T_{t=1} \left(\frac{|e_t|}{\frac{1}{n-1} \sum^n_{i=2} |y_i - y_{i-1}|}\right)$

    See "Another look at measures of forecast accuracy", Rob J Hyndman

    The original implementation of mase() calls for using the in-sample naive mean absolute error to compute scaled errors with. It uses this instead of the out-of-sample error because there is a chance that the out-of-sample error cannot be computed when forecasting a very short horizon (i.e. the out of sample size is only 1 or 2). However, yardstick only knows about the out-of-sample truth and estimate values. Because of this, the out-of-sample error is used in the computation by default.
    Unit: [unitless]
    """
    ae = np.abs(df[true].values - df[pred].values)
    mad = np.mean(np.abs(np.diff(df[true].values)))
    mase = np.mean(ae / mad)
    if only_mase:
        return np.round(mase, 2)
    else:
        return {'mae': np.round(np.mean(ae), 2),
                'mad': np.round(mad, 2),
                'mase': np.round(mase, 2)}

def me(df, true, pred):
    """$text{ME} = frac{1}{T} sum^T_{t=1} e_t$
    Unit: [kW]
    """
    me = np.mean(df[true] - df[pred])
    return np.round(me, 2)

"""
Value metrics
"""

def value_metrics(df, cons, prod, storage):
    """Return self-consumption and sufficiency with and without battery.
    """
    assert int(pandas.infer_freq(df.index)[:-1]) == 15

    # With batteries
    _df = allocation.merge_storage(df, cons, prod, storage)
    metrics = self_consumption_production(_df, cons, prod)

    # Without batteries
    _ = self_consumption_production(df, cons, prod)
    metrics['scons_%_nobatt'] = _['scons_%']
    metrics['ssuff_%_nobatt'] = _['ssuff_%']
    return metrics

def self_consumption_production(df, cons, prod):
    """Return self-consumption and self-production in %
    """
    _result = {}
    local = df[[cons, prod]].min(axis=1).sum()  # unit do not matter
    _result[f'scons_%'] = local * 100 / df[prod].sum()
    _result[f'ssuff_%'] = local * 100 / df[cons].sum()
    return _result

def merge_storage(df, cons, prod, stor):
    """Merge positve storage in consumption and negative part in the production
    """
    _df = df.copy()
    assert not _df.isnull().values.any(), 'Include NaN values'
    _df[prod] = _df[prod] - _df[stor].clip(upper=0)
    _df[cons] = _df[cons] + _df[stor].clip(lower=0)

    # Assert that total energy remains the same
    assert ((df[cons] - df[prod] + df[stor]).sum() ==
            pytest.approx((_df[cons] - _df[prod]).sum(), abs=1e-6))
    assert any(_df[cons] >= 0)
    assert any(_df[prod] >= 0)
    _df.drop(stor, axis=1, inplace=True)  # not needed
    return _df
