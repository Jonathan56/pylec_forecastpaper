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
    """$\text{MAPE} = \frac{1}{T} \sum^T_{t=1}\left(\frac{|e_t|}{y_{t}} \times 100\right)$
    Unit: [%]
    """
    # Check for small values (removed because dealt with after)
    # assert np.any(np.abs(df[true].values) < 10e-3), 'Warning small values'

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
    # Check for small values (removed because dealt with after)
    # assert np.any(np.abs(df[true].values) < 10e-3), 'Warning small values'

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
    """$\text{MAE} = \frac{1}{T} \sum^T_{t=1} e_t$
    Unit: [kW]
    """
    e = (df[true] - df[pred])
    me = np.mean(e)
    return np.round(me, 2)

"""
Value metrics
"""

def netconsumption(df, consumption, production):
    """Net consumption is equivalent to local consumption for infinite horizon. In other word, time of consumption does not have to match time of production. Net-consumption is either the production, or the consumption if production exceed consumption.

    $\text{net-consumption} = \min{\sum\text{consumption}, \sum\text{production}}$

    Example:
    r = metric.netconsumption(result['perfect'], 'r_houses_kW', 'r_pv_kW')
    max_selfcons_kWh = r['netconsumption']
    max_selfcons = r['netconsumption_selfcons']
    max_selfsuff = r['netconsumption_selfprod']
    """
    assert not df.isnull().values.any()
    df_freq = int(pandas.infer_freq(df.index)[:-1])

    # netconsumption in kWh
    netconsumption = min(df[consumption].sum(), df[production].sum())
    netconsumption = netconsumption * df_freq / 60

    # netconsumption in % of production
    netconsumption_selfcons = (netconsumption * 100 /
                               (df[production].sum() * df_freq / 60))
    assert netconsumption_selfcons <= 100.0001, f'value is {netconsumption_selfcons}'

    # netconsumption in % of consumption
    netconsumption_selfprod = (netconsumption * 100 /
                               (df[consumption].sum() * df_freq / 60))
    assert netconsumption_selfprod <= 100, f'value is {netconsumption_selfcons}'
    return {'netconsumption': np.round(netconsumption, 2),
            'netconsumption_selfcons': np.round(netconsumption_selfcons, 2),
            'netconsumption_selfprod': np.round(netconsumption_selfprod, 2)}

def _merge_storage(df, consumption, storage, production):
    """
    Helper function

    Note: a consequence of consumption $> 0$ is that any export to the grid from the storage unit is not included in self-consumption. However it is included in self-production (cf French law consider battery injection as production). Plus it penalized large self-consumption based on batteries discharging on the grid.
    """
    if storage:
        ad = df[[consumption, storage, production]].copy()

        # Compute production (include grid injection from the battery)
        ad[production] = (ad[production] -
                         (ad[consumption] + ad[storage]).clip(upper=0))

        # Compute consumption
        ad[consumption] = (ad[consumption] + ad[storage]).clip(lower=0)

        # Assert that total energy remains the same
        assert (pytest.approx(df[[consumption, storage, production]].sum().sum(), rel=0.1) ==
                ad[[consumption, production]].sum().sum())
    else:
        ad = df[[consumption, production]].copy()
    return ad

def localconsumption(df, consumption, storage, production, horizon,
                     timeseries=False, warning=True):
    """
    \begin{subequations}
    \begin{gather}
    \text{self-consumption} = \min{\text{consumption}, \text{production}} \\
    \text{consumption} = \max{0, \text{consumption}} \\  # consumption > 0
    \text{production} = \max{0, \text{production}} \\ # production > 0
    \text{consumption} = \text{consumption} + \text{storage}
    \text{production} = \text{production} - \min{0, \text{consumption}}
    \end{gather}
    \end{subequations}

    Note: horizon is in minutes (int) for the unit to be kWh

    Note: if timeseries=True the result of this function is a local_consumption_kW with unit kW and freq horizon (see assert warning).

    Result:
    - local_consumption_kWh
    - self_consumption_% $\text{self-consumption} = \frac{\text{local-consumption}}{\text{production}}$
    - self_sufficiency_% $\text{self-sufficiency} = \frac{\text{local-consumption}}{\text{consumption}}$
    """
    # Distribute the impact of storage to production and consumption
    ad = _merge_storage(df, consumption, storage, production)

    # Run classic checks
    assert not ad.isnull().values.any(), 'Include NaN values'
    df_freq = int(pandas.infer_freq(ad.index)[:-1])
    if warning:
        assert df_freq == horizon, 'Warning check that it makes sense here'

    # Local consumption with horizon
    assert not any(ad[consumption] < 0)
    assert not any(ad[production] < 0)
    freq = str(horizon) + 'T'
    local_consumption_kW = (
        ad.groupby(pandas.Grouper(freq=freq)).sum()
        [[consumption, production]].min(axis=1) * df_freq / horizon)
    local_consumption_kWh = local_consumption_kW.sum() * horizon / 60

    # Make sure that energy consumption or energy production is at least more
    assert ad[consumption].sum(axis=0) * df_freq / 60 >= local_consumption_kWh
    assert ad[production].sum(axis=0) * df_freq / 60 >= local_consumption_kWh

    # Calculate self-consumption
    selfconsumption = (local_consumption_kWh * 100 /
                      (ad[production].sum(axis=0) * df_freq / 60))

    # Calculate self-production
    selfsufficiency = (local_consumption_kWh * 100 /
                      (ad[consumption].sum(axis=0) * df_freq / 60))

    if timeseries:
        return local_consumption_kW
    return {'local_consumption_kWh': np.round(local_consumption_kWh, 2),
            'self_consumption_%': np.round(selfconsumption, 2),
            'self_sufficiency_%': np.round(selfsufficiency, 2)}

def operationcost(df, consumption, storage, production, horizon,
                  buying_price=0.1515, selling_price=0.1, network_fees=0.07, timeseries=False, warning=True):
    """
    \text{Cost} = \beta \sum^T_{t=1}(d_t - g_t)^+ - \gamma \sum^T_{t=1}(g_t - d_t)^+ + \theta \sum^T_{t=1} (l_t + p^b_{t, out not injected})

    Note on prices:
    https://selectra.info/energie/fournisseurs/edf/tarifs-reglementes
    https://www.fournisseurs-electricite.com/guides/technique/horaires-heures-creuses
    onpeak_price = 0.1710 # kWh/euro
    offpeak_price = 0.1320 # kWh/euro 23h — 7h
    buying_price = 0.1515 # kWh/euro (average of on/off peak)
    selling_price = 0.1  # kWh/euro
    """
    # Distribute the impact of storage to production and consumption
    od = _merge_storage(df, consumption, storage, production)

    # Netload
    df_freq = int(pandas.infer_freq(df.index)[:-1])
    od['netload'] = od[consumption] - od[production]
    assert od['netload'].sum() <= od[consumption].sum()

    # Buying Energy
    buying_cost = (od[od['netload'] > 0]['netload'].sum()
                   * df_freq / 60 * buying_price)
    assert (buying_cost <=  # Buying netload less than buying total cons
            df[consumption].sum() * df_freq / 60 * buying_price)

    # Selling Energy
    selling_gain = - (od[od['netload'] < 0]['netload'].sum()
                   * df_freq / 60 * selling_price)
    assert (selling_gain <=  # Selling netload less than selling all prod
            df[production].sum() * df_freq / 60 * selling_price)

    # Network fees on solar consumption (passive and charging battery)
    temp = localconsumption(df, consumption, storage, production, horizon)
    local_consumption_fees = temp['local_consumption_kWh'] * network_fees
    assert (local_consumption_fees <=  # Cost of consumption at least more
            df[consumption].sum() * df_freq / 60 * buying_price)

    # Network fees on battery discharge, but not on grid injection
    battery_disch_cost = ((- df[df[storage] < 0][storage].sum() +
                         (df[consumption] + df[storage]).clip(upper=0).sum())
                         * df_freq / 60 * network_fees)
    assert (battery_disch_cost <=
            - df[df[storage] < 0][storage].sum() * df_freq / 60 * network_fees)

    # Total operation Cost
    operationcost = (buying_cost - selling_gain + local_consumption_fees
                     + battery_disch_cost)
    assert operationcost <= df[consumption].sum() * df_freq / 60 * buying_price


    return {'buying_cost': np.round(buying_cost, 2),
            'selling_gain': np.round(selling_gain, 2),
            'local_consumption_fees': np.round(local_consumption_fees, 2),
            'battery_disch_cost': np.round(battery_disch_cost, 2),
            'operationcost': np.round(operationcost, 2)}

def fairness(df, consumption, storage, production, horizon,
             individual_consumptions, sharing_keys):
    """
    """
    # Compute standard deviation

    # Get fairness
    return None
