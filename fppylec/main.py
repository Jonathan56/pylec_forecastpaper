import pandas
from datetime import datetime, timedelta
from fppylec import metric, forecast, optimization, validate

def main(df, start, end, pv_capacity, battery_kWh, battery_kW,
         f_method, f_kwarg, f_horizon, f_train_period, control_func):
    """
    Core function to run forecast -> optimization.
    Notes:
        f_horizon=timedelta(days=2) - timedelta(minutes=15)
        f_train_period=timedelta(days=14)
    """

    # Result frame (add training data or "real data")
    start_timer = datetime.now()
    one_day = timedelta(hours=23, minutes=45)
    result = df.loc[start-f_train_period:start-timedelta(minutes=15)].copy()
    result['r_houses_kW'] = result['vo_houses_kW']
    result['r_pv_kW'] = result['vo_pv_coef'] * pv_capacity
    result['r_battery_kW'] = [0] * len(result)

    # Main loop optimize once a day
    days = (df.loc[start:end].groupby(pandas.Grouper(freq='D')).sum().index)
    SOC_end = [battery_kWh / 2]
    for day in days:
        # Retrieve historical data, and prepare future results
        training = result.loc[day-f_train_period:day].copy()
        dfalgo = df.loc[day:day+f_horizon].copy()

        # DFALGO
        # Forecasts (PV and consumption)
        dfalgo['f_pv_kW'] = dfalgo['vo_pv_coef'] * pv_capacity
        if f_method == forecast.perfect:
            dfalgo['f_houses_kW'] = dfalgo['vo_houses_kW']
        else:
            dfalgo['f_houses_kW'] = f_method(
                training['r_houses_kW'], f_horizon, **f_kwarg)

        # Control signal
        dfalgo['f_battery_kW'], SOC = control_func(
                        dfalgo['f_houses_kW'], dfalgo['f_pv_kW'],
                        extra={'battery_kWh': battery_kWh,
                               'battery_kW':  battery_kW,
                               'initial_kwh': SOC_end, 'eta': 0.95})
        emin, emax = validate.battery(
            dfalgo, (0, battery_kWh), (-battery_kW, battery_kW), SOC_end[0])
        SOC_end = SOC['SOC_end']

        # DFDAY
        # Select results for only one day
        dfday = df.loc[day:day+one_day].copy()
        dfday['f_houses_kW'] = dfalgo.loc[day:day+one_day, 'f_houses_kW'].copy()
        dfday['f_pv_kW'] = dfalgo.loc[day:day+one_day, 'f_pv_kW'].copy()
        dfday['f_battery_kW'] = dfalgo.loc[day:day+one_day, 'f_battery_kW'].copy()

        # Insert some impact of the coordination in the overall metered consump.
        dfday['r_battery_kW'] = dfday['f_battery_kW']  # Perfect forecast
        dfday['r_houses_kW'] = dfday['vo_houses_kW']  # Real = historic values
        dfday['r_pv_kW'] = dfday['f_pv_kW']  # Perfect forecast

        # Save for the next iteration
        result = pandas.concat([result, dfday], axis=0, sort=True)

    # Remove training from the results ?
    result = result.loc[
        start:end, ['vo_houses_kW', 'vo_pv_coef',
                    'f_houses_kW', 'f_pv_kW', 'f_battery_kW',
                    'r_houses_kW', 'r_pv_kW', 'r_battery_kW']]
    time_elapsed = datetime.now() - start_timer
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    print('')

    # Quality
    metrics = {}
    metrics['MAPE_%'] = metric.mape(
        result, 'vo_houses_kW', 'f_houses_kW', threshold=0.1)
    metrics['MAPE_9a8p_%'] = metric.mape_hod(
        result, 'vo_houses_kW', 'f_houses_kW', threshold=0.1,
        start='9:00', end='20:00')
    metrics['MAE_kW'] = metric.mae(result, 'vo_houses_kW', 'f_houses_kW')
    metrics['MASE'] = metric.mase(result, 'vo_houses_kW', 'f_houses_kW')
    metrics['ME_kW'] = metric.me(result, 'vo_houses_kW', 'f_houses_kW')

    # Value
    r = metric.value_metrics(result, 'r_houses_kW', 'r_pv_kW', 'r_battery_kW')
    metrics['scons_%'] = r['scons_%']
    metrics['ssuff_%'] = r['ssuff_%']
    metrics['scons_%_nobatt'] = r['scons_%_nobatt']
    metrics['ssuff_%_nobatt'] = r['ssuff_%_nobatt']
    return metrics
