from datetime import datetime, timedelta
import pandas
import numpy as np
import warnings

from pyts.approximation import DiscreteFourierTransform  # ARIMA
#from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima, ARIMA  # ARIMA
import fbprophet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import keras
import tensorflow as tf

def perfect(training, horizon):
    return 'Not implemented'

def ma(training, horizon):
    """Moving average forecast"""
    # Start of the forecast 15 min after last training data point
    f_start = training.index[-1] + timedelta(minutes=15)
    f_end = f_start + horizon

    # Extend training horizon to include forecasting timestamps
    ix = pandas.date_range(
        start=training.index[0], end=f_end, freq='15T')
    tmp = training.loc[:].reindex(ix)

    # Moving average
    tmp = tmp.loc[:].rolling(window='7D', closed='left').mean()

    # Return just the forecasting horizon
    return tmp.loc[f_start:f_end]


def snaive(training, horizon):
    """Moving average forecast"""
    # Start of the forecast 15 min after last training data point
    f_start = training.index[-1] + timedelta(minutes=15)
    f_end = f_start + horizon

    # Extend training horizon to include forecasting timestamps
    ix = pandas.date_range(
        start=training.index[0], end=f_end, freq='15T')
    tmp = training.loc[:].reindex(ix)

    # Shift data by 7 days
    tmp = tmp.loc[:].shift(1, freq=timedelta(days=7))
    assert (tmp.loc[f_start:f_end].sum() ==
            training.loc[f_start-timedelta(days=7):
                         f_end-timedelta(days=7)].sum())

    # Return just the forecasting horizon
    return tmp.loc[f_start:f_end]

def holtwinters(training, horizon):
    """Holt-winters"""
    # Start of the forecast 15 min after last training data point
    f_start = training.index[-1] + timedelta(minutes=15)
    f_end = f_start + horizon

    # Fit model
    day_len = 24*4
    training_len = len(training)
    training.index = pandas.DatetimeIndex(  # Just add info freq = 15T
        training.index.values, freq='15T')

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = ExponentialSmoothing(
            training, trend=None, seasonal='add', seasonal_periods=day_len)
        model_fit = model.fit(
            optimized=True, use_boxcox=False, remove_bias=False)

    # Predict and save results
    yhat = model_fit.predict(
        start=f_start,
        end=f_end)
    assert (len(yhat) == len(pandas.date_range(
            start=f_start, end=f_end, freq='15T')))
    return yhat


def prophet(training, horizon, exog=False):
    """Facebook algorithm for timeseries"""
    # Make sure we are just forecasting 2 days ahead (hardcoded below)
    assert horizon == (timedelta(days=2) - timedelta(minutes=15))

    # Prepare DataFrame for prophet
    days = 2
    day_len = 24*4
    df = pandas.DataFrame(index=range(0, len(training)),
                          data={'ds': training.index,
                                'y': training.values})

    # Create model
    model = fbprophet.Prophet(weekly_seasonality=True,
                          seasonality_mode='multiplicative',
                          seasonality_prior_scale=1)
    if exog:  # Add regressors
        df['hour'] = df['ds'].apply(lambda x: x.hour)
        df['weekday'] = df['ds'].apply(lambda x: x.weekday())
        model.add_regressor('hour', mode='additive')
        model.add_regressor('weekday', mode='additive')

    # Fit model
    model.fit(df)

    # Predict
    future = model.make_future_dataframe(
        periods=days*day_len, freq='15T', include_history=False)
    if exog:
        future['hour'] = future['ds'].apply(lambda x: x.hour)
        future['weekday'] = future['ds'].apply(lambda x: x.weekday())
    forecast = model.predict(future)

    # Check
    assert len(forecast.yhat.values) == days * day_len
    f_start = training.index[-1] + timedelta(minutes=15)
    f_end = f_start + horizon
    ix = pandas.date_range(start=f_start, end=f_end, freq='15T')
    r = pandas.DataFrame(index=ix, data={'yhat': forecast.yhat.values})
    return r.loc[f_start:f_end, 'yhat']


def sklearn_models(training, horizon, model=False):
    """Multi-purpose forecasting func, see available models"""
    # Forecasting horizon (not used here)
    f_start = training.index[-1] + timedelta(minutes=15)
    f_end = f_start + horizon

    # Format for sklearn
    formatted = pandas.DataFrame()
    days = (training.groupby(pandas.Grouper(freq='D')).sum().index)
    assert len(days) == 14, 'We need 2 weeks of training data'
    assert len(days[::2]) == 7, 'We need 2 weeks of training data'
    assert days[-1] < f_start, 'Forecasting start after training'

    # Turn days of the week into a binary vector
    categories = np.array(range(0, 7))
    categories = categories.reshape(-1, 1)
    encoding = OneHotEncoder(
        categories="auto").fit_transform(categories).toarray()
    encoding_name1 = ['1-Mon', '1-Tue', '1-Wed',
                      '1-Thu', '1-Fri', '1-Sat', '1-Sun']
    encoding_name2 = ['2-Mon', '2-Tue', '2-Wed',
                      '2-Thu', '2-Fri', '2-Sat', '2-Sun']

    # Split the data per 2 days
    for day, index in zip(days[::2], list(range(1, len(days)+1)[::2])):
        d_start = day
        d_end = day + horizon
        df_day = pandas.DataFrame(training.loc[d_start:d_end]).T
        df_day.columns = [s.strftime('%H:%M:%S') for s in df_day.columns]
        df_day.index = ['day ' + str(index) + ' and ' + 'day ' + str(index+1)]

        # Encoding for day 1 and day 2 (loc[:, []] wasn't working --> for loop)
        for j, code_name in enumerate(encoding_name1):
            df_day.loc[:, code_name] = encoding[d_start.weekday()][j]
        for j, code_name in enumerate(encoding_name2):
            df_day.loc[:, code_name] = encoding[d_end.weekday()][j]

        # Concat
        formatted = pandas.concat([formatted, df_day], axis=0)

    # Build Y = f(X) where X is the input and Y the target (no need for 1-Mon...)
    two_days = 4 * 24 * 2
    x = formatted.iloc[:-1, :].to_numpy()
    y = formatted.iloc[1:, :two_days].to_numpy()
    assert x.shape[0] == y.shape[0]

    # Split training and test: 10 days, and 2 days
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=False)
    assert len(x_test) == 1, 'Testing on the last tuple of 2 days'

    # Available models
    models = {
        'median': DummyRegressor("median"),
        'lasso': Lasso(alpha=0.01),
        'linear_regression': LinearRegression(),
        'kneighbors': KNeighborsRegressor(n_neighbors=3),  # Calibration ?
        'decisiontree': DecisionTreeRegressor(criterion="mse", max_depth=2),
    }
    model = models[model]

    # Fit and predict
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(model, Lasso):
            model = GridSearchCV(Lasso(), [{"alpha": [0.01, 0.1, 1.0]}],
                                 cv=2, scoring='neg_mean_absolute_error')
        model.fit(x_train, y_train)
        f = model.predict(x_test)
    assert len(f[0]) == two_days, 'forecast for 2 days with 15-min interval'

    # Return result with the right time range (potentially less)
    # (e.g. not 192 but 96 for the last day of the year)
    ix = pandas.date_range(start=f_start, end=f_end, freq='15T')
    r = pandas.DataFrame(index=ix, data={'yhat': f[0]})
    return r.loc[f_start:f_end, 'yhat']


def keras_models(training, horizon, model=None):
    """Multi-purpose forecasting func, see available models"""
    # Forecasting horizon (not used here)
    f_start = training.index[-1] + timedelta(minutes=15)
    f_end = f_start + horizon

    # Format for sklearn
    formatted = pandas.DataFrame()
    days = (training.groupby(pandas.Grouper(freq='D')).sum().index)
    assert len(days) == 14, 'We need 2 weeks of training data'
    assert len(days[::2]) == 7, 'We need 2 weeks of training data'
    assert days[-1] < f_start, 'Forecasting start after training'

    ## Turn days of the week into a binary vector
    #categories = np.array(range(0, 7))
    #categories = categories.reshape(-1, 1)
    #encoding = OneHotEncoder(categories="auto").fit_transform(categories).toarray()
    #encoding_name1 = ['1-Mon', '1-Tue', '1-Wed', '1-Thu', '1-Fri', '1-Sat', '1-Sun']
    #encoding_name2 = ['2-Mon', '2-Tue', '2-Wed', '2-Thu', '2-Fri', '2-Sat', '2-Sun']

    # Split the data per 2 days
    for day, index in zip(days[::2], list(range(1, len(days)+1)[::2])):
        d_start = day
        d_end = day + horizon
        df_day = pandas.DataFrame(training.loc[d_start:d_end]).T
        df_day.columns = [s.strftime('%H:%M:%S') for s in df_day.columns]
        df_day.index = ['day ' + str(index) + ' and ' + 'day ' + str(index+1)]

        ## Encoding for day 1 and day 2 (loc[:, []] wasn't working --> for loop)
        #for j, code_name in enumerate(encoding_name1):
        #    df_day.loc[:, code_name] = encoding[d_start.weekday()][j]
        #for j, code_name in enumerate(encoding_name2):
        #    df_day.loc[:, code_name] = encoding[d_end.weekday()][j]

        # Concat
        formatted = pandas.concat([formatted, df_day], axis=0)

    # Build Y = f(X) where X is the input and Y the target (no need for 1-Mon...)
    two_days = 4 * 24 * 2
    x = formatted.iloc[:-1, :].to_numpy()
    y = formatted.iloc[1:, :two_days].to_numpy()
    assert x.shape[0] == y.shape[0]

    # Split training and test: 10 days, and 2 days
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=False)
    assert len(x_test) == 1, 'Testing on the last tuple of 2 days'

    # ANN
    # LSTM Neural Network
    if model == "lstm":
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        #y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
        #y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

        model = keras.Sequential([
            keras.layers.LSTM(192, input_shape=(192, 1), return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(192),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(192)
        ])

    # Fit and predict
    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
    optimizer = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=["mean_absolute_error"])
    model.fit(x_train , y_train, epochs=50, verbose=0)
    f = model.predict(x_test)
    assert len(f[0]) == two_days, 'forecast for 2 days with 15-min interval'

    # Return result with the right time range (potentially less)
    # (e.g. not 192 but 96 for the last day of the year)
    ix = pandas.date_range(start=f_start, end=f_end, freq='15T')
    r = pandas.DataFrame(index=ix, data={'yhat': f[0]})
    return r.loc[f_start:f_end, 'yhat']
