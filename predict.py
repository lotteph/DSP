import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import seaborn as sns
import os
import datetime as dt
import xgboost as xgb
from sklearn.svm import OneClassSVM
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def clean_resono(df, merge=True):
    '''
    ~~Probably defunct once we merge the datasets~~
    Rename the columns and set Datetime as index

    :df: Dataframe to clean
    :merge: True if Noord/Zuid and Oost/West need to be merged (default = True)

    Returns a cleaned Dataframe
    '''
    df['End'] = pd.to_datetime(df['End'])
    df = df.rename(columns = {'End' : 'Datetime',
                              'End_Dates' : 'Date',
                              'End_Time' : 'Time'})
    df = df.set_index('Datetime')
    return df

def add_time_vars(data, onehot=True):
    '''
    Adds columns for the month and weekday, and also the one-hot encoding or the cyclical versions of those features.

    :data: Dataframe that contains the a column with the datetime
    :onehot: Use onehot encoding if true and cyclical features if false (default = True)

    Returns a Dataframe with either the one-hot encoding or the sine and cosine of the month, weekday and time added
    '''
    data = data.reset_index()
    if onehot == True:
        years = pd.Categorical(data['Datetime'].dt.year)
        data['Month'] = pd.Categorical(data['Datetime'].dt.month)
        data['Weekday'] = pd.Categorical(data['Datetime'].dt.weekday)
        data['Hour'] =  pd.Categorical(data['Datetime'].dt.hour)
        data['Minute'] =  pd.Categorical(data['Datetime'].dt.minute)

        year_dummies = pd.get_dummies(years, prefix='Year_')
        month_dummies = pd.get_dummies(data[['Month']], prefix='Month_')
        weekday_dummies = pd.get_dummies(data[['Weekday']], prefix='Weekday_')
        hour_dummies = pd.get_dummies(data[['Hour']], prefix='Hour_')
        minute_dummies = pd.get_dummies(data[['Minute']], prefix='Minute_')

        data = data.merge(year_dummies, left_index = True, right_index = True)
        data = data.merge(month_dummies, left_index = True, right_index = True)
        data = data.merge(weekday_dummies, left_index = True, right_index = True)
        data = data.merge(hour_dummies, left_index = True, right_index = True)
        data = data.merge(minute_dummies, left_index = True, right_index = True)

    else:
        dates = data['Date'].values
        weekdays = []
        months = []
        hours = []
        minutes = []

        for d in dates:
            year, month, day = (int(x) for x in d.split('-'))
            ans = dt.date(year, month, day)
            weekdays.append(ans.isocalendar()[2])
            months.append(month)

        for t in data['Time']:
            hour, minute, second = (int(x) for x in t.split(':'))
            hours.append(hour)
            minutes.append(minute)

        data['Weekday'] = weekdays
        data['Month'] = months
        data['Hour'] = hours
        data['Minute'] = minutes
        data['Weekday_sin'] = np.sin(data['Weekday'] * (2 * np.pi / 7))
        data['Weekday_cos'] = np.cos(data['Weekday'] * (2 * np.pi / 7))
        data['Month_sin'] = np.sin(data['Month'] * (2 * np.pi / 12))
        data['Month_cos'] = np.cos(data['Month'] * (2 * np.pi / 12))
        data['Hour_sin'] = np.sin(data['Hour'] * (2 * np.pi / 24))
        data['Hour_cos'] = np.cos(data['Hour'] * (2 * np.pi / 24))
        data['Minute_sin'] = np.sin(data['Minute'] * (2 * np.pi / 60))
        data['Minute_cos'] = np.cos(data['Minute'] * (2 * np.pi / 60))

    data = data.set_index('Datetime')
    return data

def remove_outliers(df, gamma=0.01, nu=0.03):
    '''
    Remove outliers with a One-Class SVM.

    :df: Dataframe to perform outlier detection on
    :gamma: Value of the kernel coefficient for ‘rbf’ (default = 0.01)
    :nu: Percentage of the data to be classified as outliers (default = 0.03)

    Returns
    :df_detected: Dataframe with the outliers replaced by NaN
    :outlier_index: List of the indexes of the outliers (used for plotting the outliers, probably
                                                         not necessary for final product)
    '''
    model = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
    df = df.reset_index()

    for loc in list(set(df.Location)):
        dt = df[(df.Location == loc)]
        dt_detected = dt.copy()

        scaler = preprocessing.StandardScaler()
        dt_scaled = scaler.fit_transform(dt['Visits'].values.reshape(-1,1))

        fit = model.fit(dt_scaled)
        pred = fit.predict(dt_scaled)
        outlier_index = np.where(pred == -1)
        idx = dt.iloc[outlier_index].index
        df.loc[idx, 'Visits'] = np.nan

    df = df.set_index('Datetime')
    return df

def interpolate_df(df, backfill=False):
    '''
    Interpolate the NaN values in the dataframe with either backfilling or linear interpolation.

    :df: Dataframe to be interpolated
    :backfill: Bool, if true, interpolate with backfilling, otherwise use linear interpolation (default = False)

    Returns a Dataframe with interpolated values
    '''
    df_int = df.copy()
    dt = df['Visits']
    dt_int = dt.copy()

    if backfill == True:
        dt_int = dt_int.backfill()

    else:
        dt_int = dt_int.interpolate()

    df_int['Visits'] = dt_int
    return df_int

def smooth_df(df, N=3):
    '''
    Smooth the data with a rolling average to remove false peaks in the data

    :df: Dataframe to be smoothed
    :N: Size of the moving window (default = 3)

    Returns a smoothed Dataframe
    '''
    df_smooth = df.copy()
    dt = df['Visits']
    df_smooth['Visits'] = dt.rolling(N).mean()

    begin_vals = df.iloc[:N-1]
    df_smooth.update(begin_vals)

    return df_smooth

def get_data(csv):
    '''
    Read csv file and perform data preprocessing on the Dataframe

    :csv: Path to the csv file

    Returns a preprocessed Dataframe
    '''
    resono_df = pd.read_csv(csv, index_col=0)
    data_clean = clean_resono(resono_df)
    no_outlier = remove_outliers(data_clean)
    no_outlier_int = interpolate_df(no_outlier)
    data_smooth = smooth_df(no_outlier_int)
    data_aug = add_time_vars(data_smooth, onehot=True)
    return data_aug

def predict(model, data, location, pred_params):
    '''
    Predict the amount of visits using XGBoost

    :model: The GXBoost model used for predicting
    :data: Dataframe with all the data
    :location: The location of the park to make predictions for
    :pred_params: A list of the names of the predictor variables

    Returns a Dataframe with the predictions
    '''
    # Select data for a specific park
    data = data[data['Location'] == location]

    # Split the data into input and output variables
    X = data[pred_params]
    y = data['Visits']

    # Convert test set to DMatrix objects
    test_dmatrix = xgb.DMatrix(data = X, label = y)

    # Fit the data and make predictions
    pred = model.predict(test_dmatrix)
    predictions = pd.DataFrame({'Predicted visitors': pred,
                                'Actual visitors': y})
    predictions = predictions.clip(lower=0)

    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(y, predictions['Predicted visitors']))
    mae = mean_absolute_error(y, predictions['Predicted visitors'])

    # print(location)
    # print("RMSE : % f" %(rmse))
    # print("MAE : % f" %(mae))
    return predictions.sort_index()
