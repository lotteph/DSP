import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import seaborn as sns
import os
import datetime as dt
import xgboost as xgb
import catboost
from sklearn.svm import OneClassSVM
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import radians, cos, sin, asin, sqrt
from dateutil.relativedelta import relativedelta
from datetime import timedelta

# Global variables for the park locations
vondelpark_west = [{'lat': 52.356496, 'lng': 4.861447}]
vondelpark_oost_3 = [{'lng': 4.869217, 'lat': 52.358252}]
vondelpark_oost_2 = [{'lng': 4.874692, 'lat': 52.359798}]
vondelpark_oost_1 = [{'lng': 4.879652, 'lat': 52.360991}]
oosterpark = [{'lng': 4.920558, 'lat': 52.360098}]
sarphatipark = [{'lng': 4.896375, 'lat': 52.354364}]
westerpark_west = [{'lng': 4.867128, 'lat': 52.387099}]
westerpark_centrum = [{'lng': 4.873268, 'lat': 52.387374}]
westerpark_oost = [{'lng': 4.878379, 'lat': 52.386379}]
westergasfabriek = [{'lng': 4.869769, 'lat': 52.385920}]
rembrandtpark_noord = [{'lng': 4.846573, 'lat': 52.366664}]
rembrandtpark_zuid = [{'lng': 4.846932, 'lat': 52.361161}]
erasmuspark = [{'lng': 4.851909, 'lat': 52.374808}]
amstelpark = [{'lng': 4.894404, 'lat': 52.330409}]
park_frankendael = [{'lng': 4.929839, 'lat': 52.350703}]
beatrixpark = [{'lng': 4.881352, 'lat': 52.342471}]
flevopark = [{'lng': 4.947881, 'lat': 52.360087}]
gaasperpark = [{'lng': 4.992192, 'lat': 52.310420}]
nelson_mandelapark = [{'lng': 4.963691, 'lat': 52.312204}]
noorderpark = [{'lng': 4.919606, 'lat': 52.392651}]
sloterpark = [{'lng': 4.811894, 'lat': 52.366219}]
wh_vliegenbos = [{'lng': 4.931495, 'lat': 52.388802}]

def predict(model, data, location, catboost=True):
    '''
    Predict the amount of visits using Catboost or XGBoost

    :model: The Catboost/GXBoost model used for predicting
    :data: Dataframe with the data to make the predictions for
    :location: The location of the park to make predictions for
    :catboost: True when predicting with Catboost, false when using XGBoost

    Returns a Dataframe with the predictions
    '''
    # Select data for a specific park
    data = data[data['Location'] == location]

    # Split the data into input and output variables
    X = data[['Location', 'Date', 'Time',
        'Journeys', 'Windspeed', 'Temperature', 'Clouds', 'Rain amount',
        'Rain duration', 'Sun duration', 'Fog', 'Rain', 'Snow', 'Thunder',
        'Ice', 'Holiday_Count', 'Year', 'Month', 'Day_x',
        'retail_and_recreation_percent_change_from_baseline',
        'grocery_and_pharmacy_percent_change_from_baseline',
        'parks_percent_change_from_baseline',
        'transit_stations_percent_change_from_baseline',
        'workplaces_percent_change_from_baseline',
        'residential_percent_change_from_baseline', 'stringency_index',
        'Holiday_Name_Ascension Thursday',
        'Holiday_Name_Easter', 'Holiday_Name_Fall holiday',
        'Holiday_Name_Good Friday', 'Holiday_Name_Kings day',
        'Holiday_Name_Liberation Day', 'Holiday_Name_May holiday',
        'Holiday_Name_Spring holiday', 'Holiday_Name_Summer holiday',
        'Holiday_Name_Whit', 'Holiday_Name_Winter holiday']]
    y = data['Visits']

    if catboost == True:
        pred = model.predict(X)
    else:
        test_dmatrix = xgb.DMatrix(data = X, label = y)
        pred = model.predict(test_dmatrix)

    # Put the predictions in a dataframe and sort them by time
    predictions = pd.DataFrame({'Predicted visitors': pred,
                                'Actual visitors': y}).sort_index()

    return predictions

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def ceil_dt(dt, delta):
    return dt + (dt.min - dt) % delta

def getParkSuggestion(df, date, lat, lng, pred, time):
    resono = df.reset_index()
    resono['Datetime'] = pd.to_datetime(resono['Datetime'])
    resono['Location'] = resono['Location'].str.replace('W.H. Vliegenbos', 'WH Vliegenbos')

    prediction_date_4_months = date + relativedelta(months=-4, days=-1)
    df_4months = resono[(resono['Datetime'] >= prediction_date_4_months) & (resono['Datetime'] < date)]

    df_4months['Weekday'] = df_4months['Datetime'].apply(lambda x: x.weekday())
    df_4months_groupby = df_4months.groupby(['Time', 'Weekday', 'Location']).mean()

    df_4months_groupby = df_4months_groupby.reset_index()
    df_baseline = df_4months_groupby[(df_4months_groupby['Weekday'] == date.weekday()) &
                                  (df_4months_groupby['Time'] == time.strftime("%H:%M:%S"))]

    df_baseline['Longitude'] = df_baseline['Location'].apply(lambda x: globals()["_".join(f"{x.lower()}".split())][0]['lng'])
    df_baseline['Latitude'] = df_baseline['Location'].apply(lambda x: globals()["_".join(f"{x.lower()}".split())][0]['lat'])

    df_baseline['Distance'] = [haversine(lat, lng, df_baseline.iloc[x]['Latitude'], df_baseline.iloc[x]['Longitude'])
                         for x in range(df_baseline.shape[0])]

    df_baseline['Predictions'] = pred
    df_baseline['Crowdedness factor'] = (df_baseline['Predictions'] - df_baseline['Visits']) / df_baseline['Visits'] #(baseline - values) / values
    crowdedness = df_baseline['Crowdedness factor'].values
    df_baseline['Crowdedness factor'] = [x/5 if x <= 0 else x for x in crowdedness]
    df_baseline['Park suggestion'] = df_baseline['Distance'] + (df_baseline['Crowdedness factor']*5)
    df_baseline['Distance'] = round(df_baseline['Distance'],2)

    df_baseline = df_baseline.reset_index(drop=True)
    df_baseline.index += 1
    df_baseline = df_baseline[df_baseline['Distance'] > 0]

    return df_baseline.sort_values(by='Park suggestion')[['Location', 'Distance', 'Park suggestion', 'Time', 'Visits', 'Predictions']].iloc[:3]

def get_baseline(df, date, time):
    '''
    Get the baseline of the last four months for the amount of visitors for a certain date and time.

    :df: A Dataframe with the Resono data
    :date: The date for which the baseline is calculated
    :time: The time for which the baseline is calculated

    Returns a dataframe
    '''
    resono = df.reset_index()
    resono['Datetime'] = pd.to_datetime(resono['Datetime'])
    resono['Location'] = resono['Location'].str.replace('W.H. Vliegenbos', 'WH Vliegenbos')

    prediction_date_4_months = date + relativedelta(months=-4, days=-1)
    df_4months = resono[(resono['Datetime'] >= prediction_date_4_months) & (resono['Datetime'] < date)]

    df_4months['Weekday'] = df_4months['Datetime'].apply(lambda x: x.weekday())
    df_4months_groupby = df_4months.groupby(['Time', 'Weekday', 'Location']).mean()

    df_4months_groupby = df_4months_groupby.reset_index()
    df_baseline = df_4months_groupby[(df_4months_groupby['Weekday'] == date.weekday()) &
                                  (df_4months_groupby['Time'] == time.strftime("%H:%M:%S"))]

    return df_baseline
