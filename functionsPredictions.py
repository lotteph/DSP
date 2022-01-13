def preprocessGVB(path_url):
    return path_url

def preprocessWeather(path_url):
    '''
    Reads in and preprocesses the weather data
    
    :path_url: The path_url to the weather data
    
    Returns a preprocessed Dataframe
    '''

    df = pd.read_csv(path_url)
    df.columns = df.columns.str.replace(' ', '')
    df[['FH', 'T', 'RH']] = df[['FH', 'T', 'RH']] / 10
    df['YYYYMMDD'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
    df['date'] = df['YYYYMMDD'] +  pd.to_timedelta(df['HH'], unit='h')
    df.drop(columns = ['#STN', 'DD', 'FF', 'FX', 'T10N', 'TD', 'Q', 
                       'P', 'VV', 'U', 'WW', 'IX', 'HH', 'YYYYMMDD'], inplace=True)
    df.set_index('date', inplace=True)
    return df

def preprocessResono(path_url):
    '''
    Reads in and preprocesses the resono data
    
    :path_url: The path_url to the resono data
    
    Returns a preprocessed Dataframe
    '''
    
    df = pd.read_csv(path_url)
    df = df.drop(columns = ["Unnamed: 0"])
    
    df['End'] = pd.to_datetime(df['End'])
    df['End'] = pd.to_datetime(df['End'].dt.strftime("%Y-%m-%d %H:%M:%S"))
    
    df = df.rename(columns = {'End' : 'Datetime',
                              'End_Dates' : 'Date',
                              'End_Time' : 'Time'})
    df = df.set_index('Datetime')
    df = df.loc['2020-10':]
    
    df = df[df.Location != 'Vondelpark Oost']
    df = df[df.Location != 'Westerpark']

    return df

def preprocessHoliday(path_url):
    holiday = pd.read_csv(path_url)
    holiday = holiday.drop(['Unnamed: 0'], axis = 1)
    holiday = holiday.drop([0,28,120,122, 128, 150,219,221,227],axis=0)
    holiday['Holiday_Name'] = holiday['Holiday_Name'].str.replace('Boxing Day', 'Christmas Day')
    return holiday

def mergeGVBdata(gvb, resono):
    return gvb

def mergeWeatherFiles(df_Weather2020, df_Weather2021):
    '''
    Merges the weather data
    
    :df_Weather2020: Weather data from 2020
    :df_Weather2021: Weather data from 2021
    
    Returns a merged weather Dataframe
    '''
    
    df_weather = pd.concat([df_Weather2020, df_Weather2021], axis=0)
    df_weather = df_weather.loc['2020-10':]

    cols_int = ['SQ', 'DR', 'N', 'M', 'R', 'S', 'O', 'Y']
    cols_float = ['FH', 'T']

    df_weather[cols_float] = df_weather[cols_float].apply(pd.to_numeric, errors='coerce', axis=1)
    df_weather[cols_int] = df_weather[cols_int].apply(pd.to_numeric, errors='coerce', axis=1)
    df_weather['RH'] = df_weather['RH'].apply(lambda x: 0.05 if x==-0.1 else x)
    
    df_weather_resample = pd.concat([df_weather[['FH', 'T', 'N']].resample('15T').interpolate(method='linear'),
                    df_weather[['RH', 'DR', 'SQ', 'M', 'R', 'S', 'O', 'Y']].resample('15T').bfill()],
                   axis=1)
    
    df_weather_resample[['DR', 'SQ']] = df_weather_resample[['DR', 'SQ']] * 1.5
    df_weather_resample['RH'] = df_weather_resample['RH'] / 4
    
    return df_weather_resample 

def mergeWeatherResonoHoliday(df_resono, df_weather, df_holiday):
    '''
    Merges the resono and weather data
    
    :df_resono: All resono data
    :df_weather: All weather data
    :df_holiday: All holiday data
    
    Returns a merged weather Dataframe
    '''
    
    
    merge_resono_weather = pd.merge(df_resono, df_weather, left_index=True, right_index=True, how='left')
    merge_resono_weather = merge_resono_weather.rename({'T': 'Temperature', 'N': 'Clouds', 'FH': 'Windspeed',
                                                    'RH': 'Rain amount', 'DR': 'Rain duration' , 'SQ': 'Sun duration',
                                                    'M': 'Fog', 'R': 'Rain', 'S': 'Snow', 'O': 'Thunder', 'Y': 'Ice'},
                                                   axis=1) 
    
    all_merged = pd.merge(merge_resono_weather, df_holiday, how='left', right_on = 'End_Dates', left_on='Date')
    all_merged = all_merged.drop(['End_Dates'], axis=1)
    return all_merged

def Target_OneHotEncoding(Resono_Holi):
    #fill the blank of Holiday count, year, month, day
    Resono_Holi['Holiday_Count'] = Resono_Holi['Holiday_Count'].replace(np.nan, 0)
    Resono_Holi['Year'] = pd.to_datetime(Resono_Holi['Date']).dt.year
    Resono_Holi['Month'] = pd.to_datetime(Resono_Holi['Date']).dt.month
    Resono_Holi['Day'] = pd.to_datetime(Resono_Holi['Date']).dt.day
    
    Resono_Holi['Holiday_Name'] = Resono_Holi['Holiday_Name'].replace(
                             ['Christmas Day', 'New year', 'Boxing Day', 'Holiday_Name_New year', 'Christmas holiday', 'Holiday_Name_Boxing Day'] ,'Winter holiday')

    Resono_Holi['Holiday_Name'] = Resono_Holi['Holiday_Name'].replace(
                                 ["King's day"] ,'Kings day')

    Resono_Holi['Holiday_Name'] = Resono_Holi['Holiday_Name'].replace(
                                 ['Easter Monday', 'Easter Sunday'] ,'Easter')

    Resono_Holi['Holiday_Name'] = Resono_Holi['Holiday_Name'].replace(
                                 ['Whit Monday', 'Whit Sunday'] ,'Whit')

    '''
    Monday =0, Tuesday=1, Wednesday=2,Thursday =3,  Friday=4 ,  Saturday =5, Sunday =6
    '''

    Resono_Holi['Date'] = Resono_Holi['Date'].astype('datetime64[ns]')

    encoder = ce.TargetEncoder(cols='Holiday_Name')
    Resono_Holi['Holiday_name'] = encoder.fit_transform(Resono_Holi['Holiday_Name'], Resono_Holi['Visits'])
    
    # Holidays
    Resono_Holi_Dummies = pd.get_dummies(Resono_Holi, columns=["Holiday_Name"])

    return Resono_Holi_Dummies

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
    df_detected = df.copy()
    
    for idx, loc in enumerate(df.columns):
        dt = df[loc]
        dt_detected = dt.copy()
        
        scaler = preprocessing.StandardScaler()
        dt_scaled = scaler.fit_transform(dt.values.reshape(-1,1))
            
        fit = model.fit(dt_scaled)
        pred = fit.predict(dt_scaled)
        outlier_index = np.where(pred == -1)
        
        if len(outlier_index) != len(dt_detected):
            dt_detected.iloc[outlier_index] = np.nan
    
        df_detected[loc] = dt_detected
        
    return df_detected#, outlier_index

def interpolate_df(df, backfill=False):
    '''
    Interpolate the NaN values in the dataframe with either backfilling or linear interpolation.
    
    :df: Dataframe to be interpolated
    :backfill: Bool, if true, interpolate with backfilling, otherwise use linear interpolation (default = False)
    
    Returns a Dataframe with interpolated values
    '''
    df_int = df.copy()
    
    if backfill == True:
        df_int = df_int.backfill()
        
    else:
        for idx, loc in enumerate(df.columns):
            dt = df[loc]
            dt_int = dt.copy()
            dt_int = dt_int.interpolate()
            df_int[loc] = dt_int
        
    return df_int

def smooth_df(df, N=3):
    '''
    Smooth the data with a rolling average to remove false peaks in the data
    
    :df: Dataframe to be smoothed
    :N: Size of the moving window (default = 3)
    
    Returns a smoothed Dataframe
    '''
    df_smooth = df.copy()
    df_smooth = df_smooth.rolling(N).mean()
    
    begin_vals = df.iloc[:N-1]
    df_smooth.update(begin_vals)
        
    return df_smooth