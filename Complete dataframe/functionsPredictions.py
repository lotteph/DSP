def preprocessGVB(path_url, resono):
    # variabelen/lijsten aanmaken
    # Center points of all the parks (https://www.latlong.net/)
    all_parks = ['vondelpark_west','vondelpark_oost_3','vondelpark_oost_2',
                 'vondelpark_oost_1' 'oosterpark', 'sarphatipark',
                'westerpark_west','westerpark_oost','westerpark_centrum', 
                 'westergasfabriek','rembrandtpark_noord', 'rembrandtpark_zuid', 
                 'erasmuspark']
    
    vondelpark_west_stations = []
    vondelpark_oost_3_stations = []
    vondelpark_oost_2_stations = []
    vondelpark_oost_1_stations = []
    oosterpark_stations = []
    sarphatipark_stations = []
    westerpark_west_stations = []
    westerpark_centrum_stations = []
    westerpark_oost_stations = []
    westergasfabriek_stations = []
    rembrandtpark_noord_stations = []
    rembrandtpark_zuid_stations = []
    erasmuspark_stations = []

    amstelpark_stations = []
    park_frankendael_stations= []
    beatrixpark_stations = []
    flevopark_stations = []
    gaasperpark_stations = []
    nelson_mandelapark_stations = []
    noorderpark_stations = []
    sloterpark_stations = []
    wh_vliegenbos_stations = []
    
    files = glob.glob(path_url)
    gvb_data = pd.concat( (pd.read_csv(file, sep=";") for file in files), ignore_index = True)
    # Set dates to datetime
    # Only select data from 10-2020 till 12-2021
    # Drop if destination is unknown
    # Remove NaN and [[ Onbekend ]] values
    gvb_data['Datum'] = pd.to_datetime(gvb_data['Datum'])
    gvb_data = gvb_data.sort_values(by=['Datum', 'UurgroepOmschrijving (van aankomst)'])
    gvb_data_range = gvb_data[(gvb_data['Datum'] >= '2020-10-1') & (gvb_data['Datum'] <= '2021-12-31')]
    gvb_data_range_cleaned = gvb_data_range[gvb_data_range['AankomstHalteCode'].notnull()]
    gvb_data_range_cleaned = gvb_data_range_cleaned[gvb_data_range_cleaned['AankomstHalteNaam'] != "[[ Onbekend ]]"]

    # Replace missing data with one week before
    gvb_data_range_cleaned_without_9_november = gvb_data_range_cleaned[gvb_data_range_cleaned['Datum'] != "2020-11-09"]
    gvb_week46 = gvb_data_range_cleaned[(gvb_data_range_cleaned['Datum'] >= '2020-11-02') & (gvb_data_range_cleaned['Datum'] <= '2020-11-08')]
    gvb_week46['Datum'] = gvb_week46["Datum"] + dt.timedelta(days=7)
    frames = [gvb_data_range_cleaned_without_9_november, gvb_week46]
    gvb_data_range_very_cleaned = pd.concat(frames)
    gvb_data_range_very_cleaned.sort_values(by="Datum", inplace=True)
    
    # Still a lot of values are missing, make sure every data point gets added and interpolated
    # take last hour from column UurgroepOmschrijving and convert to datetime
    # add one minute to get hour, so 17:00 means 16:00 - 16:59
    # cmobine date and hour to make index unique
    gvb_data_range_travels = gvb_data_range_very_cleaned.copy()

    gvb_data_range_travels['hour'] = gvb_data_range_travels['UurgroepOmschrijving (van aankomst)'].str[:5]
    gvb_data_range_travels['hour'] = pd.to_datetime(gvb_data_range_travels['hour'], format='%H:%M').dt.time
    gvb_data_range_travels['date'] = gvb_data_range_travels.apply(lambda r : pd.datetime.combine(r['Datum'],r['hour']),1)
    gvb_data_range_travels = gvb_data_range_travels.drop(columns=['Datum', 'UurgroepOmschrijving (van aankomst)',
                                                                  'AankomstHalteCode','hour'])
    
    # Create DF with all stations with their lon and lat
    stations_lon_lat = gvb_data_range_travels.drop_duplicates(subset=['AankomstHalteNaam'])[['AankomstHalteNaam', 'AankomstLon', 'AankomstLat']]
    stations_lon_lat = stations_lon_lat.set_index('AankomstHalteNaam')
    stations_lon_lat.rename(columns={"AankomstLat": "lng", "AankomstLon": "lat"}, inplace=True)  
    
    gvb_data_range_travels = gvb_data_range_travels.drop(columns=['AankomstLat', 'AankomstLon'])    
    cleaned_gvb = gvb_data_range_travels.copy()
    long_lat = stations_lon_lat.copy()

    # calculate all stations within 1 km from the park
    add_nearby_stations(1,vondelpark_west, vondelpark_west_stations, long_lat)
    add_nearby_stations(1,vondelpark_oost_3, vondelpark_oost_3_stations, long_lat)
    add_nearby_stations(1,vondelpark_oost_2, vondelpark_oost_2_stations, long_lat)
    add_nearby_stations(1,vondelpark_oost_1, vondelpark_oost_1_stations, long_lat)
    add_nearby_stations(1,oosterpark,oosterpark_stations, long_lat)
    add_nearby_stations(1,sarphatipark,sarphatipark_stations, long_lat)
    add_nearby_stations(1,westergasfabriek, westergasfabriek_stations, long_lat)
    add_nearby_stations(1,westerpark_west, westerpark_west_stations, long_lat)
    add_nearby_stations(1,westerpark_centrum, westerpark_centrum_stations, long_lat)
    add_nearby_stations(1,westerpark_oost, westerpark_oost_stations, long_lat)
    add_nearby_stations(1,rembrandtpark_noord,rembrandtpark_noord_stations, long_lat)
    add_nearby_stations(1,rembrandtpark_zuid,rembrandtpark_zuid_stations, long_lat)
    add_nearby_stations(1,erasmuspark,erasmuspark_stations, long_lat)
    add_nearby_stations(1,amstelpark,amstelpark_stations, long_lat)
    add_nearby_stations(1,park_frankendael,park_frankendael_stations, long_lat)
    add_nearby_stations(1,beatrixpark,beatrixpark_stations, long_lat)
    add_nearby_stations(1,flevopark,flevopark_stations, long_lat)
    add_nearby_stations(1,gaasperpark,gaasperpark_stations, long_lat)
    add_nearby_stations(1,nelson_mandelapark,nelson_mandelapark_stations, long_lat)
    add_nearby_stations(1,noorderpark,noorderpark_stations, long_lat)
    add_nearby_stations(1,sloterpark,sloterpark_stations, long_lat)
    add_nearby_stations(1,wh_vliegenbos,wh_vliegenbos_stations, long_lat)

    vondelpark_west_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(vondelpark_west_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()
    vondelpark_oost1_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(vondelpark_oost_1_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()
    vondelpark_oost2_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(vondelpark_oost_2_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()
    vondelpark_oost3_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(vondelpark_oost_3_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()
    oosterpark_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(oosterpark_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()
    sarphatipark_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(sarphatipark_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()
    westerpark_west_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(westerpark_west_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()
    westerpark_centrum_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(westerpark_centrum_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()
    westerpark_oost_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(westerpark_oost_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()
    westergasfabriek_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(westergasfabriek_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()
    rembrandtpark_noord_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(rembrandtpark_noord_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()
    rembrandtpark_zuid_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(rembrandtpark_zuid_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()
    erasmuspark_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(erasmuspark_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()

    amstelpark_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(amstelpark_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()
    park_frankendael_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(park_frankendael_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()
    beatrixpark_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(beatrixpark_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()
    flevopark_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(flevopark_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()
    gaasperpark_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(gaasperpark_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()
    nelson_mandelapark_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(nelson_mandelapark_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()
    noorderpark_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(noorderpark_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()
    sloterpark_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(sloterpark_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()
    wh_vliegenbos_journeys = cleaned_gvb[cleaned_gvb['AankomstHalteNaam'].isin(wh_vliegenbos_stations)].drop(columns=['AankomstHalteNaam']).groupby('date').sum()

    # concatenate all dataframes into one, for later usage
    vondelpark_west_journeys["park"] = 'vondelpark_west'
    vondelpark_oost1_journeys["park"] = 'vondelpark_oost_1'
    vondelpark_oost2_journeys["park"] = 'vondelpark_oost_2'
    vondelpark_oost3_journeys["park"] = 'vondelpark_oost_3'
    oosterpark_journeys["park"] = "oosterpark"
    sarphatipark_journeys["park"] = "sarphatipark"
    westerpark_west_journeys["park"] = "westerpark_west"
    westerpark_oost_journeys["park"] = "westerpark_oost"
    westerpark_centrum_journeys["park"] = "westerpark_centrum"
    westergasfabriek_journeys["park"] = "westergasfabriek"
    rembrandtpark_noord_journeys["park"] = "rembrandtpark_noord"
    rembrandtpark_zuid_journeys["park"] = "rembrandtpark_zuid"
    erasmuspark_journeys["park"] = "erasmuspark"

    amstelpark_journeys["park"] = "amstelpark"
    park_frankendael_journeys["park"] = "park_frankendael"
    beatrixpark_journeys["park"] = "beatrixpark"
    flevopark_journeys["park"] = "flevopark"
    gaasperpark_journeys["park"] = "gaasperpark"
    nelson_mandelapark_journeys["park"] = "nelson_mandelapark"
    noorderpark_journeys["park"] = "noorderpark"
    sloterpark_journeys["park"] = "sloterpark"
    wh_vliegenbos_journeys["park"] = "wh_vliegenbos"

    frames = [vondelpark_west_journeys, vondelpark_oost1_journeys, vondelpark_oost2_journeys, 
              vondelpark_oost3_journeys,oosterpark_journeys, sarphatipark_journeys, westerpark_west_journeys,
             westerpark_centrum_journeys, westerpark_oost_journeys, westergasfabriek_journeys,
             rembrandtpark_noord_journeys, rembrandtpark_zuid_journeys, erasmuspark_journeys,
             amstelpark_journeys, park_frankendael_journeys, beatrixpark_journeys, flevopark_journeys,
             gaasperpark_journeys, nelson_mandelapark_journeys, noorderpark_journeys, sloterpark_journeys, wh_vliegenbos_journeys]

    all_parks_journeys = pd.concat(frames)


    # Make all GVB data 15 min
    vondelpark_west_journeys_15min = vondelpark_west_journeys.resample('15T').pad()
    vondelpark_west_journeys_15min['AantalReizen'] = vondelpark_west_journeys_15min['AantalReizen'] / 4

    vondelpark_oost1_journeys_15min = vondelpark_oost1_journeys.resample('15T').pad()
    vondelpark_oost1_journeys_15min['AantalReizen'] = vondelpark_oost1_journeys_15min['AantalReizen'] / 4

    vondelpark_oost2_journeys_15min = vondelpark_oost2_journeys.resample('15T').pad()
    vondelpark_oost2_journeys_15min['AantalReizen'] = vondelpark_oost2_journeys_15min['AantalReizen'] / 4

    vondelpark_oost3_journeys_15min = vondelpark_oost3_journeys.resample('15T').pad()
    vondelpark_oost3_journeys_15min['AantalReizen'] = vondelpark_oost3_journeys_15min['AantalReizen'] / 4

    oosterpark_journeys_15min = oosterpark_journeys.resample('15T').pad()
    oosterpark_journeys_15min['AantalReizen'] = oosterpark_journeys_15min['AantalReizen'] / 4

    sarphatipark_journeys_15min = sarphatipark_journeys.resample('15T').pad()
    sarphatipark_journeys_15min['AantalReizen'] = sarphatipark_journeys_15min['AantalReizen'] / 4

    rembrandtpark_noord_journeys_15min = rembrandtpark_noord_journeys.resample('15T').pad()
    rembrandtpark_noord_journeys_15min['AantalReizen'] = rembrandtpark_noord_journeys_15min['AantalReizen'] / 4

    rembrandtpark_zuid_journeys_15min = rembrandtpark_zuid_journeys.resample('15T').pad()
    rembrandtpark_zuid_journeys_15min['AantalReizen'] = rembrandtpark_zuid_journeys_15min['AantalReizen'] / 4

    westerpark_centrum_journeys_15min = westerpark_centrum_journeys.resample('15T').pad()
    westerpark_centrum_journeys_15min['AantalReizen'] = westerpark_centrum_journeys_15min['AantalReizen'] / 4

    westerpark_oost_journeys_15min = westerpark_oost_journeys.resample('15T').pad()
    westerpark_oost_journeys_15min['AantalReizen'] = westerpark_oost_journeys_15min['AantalReizen'] / 4

    westerpark_west_journeys_15min = westerpark_west_journeys.resample('15T').pad()
    westerpark_west_journeys_15min['AantalReizen'] = westerpark_west_journeys_15min['AantalReizen'] / 4

    westergasfabriek_journeys_15min = westergasfabriek_journeys.resample('15T').pad()
    westergasfabriek_journeys_15min['AantalReizen'] = westergasfabriek_journeys_15min['AantalReizen'] / 4

    erasmuspark_journeys_15min = erasmuspark_journeys.resample('15T').pad()
    erasmuspark_journeys_15min['AantalReizen'] = erasmuspark_journeys_15min['AantalReizen'] / 4

    amstelpark_journeys_15min= amstelpark_journeys.resample('15T').pad()
    amstelpark_journeys_15min['AantalReizen'] = amstelpark_journeys_15min['AantalReizen'] / 4

    park_frankendael_journeys_15min = park_frankendael_journeys.resample('15T').pad()
    park_frankendael_journeys_15min['AantalReizen'] = park_frankendael_journeys_15min['AantalReizen'] / 4

    beatrixpark_journeys_15min = beatrixpark_journeys.resample('15T').pad()
    beatrixpark_journeys_15min['AantalReizen'] = beatrixpark_journeys_15min['AantalReizen'] / 4

    flevopark_journeys_15min = flevopark_journeys.resample('15T').pad()
    flevopark_journeys_15min['AantalReizen'] = flevopark_journeys_15min['AantalReizen'] / 4

    gaasperpark_journeys_15min = gaasperpark_journeys.resample('15T').pad()
    gaasperpark_journeys_15min['AantalReizen'] = gaasperpark_journeys_15min['AantalReizen'] / 4

    nelson_mandelapark_journeys_15min = nelson_mandelapark_journeys.resample('15T').pad()
    nelson_mandelapark_journeys_15min['AantalReizen'] = nelson_mandelapark_journeys_15min['AantalReizen'] / 4

    noorderpark_journeys_15min = noorderpark_journeys.resample('15T').pad()
    noorderpark_journeys_15min['AantalReizen'] = noorderpark_journeys_15min['AantalReizen'] / 4

    sloterpark_journeys_15min = sloterpark_journeys.resample('15T').pad()
    sloterpark_journeys_15min['AantalReizen'] = sloterpark_journeys_15min['AantalReizen'] / 4

    # create list of unique locations and empty list with potential resono x park lists
    locations = resono["Location"].unique().tolist()
    resono_park_list_15min = []

    for location in locations:   
        # create dynamic name (https://www.delftstack.com/howto/python/python-dynamic-variable-name/)
        name = f"min15_resono_{location.lower()}"
        name = "_".join(name.split())
        resono_park_list_15min.append(name)

        # prepare resono data
        resono_park = resono[resono['Location'] == location]
        #resono_park = resono_park.set_index('End')
        resono_park.index = pd.to_datetime(resono_park.index, utc=True)
        resono_park.index = resono_park.index.tz_convert(None)
        resono_park.index = resono_park.index.tz_localize('utc') 

        parknaam = "_".join(location.lower().split())

        # prepare gvb data of certain park 
        gvb_park = all_parks_journeys[all_parks_journeys["park"] == parknaam]
        gvb_park.index = gvb_park.index.tz_localize('utc')

        # merge gvb and resono to new dynamic df
        globals()[name] = resono_park.loc["2020-10":].join(gvb_park) 
        globals()[name].index = globals()[name].index.tz_convert(None)

    frames = [min15_resono_amstelpark, min15_resono_beatrixpark, min15_resono_erasmuspark,
             min15_resono_flevopark, min15_resono_gaasperpark, min15_resono_nelson_mandelapark,
             min15_resono_noorderpark, min15_resono_oosterpark, min15_resono_park_frankendael,
             min15_resono_sarphatipark, min15_resono_vondelpark_oost_1, min15_resono_sloterpark,
             min15_resono_wh_vliegenbos, min15_resono_vondelpark_oost_2, min15_resono_vondelpark_oost_3, 
             min15_resono_vondelpark_west, min15_resono_westergasfabriek, min15_resono_westerpark_centrum,
             min15_resono_westerpark_oost, min15_resono_westerpark_west]
    min15_all_resono_park = pd.concat(frames)

    min15_all_resono_park['Journeys'] = min15_all_resono_park['AantalReizen'].interpolate(method='linear')
    min15_all_resono_park = min15_all_resono_park.drop(columns=['AantalReizen', 'park']).dropna()
    min15_all_resono_park['Journeys'].loc[min15_all_resono_park.between_time('01:01:00', '06:30:00')['Journeys'].index] = 0 
    return min15_all_resono_park
    
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

def fill_missing_values_dataframe(df, park, minimal_visits):
    '''
    Add missing dates and give value from day before or if not available interpolate. Also replace value if visits
    is lower than minimal_visits.
    
    :df: the dataframe of the park
    
    :park: give as input about which park it is   
    '''     
    idx = pd.date_range(df.index.min(), df.index.max(), freq="15min")
    df_without_missing = df.reindex(idx)
    df_without_missing[df_without_missing['Visits'] < minimal_visits] = np.NaN
    df_without_missing['Location'] = park
    df_without_missing['Date'] = df_without_missing.index.date
    df_without_missing['Time'] = df_without_missing.index.time
    df_without_missing = df_without_missing.groupby(df_without_missing.index.hour).ffill()
    df_without_missing = df_without_missing.interpolate()
    df_without_missing = df_without_missing.backfill()
    df_without_missing = df_without_missing.reset_index().rename(columns={"index": "Datetime"})
    return df_without_missing

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
    df = df[df.Location != 'Rembrandtpark Noord']
    df = df[df.Location != 'Rembrandtpark Zuid']
    
    df['Location'] = df['Location'].str.replace('W.H. Vliegenbos', 'WH Vliegenbos')
    
    df_no_missing = pd.DataFrame(columns=['Datetime','Location','Visits','Date','Time'])

    for park in df["Location"].unique():
        result = fill_missing_values_dataframe(df[df['Location'] == park], park, 25)
        df_no_missing = df_no_missing.append(result)    
    
    return df_no_missing.set_index("Datetime")

def preprocessCOVID(df_holi_gvb_weather):
    Mobility_df_2020 = pd.read_csv('2020_NL_Region_Mobility_Report.csv')
    Mobility_df_2021 = pd.read_csv('2021_NL_Region_Mobility_Report.csv')
    Mobility_df_2022 = pd.read_csv('2022_NL_Region_Mobility_Report.csv')
    Mobility_df = Mobility_df_2020.append(Mobility_df_2021).append(Mobility_df_2022)
    stringency_df = pd.read_csv('covid-stringency-index.csv')
    
    amsterdam_mobility = Mobility_df[Mobility_df['sub_region_2'] == 'Government of Amsterdam']
    amsterdam_mobility = amsterdam_mobility.drop(columns=['metro_area', 'iso_3166_2_code', 'census_fips_code', 'place_id','sub_region_1', 'sub_region_2','country_region','country_region_code'])
    amsterdam_mobility['date'] = pd.to_datetime(amsterdam_mobility['date'])

    df_holi_gvb_weather['Date'] = pd.to_datetime(df_holi_gvb_weather['Date'])

    stringency_df['Day'] = pd.to_datetime(stringency_df['Day'])
    stringency_df = stringency_df[stringency_df['Entity']=='Netherlands']
    stringency_df = stringency_df.drop(columns=['Code', 'Entity'])

    merged_df = df_holi_gvb_weather.merge(amsterdam_mobility, how='left', left_on='Date', right_on='date')
    merged_df = merged_df.iloc[:, :-3]

    double_merged_df = merged_df.merge(stringency_df, how = 'left', left_on='Date', right_on='Day')
    double_merged_df = double_merged_df.drop(columns=['Day_y', 'date'])
    return double_merged_df

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
    
    merge_resono_weather['Date'] = pd.to_datetime(merge_resono_weather['Date'])
    df_holiday['End_Dates'] = pd.to_datetime(df_holiday['End_Dates'])
    
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
    #Resono_Holi = Resono_Holi.drop('Holiday_name', axis=1)
    
    
    # Holidays
    Resono_Holi_Dummies = pd.get_dummies(Resono_Holi, columns=["Holiday_Name"])

    return Resono_Holi_Dummies

def remove_outliers(df, gamma=0.01, nu=0.03):
    '''
    Remove outliers for each location with a One-Class SVM.
    
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

def add_time_vars(data, onehot=True):
    '''
    Adds columns for the month and weekday, and also the one-hot encoding or the cyclical versions of those features.

    :data: Dataframe that contains the a column with the datetime
    :onehot: Use onehot encoding if true and cyclical features if false (default = True)
    
    Returns a Dataframe with either the one-hot encoding or the sine and cosine of the month, weekday and time added
    '''
    data = data.reset_index()
    if onehot == True:
        data['Year'] = pd.Categorical(data['Datetime'].dt.year)
        data['Month'] = pd.Categorical(data['Datetime'].dt.month)
        data['Weekday'] = pd.Categorical(data['Datetime'].dt.weekday)
        data['Hour'] =  pd.Categorical(data['Datetime'].dt.hour)
        data['Minute'] =  pd.Categorical(data['Datetime'].dt.minute)

        year_dummies = pd.get_dummies(data[['Year']], prefix='Year_')
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


def predict(data, location, pred_params, N_boost=100):
    '''
    Predict the amount of visits using XGBoost
    
    :data: Dataframe with all the data
    :location: The location of the park to make predictions for
    :pred_params: A list of the names of the predictor variables
    :N_boost: Number of boost rounds during training (default = 100)
    
    Returns nothing (yet)
    '''
    # Select data for a specific park
    data = data[data['Location'] == location]
    
    # Split the data into input and output variables
    X = data[pred_params]
    y = data['Visits']

    # Split the data into test and train sets
    train_X, test_X, train_y, test_y = train_test_split(X, y,
                          test_size = 0.3, random_state = 123)

    # Convert test and train set to DMatrix objects
    train_dmatrix = xgb.DMatrix(data = train_X, label = train_y)
    test_dmatrix = xgb.DMatrix(data = test_X, label = test_y)
    
    # Set parameters for base learner
    params = {
        'booster': 'gblinear',
#         'colsample_bynode': 0.8,
        'learning_rate': 1,
#         'max_depth': 15,
#         'num_parallel_tree': 100,
        'objective': 'reg:squarederror',
#         'subsample': 0.8,
#         'tree_method': 'gpu_hist'
    }

    # Fit the data and make predictions
    model = xgb.train(params = params, dtrain = train_dmatrix, num_boost_round = N_boost)
    pred = model.predict(test_dmatrix)
    pred1 = [0 if x <= 0 else x for x in pred]
    predictions = pd.DataFrame({'Predicted visitors': pred1,
                                'Actual visitors': test_y})
    predictions = predictions.clip(lower=0)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_y, predictions['Predicted visitors']))
    mae = mean_absolute_error(test_y, predictions['Predicted visitors'])
    print("RMSE : % f" %(rmse))
    print("MAE : % f" %(mae))
    return predictions, pred1
    
def getDataframe():
    df_Weather2020 = preprocessWeather("KNMI (Weather) 2020-2021/uurgeg_240_2011-2020.txt")
    df_Weather2021 = preprocessWeather("KNMI (Weather) 2020-2021/uurgeg_240_2021-2030.txt")
    df_resono = preprocessResono("resono_2020_2022.csv")
    df_gvb = preprocessGVB(r'GVB/*.csv', df_resono)

    df_holiday = preprocessHoliday('holidays.csv')
    df_weather = mergeWeatherFiles(df_Weather2020, df_Weather2021)
    df_resono_weather = mergeWeatherResonoHoliday(df_gvb, df_weather, df_holiday)
    df_merged = Target_OneHotEncoding(df_resono_weather)
    dataframe = preprocessCOVID(df_merged)
    return dataframe

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

def add_nearby_stations(radius, center_point_dict, add_to_list, long_lat):
    """
    radius in km
    center_point_dict needs to be a dictionary with the lon and lat from a location
    add_to_list specify to which list this needs to be added (for example vondelpark)
    """
    latpark = center_point_dict[0]['lat']
    lonpark = center_point_dict[0]['lng']

    # check for every station if it is within 1 km distance of the park
    for station in range(len(long_lat)):
        name_station = long_lat.iloc[station].name
        latstation = long_lat.iloc[station].lat
        lonstation = long_lat.iloc[station].lng

        a = haversine(lonpark, latpark, lonstation, latstation)
        
        if a <= radius:
            add_to_list.append(name_station)
    
def removeOutliers(dataframe):
    dataframe[["Date", 'Time']] = dataframe[["Date", 'Time']].astype('str')
    dataframe['Datetime'] = pd.to_datetime(dataframe.Date + ' ' + dataframe.Time, format='%Y/%m/%d %H:%M:%S')
    dataframe = dataframe.set_index('Datetime')

    df_no_outliers = remove_outliers(dataframe)
    df_no_outliers_int = interpolate_df(df_no_outliers, backfill=False)

    df_resono_no_outliers = dataframe.copy()
    df_resono_no_outliers['Visits'] = df_no_outliers_int['Visits']
    return df_resono_no_outliers    