import pandas as pd
import numpy as np
import folium
import json
import urllib.parse
import matplotlib.pyplot as plt
import seaborn as sns
import map_helper as h
import datetime as dt
from datetime import timedelta
import xgboost as xgb
import os
import warnings
import glob
import catboost

#Ignore filter warnings
warnings.filterwarnings('ignore')

# Calculate how long it takes to run the file
begin_time = dt.datetime.now()

# Image ratio for rescaling the plots
IMG_RATIO = 1280/380

# Height and width for the plots
IMG_HEIGHT = int(880 / IMG_RATIO)
IMG_WIDTH = int(1280 / IMG_RATIO)

# Dict with the latitude and longitudes of the parks
park_locations = {
'Vondelpark West' : [{'lat': 52.356496, 'lng': 4.861447}],
'Vondelpark Oost 3' : [{'lng': 4.869217, 'lat': 52.358252}],
'Vondelpark Oost 2' : [{'lng': 4.874692, 'lat': 52.359798}],
'Vondelpark Oost 1' : [{'lng': 4.879652, 'lat': 52.360991}],
'Oosterpark' : [{'lng': 4.920558, 'lat': 52.360098}],
'Sarphatipark' : [{'lng': 4.896375, 'lat': 52.354364}],
'Westerpark West' : [{'lng': 4.867128, 'lat': 52.387099}],
'Westerpark Centrum' : [{'lng': 4.873268, 'lat': 52.387374}],
'Westerpark Oost' : [{'lng': 4.878379, 'lat': 52.386379}],
'Westergasfabriek' : [{'lng': 4.869769, 'lat': 52.385920}],
'Rembrandtpark Noord' : [{'lng': 4.846573, 'lat': 52.366664}],
'Rembrandtpark Zuid' : [{'lng': 4.846932, 'lat': 52.361161}],
'Erasmuspark' : [{'lng': 4.851909, 'lat': 52.374808}],
'Amstelpark' : [{'lng': 4.894404, 'lat': 52.330409}],
'Park Frankendael' : [{'lng': 4.929839, 'lat': 52.350703}],
'Beatrixpark' : [{'lng': 4.881352, 'lat': 52.342471}],
'Flevopark' : [{'lng': 4.947881, 'lat': 52.360087}],
'Gaasperpark' : [{'lng': 4.992192, 'lat': 52.310420}],
'Nelson Mandelapark' : [{'lng': 4.963691, 'lat': 52.312204}],
'Noorderpark' : [{'lng': 4.919606, 'lat': 52.392651}],
'Sloterpark' : [{'lng': 4.811894, 'lat': 52.366219}],
'WH Vliegenbos' : [{'lng': 4.931495, 'lat': 52.388802}]
}

# Create a filepath for the plots
if not os.path.exists("images"):
    os.mkdir("images")

def get_location_url(location):
    '''
    Encode the locations in URL for the Google Maps links

    :location: string with the name of the park

    Returns the URL encoding of the park name
    '''
    query = location + ', Amsterdam'
    return urllib.parse.quote(query)

def get_html(data_now, data_next, park_suggs):
    '''
    Create the HTML code for the popups

    :data_now: Dataframe with data of the previous two hours up to the current time
    :data_next: Dataframe with the predictions of the next hour
    :park_suggs: Dataframe with the park suggestions

    Returns a string with HTML code
    '''
    html = f'''
        <html>
            <head>
                <link rel="stylesheet" type="text/css" href="popup.css">
            </head>
            <body>
                <div class=popup>
                <div class='header'>
                    {data_now.Location}
                </div>

                <div class='park_predictions'>
                    <div class='data'>
                        <div class='people'>
                            {int(data_now.Visits)} people
                        </div>

                        <div class='text_pred'>
                            Measured at {data_now.Time[:5]}
                        </div>

                        <div class='people'>
                            {int(data_next.loc['Predicted visitors'])} people
                        </div>

                        <div class ='text_pred'>
                            Expected in the next hour
                        </div>
                    </div>

                    <div class='park_business'>
                        {baseline[baseline['Location'] == data_now.Location]['Business'].values[0]}
                    </div>

                    <div class='image'>
                        <img id='predictions' src="images/{data_now.Location}.png" alt="{data_now.Location}" height={IMG_HEIGHT} width={IMG_WIDTH}><br>
                    </div>

                    <div class='mapslink'>
                        <a id='mapslink' href=https://www.google.com/maps/dir/?api=1&destination={get_location_url(data_now.Location)}, target="_blank">Get directions to {data_now.Location}</a>
                    </div>
                </div>

                <div class='park_recommendations'>
                    <div class='park_header'>
                        Top 3 park suggestions:
                    </div>
                    <div class='text_sugg'>
                        Park name:<br>
                        <a id='mapslink2' href=https://www.google.com/maps/dir/?api=1&origin={get_location_url(data_now.Location)}&destination={get_location_url(park_suggs.iloc[0, 0])}, target="_blank">{park_suggs.iloc[0, 0]}</a><br>
                        <a id='mapslink2' href=https://www.google.com/maps/dir/?api=1&origin={get_location_url(data_now.Location)}&destination={get_location_url(park_suggs.iloc[1, 0])}, target="_blank">{park_suggs.iloc[1, 0]}</a><br>
                        <a id='mapslink2' href=https://www.google.com/maps/dir/?api=1&origin={get_location_url(data_now.Location)}&destination={get_location_url(park_suggs.iloc[2, 0])}, target="_blank">{park_suggs.iloc[2, 0]}</a><br>
                    </div>
                    <div class='business_sugg'>
                        Predicted crowdedness at {park_suggs.iloc[0, 3][:5]}:<br>
                        {park_suggs.iloc[0, -1]}<br>
                        {park_suggs.iloc[1, -1]}<br>
                        {park_suggs.iloc[2, -1]}<br>
                    </div>
                    <div class='distance'>
                        Distance:<br>
                        {park_suggs.iloc[0, 1]} km<br>
                        {park_suggs.iloc[1, 1]} km<br>
                        {park_suggs.iloc[2, 1]} km<br>
                    </div>
                </div>
                </div>
            </body>
        </html>'''
    return html

def create_plot(prev, data, file_name):
    '''
    Create the plots for the predictions and saves them in an images folder.

    :prev: Dataframe with data of the previous two hours
    :data: Dataframe with the predictions for the next hour
    :file_name: String with the name of the park

    Returns nothing
    '''
    sns.set_context("poster")
    plt.style.use('seaborn-poster')

    # Get the time of the next hour for the x-axis labels
    time = pd.to_datetime(prev['Time'][-1])
    next_dt = time + dt.timedelta(hours=1)

    # Make the first row of predictions the actual visitors so we get a continious line
    data.loc[data.index[0], 'Predicted visitors'] = data.loc[data.index[0], 'Actual visitors']

    # Create the lines for the plot
    f, axes = plt.subplots()
    plt.plot(prev['Visits'], 'o-', linewidth=8, color='#b52222', markersize=15, markerfacecolor='#661111')
    plt.plot(data['Predicted visitors'], 'o:', color='#b52222', linewidth=8, markersize=15, markerfacecolor='#661111')
    plt.axhline(y=baseline[baseline['Location'] == file_name]['Visits'].values[0], color='#b04343', linestyle='-', linewidth=4)

    # Set the x-axis labels to the correct times
    plt.setp(axes, xticks=np.linspace(0,12,4),
         xticklabels=[prev['Time'][0], prev['Time'][int(len(prev) / 2)], prev['Time'][-1], next_dt.strftime('%H:%M:%S')])

    # Set the size of the axes
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Save the plot to the image folder
    plt.savefig('images/'+file_name, transparent=True)

    # Clear the axes and plot
    plt.cla()
    plt.clf()

def get_geodict(path):
    '''
    Create a dict with all the GeoJSON files

    :path: String with the path to all the GeoJSON files

    Returns a dict
    '''
    geofiles = glob.glob(path)
    d = {}
    for file in geofiles:
        with open(file) as f:
            geo = json.load(f)
        d[file.split('.')[0][14:]] = geo
    return d

def split_data(data):
    '''
    Create two Dataframes with the data of the previous two hours and the next
    hour based on the current time

    :data: Dataframe with the data of the day to make predictions for

    Returns
    :prev: Dataframe with data of the previous two hours
    :next: Dataframe with data of the next hour
    '''
    # Get the current time and round it to the previous quarter hour
    time = dt.datetime.now()
    time = time - dt.timedelta(minutes=time.minute % 15,
                               seconds=time.second)
    # Get the date to make predictions for
    date = data.index[0].split(' ')[0].split('-')
    prediction_date = dt.datetime(int(date[0]), int(date[1]), int(date[2]))

    # Get the datetime index for two hours before and one hour after the current time
    idx =  dt.datetime(prediction_date.year, prediction_date.month, prediction_date.day, time.hour, time.minute, 0)
    idx_prev = dt.datetime(prediction_date.year, prediction_date.month, prediction_date.day, time.hour - 2, time.minute, 0)
    idx_next = dt.datetime(prediction_date.year, prediction_date.month, prediction_date.day, time.hour + 1, time.minute, 0)

    # If it's after 23:00, take all remaining data as prediction data or if it's
    # before 2:00, take all data before the current time as previous data.
    if time.hour == 23:
        prev = data.loc[str(idx_prev):str(idx)]
        next = data.loc[str(idx):]
    elif time.hour < 2:
        prev = data.loc[:str(idx)]
        next = data.loc[str(idx):str(idx_next)]
    else:
        prev = data.loc[str(idx_prev):str(idx)]
        next = data.loc[str(idx):str(idx_next)]

    return prev, next

def create_park_suggestions(data, model, current_location):
    '''
    Create park suggestions for ~30 minutes after the current time based on the
    current location

    :data: Dataframe with the data of the prediction date
    :model: Model used to make predictions
    :current_location: String with the name of the current location

    Returns a Dataframe with the park suggestions
    '''
    # Get the date and time to make the predictions for
    date = data.index[0].split(' ')[0].split('-')
    prediction_date = dt.datetime(int(date[0]), int(date[1]), int(date[2]))
    time = dt.datetime.now()
    prediction_time = dt.datetime(prediction_date.year, prediction_date.month, prediction_date.day, time.hour, time.minute, 0)
    time = h.ceil_dt(prediction_time + dt.timedelta(minutes=15), dt.timedelta(minutes=15))

    # Get the latitude and longitude of the current location
    current_lat = park_locations[current_location][0]['lat']
    current_lng = park_locations[current_location][0]['lng']

    # Make the predictions to base the park suggestions on
    preds = {}
    for loc in locations:
        pred_data = data.loc[str(time)]
        pred = h.predict(model, pred_data, loc, catboost=True)
        preds[loc] = pred['Predicted visitors'].values[0]

    # Make the park suggestions
    values = list(preds.values())
    park_suggestion = h.getParkSuggestion(resono, prediction_date, current_lat, current_lng, values, time)

    # Create the business categories
    park_suggestion['Business'] = ['test'] * len(park_suggestion.index)

    for loc in park_suggestion.Location:
        if park_suggestion[park_suggestion['Location'] == loc]['Predictions'].values[0] < 0.85 * park_suggestion[park_suggestion['Location'] == loc]['Visits'].values[0]:
            park_suggestion.loc[park_suggestion['Location'] == loc, 'Business'] = 'Not busy'
        elif park_suggestion[park_suggestion['Location'] == loc]['Predictions'].values[0] > 1.15 * park_suggestion[park_suggestion['Location'] == loc]['Visits'].values[0]:
            park_suggestion.loc[park_suggestion['Location'] == loc, 'Business'] = 'Busier than usual'
        else:
            park_suggestion.loc[park_suggestion['Location'] == loc, 'Business'] = 'As busy as usual'

    return park_suggestion

def get_baseline(data, resono):
    '''
    Create the baseline for every park for the current time

    :data: Dataframe with the data of the prediction date
    :resono: Dataframe with all the preprocessed Resono data

    Returns a Dataframe with the baselines for the parks
    '''
    # Get the date and time to make the baseline for
    date = data.index[0].split(' ')[0].split('-')
    prediction_date = dt.datetime(int(date[0]), int(date[1]), int(date[2]))
    time = dt.datetime.now()
    time = time - dt.timedelta(minutes=time.minute % 15, seconds=time.second)
    idx = dt.datetime(prediction_date.year, prediction_date.month, prediction_date.day, time.hour, time.minute, 0)

    # Get the location and visits data for the current time
    current_data = data.loc[str(idx)][['Location', 'Visits']]

    # Create the baseline for the current date and time
    baseline = h.get_baseline(resono, prediction_date, time)

    # Add the business categories based on the baseline and the current visitors
    baseline['Business'] = ['test'] * len(baseline.index)
    for loc in locations:
        if current_data[current_data['Location'] == loc]['Visits'].values[0] < (0.85 * baseline[baseline['Location'] == loc]['Visits'].values[0]):
            baseline.loc[baseline['Location'] == loc, 'Business'] = 'Not busy'
        elif current_data[current_data['Location'] == loc]['Visits'].values[0] > (1.15 * baseline[baseline['Location'] == loc]['Visits'].values[0]):
            baseline.loc[baseline['Location'] == loc, 'Business'] = 'Busier than usual'
        else:
            baseline.loc[baseline['Location'] == loc, 'Business'] = 'As busy as usual'

    return baseline

# Get the data for the prediction date
data = pd.read_csv('Models Catboost/dec_19_2021.csv', index_col=1)
locations = data.Location.unique()

# Get all the Resono data
resono = pd.read_csv('data_preprocessing/data_preprocessing.csv', index_col=0)

# Get all the GeoJSON files
d = get_geodict('GeoJSON files/*.geojson')
geos = list(d.keys())

# Select wether to use XGBoost or Catboost
# model = xgb.Booster()
cls = catboost.CatBoostRegressor()

# Create the baseline
baseline = get_baseline(data, resono)

# ~~~~~  MAP IMPLEMENTATION  ~~~~~

# Style functions for the different business levels of the parks
style_function_gr = lambda x: {'fillColor': '#00ff00',
                            'color':'#000000',
                            'fillOpacity': 0.4,
                            'weight': 0.5}

style_function_yl = lambda x: {'fillColor': '#ff7b00',
                            'color':'#000000',
                            'fillOpacity': 0.4,
                            'weight': 0.5}

style_function_rd = lambda x: {'fillColor': '#ff0000',
                            'color':'#000000',
                            'fillOpacity': 0.4,
                            'weight': 0.5}

highlight_function = lambda x: {'fillColor': '#000000',
                                'color':'#000000',
                                'fillOpacity': 0.5,
                                'weight': 0.5}

# Create the base map and set the location to Amsterdam
m = folium.Map(
    location=[52.349639, 4.916978],
    tiles="cartodbpositron",
    zoom_start=13,
)

# Create a feature group for all the parks
fgp = folium.FeatureGroup(name="Parks")

print('Making predictions for the parks...')

# Loop through all the parks we have GeoJSON files for
for park in geos:
    # Extract the GeoJSON file from the dict
    g = d[park]
    name = g['features'][0]['properties']['Naam']
    print(name)
    # Only make predictions if there also is Resono data for the parks
    if name in locations:
        # Get the data for the current park
        data_loc = data[data['Location'] == name]
        # Split the data into historic and prediction data
        prev, next = split_data(data_loc)
        # Load the model for the current park
        model = cls.load_model('Models Catboost/model_' + name.lower().replace(" ", "_") + '.json', "json")
        # Make the predictions
        preds = h.predict(model, next, name, catboost=True)
        # Create the park suggestions
        park_suggs = create_park_suggestions(data, model, name)
        # Put the historic data and the predictions in a plot
        create_plot(prev, preds, name)

        # Check which style function needs to be used
        if baseline[baseline['Location'] == name][['Business']].values[0] == 'Not busy':
            style_function = style_function_gr
        elif baseline[baseline['Location'] == name][['Business']].values[0] == 'Busier than usual':
            style_function = style_function_rd
        else:
            style_function = style_function_yl

        # Convert the information to HTML code
        html = folium.Html(get_html(prev.iloc[-1], preds.iloc[-1], park_suggs), script=True)
        # Put the HTML code into a popup
        popup = folium.Popup(html, max_width=400)
        # Create a GeoJson object for the park
        gjson = folium.GeoJson(g,
                               name=name,
                               style_function=style_function,
                               highlight_function=highlight_function,
                              )
        # Add the popup to the GeoJson object
        gjson.add_child(popup)
        # Add the GeoJson object to the feature group
        fgp.add_child(gjson)

# Add the feature group to the base map
fgp.add_to(m)
print('Done!')
# Convert the map to HTML code and save it as index.html
m.save('index.html')

# Print how long it took to run the code
print(dt.datetime.now() - begin_time)
