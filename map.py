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
# import plotly
# import plotly.tools as tls

warnings.filterwarnings('ignore')

begin_time = dt.datetime.now()

IMG_RATIO = 1280/380

IMG_HEIGHT = int(880 / IMG_RATIO)
IMG_WIDTH = int(1280 / IMG_RATIO)

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

if not os.path.exists("images"):
    os.mkdir("images")

def get_location_url(location):
    query = location + ', Amsterdam'
    return urllib.parse.quote(query)

def get_html(data_now, data_next, park_suggs):
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
    sns.set_context("poster")
    plt.style.use('seaborn-poster')

    time = pd.to_datetime(prev['Time'][-1])
    next_dt = time + dt.timedelta(hours=1)

    data.loc[data.index[0], 'Predicted visitors'] = data.loc[data.index[0], 'Actual visitors']

    f, axes = plt.subplots()
    plt.plot(prev['Visits'], 'o-', linewidth=8, color='#b52222', markersize=15, markerfacecolor='#661111')
    plt.plot(data['Predicted visitors'], 'o:', color='#b52222', linewidth=8, markersize=15, markerfacecolor='#661111')
    plt.axhline(y=baseline[baseline['Location'] == file_name]['Visits'].values[0], color='#b04343', linestyle='-', linewidth=4)
    # plt.plot(data['Actual visitors'], label='Actual visitors')

    plt.setp(axes, xticks=np.linspace(0,12,4),
         xticklabels=[prev['Time'][0], prev['Time'][int(len(prev) / 2)], prev['Time'][-1], next_dt.strftime('%H:%M:%S')])

    # plt.xlabel('Time')
    # plt.ylabel('Visitors')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.legend(loc='upper left', prop={'size': 25})

    plt.savefig('images/'+file_name, transparent=True)
    # plotly_fig = tls.mpl_to_plotly(f)
    # plotly.offline.plot(plotly_fig, filename='test.html', auto_open=False, config={'displayModeBar': False})

    plt.cla()
    plt.clf()

def get_predictions(model, data, park_name):
    predictor_cols = data.columns.to_list()[8:]
    # preds = h.predict_XGBoost(model, data, park_name, predictor_cols)
    preds = h.predict_catboost(model, data, park_name)
    return preds

def get_geodict(path):
    geofiles = glob.glob(path)
    d = {}
    for file in geofiles:
        with open(file) as f:
            geo = json.load(f)
        d[file.split('.')[0][14:]] = geo

    return d

def get_time():
    time = dt.datetime.now()
    time = time - dt.timedelta(minutes=time.minute % 15,
                               seconds=time.second)
    return time

def get_data(data):
    time = get_time()
    date = data.index[0].split(' ')[0].split('-')
    prediction_date = dt.datetime(int(date[0]), int(date[1]), int(date[2]))
    idx =  dt.datetime(prediction_date.year, prediction_date.month, prediction_date.day, time.hour, time.minute, 0)
    idx_prev = dt.datetime(prediction_date.year, prediction_date.month, prediction_date.day, time.hour - 2, time.minute, 0)
    idx_next = dt.datetime(prediction_date.year, prediction_date.month, prediction_date.day, time.hour + 1, time.minute, 0)

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
    date = data.index[0].split(' ')[0].split('-')
    prediction_date = dt.datetime(int(date[0]), int(date[1]), int(date[2]))
    time = dt.datetime.now()
    prediction_time = dt.datetime(prediction_date.year, prediction_date.month, prediction_date.day, time.hour, time.minute, 0)
    time = h.ceil_dt(prediction_time + dt.timedelta(minutes=15), dt.timedelta(minutes=15))

    current_lat = park_locations[current_location][0]['lat']
    current_lng = park_locations[current_location][0]['lng']

    preds = {}
    for loc in locations:
        pred_data = data.loc[str(time)]
        pred = get_predictions(model, pred_data, loc)
        preds[loc] = pred['Predicted visitors'].values[0]

    values = list(preds.values())
    park_suggestion = h.getParkSuggestion(resono, prediction_date, current_lat, current_lng, values, time)

    business = []

    for loc in park_suggestion.Location:
        if park_suggestion[park_suggestion['Location'] == loc]['Predictions'].values[0] < 0.85 * park_suggestion[park_suggestion['Location'] == loc]['Visits'].values[0]:
            business.append('Not busy')
        elif park_suggestion[park_suggestion['Location'] == loc]['Predictions'].values[0] > 1.15 * park_suggestion[park_suggestion['Location'] == loc]['Visits'].values[0]:
            business.append('Busier than usual')
        else:
            business.append('As busy as usual')

    park_suggestion['Business'] = business
    # print(park_suggestion)

    return park_suggestion

def get_baseline(data, resono):
    date = data.index[0].split(' ')[0].split('-')
    prediction_date = dt.datetime(int(date[0]), int(date[1]), int(date[2]))
    time = dt.datetime.now()
    time = time - dt.timedelta(minutes=time.minute % 15, seconds=time.second)
    idx = dt.datetime(prediction_date.year, prediction_date.month, prediction_date.day, time.hour, time.minute, 0)
    current_data = data.loc[str(idx)][['Location', 'Visits']]

    baseline = h.get_baseline(resono, prediction_date, time)
    business = []

    for loc in locations:
        if current_data[current_data['Location'] == loc]['Visits'].values[0] < 0.85 * baseline[baseline['Location'] == loc]['Visits'].values[0]:
            business.append('Not busy')
        elif current_data[current_data['Location'] == loc]['Visits'].values[0] > 1.15 * baseline[baseline['Location'] == loc]['Visits'].values[0]:
            business.append('Busier than usual')
        else:
            business.append('As busy as usual')

    baseline['Business'] = business
    return baseline

data = pd.read_csv('Models Catboost/dec_19_2021.csv', index_col=1)
locations = data.Location.unique()

resono = pd.read_csv('CSVs/resono_2020_2022.csv', index_col=0)

d = get_geodict('GeoJSON files/*.geojson')
geos = list(d.keys())

# print('Loading model...')
# model = xgb.Booster()
# model.load_model("shittymodel.json")
cls = catboost.CatBoostRegressor()
# print('Done!')

baseline = get_baseline(data, resono)

# ~~~~~  MAP IMPLEMENTATION  ~~~~~

style_function_gr = lambda x: {'fillColor': '#00ff00',
                            'color':'#000000',
                            'fillOpacity': 0.4,
                            'weight': 0.5}

style_function_yl = lambda x: {'fillColor': '#ffff00',
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

m = folium.Map(
    location=[52.349639, 4.916978],
    tiles="cartodbpositron",
    zoom_start=13,
)

fgp = folium.FeatureGroup(name="Parks")

print('Making predictions for the parks...')

for park in geos:
    g = d[park]
    name = g['features'][0]['properties']['Naam']
    print(name)
    if name in locations:
        data_loc = data[data['Location'] == name]
        prev, next = get_data(data_loc)

        model = cls.load_model('Models Catboost/model_' + name.lower().replace(" ", "_") + '.json', "json")  # load model
        # cat_model.save_model('model_' + x.lower().replace(" ", "_") + '.json', format="json")
        preds = get_predictions(model, next, name)
        park_suggs = create_park_suggestions(data, model, name)
        create_plot(prev, preds, name)

        if baseline[baseline['Location'] == name][['Business']].values[0] == 'Not busy':
            style_function = style_function_gr
        elif baseline[baseline['Location'] == name][['Business']].values[0] == 'Busier than usual':
            style_function = style_function_rd
        else:
            style_function = style_function_yl

        html = get_html(prev.iloc[-1], preds.iloc[-1], park_suggs)
        test = folium.Html(html, script=True)
        popup = folium.Popup(test, max_width=400)

        gjson = folium.GeoJson(g,
                               name=name,
                               style_function=style_function,
                               highlight_function=highlight_function,
                              )
        gjson.add_child(popup)

        fgp.add_child(gjson)

fgp.add_to(m)
print('Done!')
m.save('index.html')

print(dt.datetime.now() - begin_time)
