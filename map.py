import pandas as pd
import numpy as np
import folium
import json
import urllib.parse
import matplotlib.pyplot as plt
import seaborn as sns
import predict as h
import datetime as dt
import xgboost as xgb
import os
import glob

IMG_RATIO = 1280/380

IMG_HEIGHT = int(880 / IMG_RATIO)
IMG_WIDTH = int(1280 / IMG_RATIO)

if not os.path.exists("images"):
    os.mkdir("images")

def get_location_url(location):
    query = location + ', Amsterdam'
    return urllib.parse.quote(query)

def get_html(data_now, data_next):
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

                    <div class='image'>
                        <img id='predictions' src="images/{data_now.Location}.png" alt="{data_now.Location}", height={IMG_HEIGHT} width={IMG_WIDTH}><br>
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
                        -Park 1<br>
                        -Park 2<br>
                        -Park 3<br>
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
    # plt.plot(data['Actual visitors'], label='Actual visitors')
    plt.setp(axes, xticks=np.linspace(0,12,4),
         xticklabels=[prev['Time'][0], prev['Time'][4], prev['Time'][-1], next_dt.strftime('%H:%M:%S')])

    # plt.xlabel('Time')
    # plt.ylabel('Visitors')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.legend(loc='upper left', prop={'size': 25})

    plt.savefig('images/'+file_name, transparent=True)

    plt.cla()
    plt.clf()

def get_predictions(model, data, park_name):
    predictor_cols = data.columns.to_list()[8:]
    preds = h.predict(model, data, park_name, predictor_cols)
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
    idx = dt.datetime(2022, 1, 5, time.hour, time.minute, 0)
    idx_prev = dt.datetime(2022, 1, 5, time.hour - 2, time.minute, 0)
    idx_next = dt.datetime(2022, 1, 5, time.hour + 1, time.minute, 0)

    prev = data.loc[str(idx_prev):str(idx)]
    next = data.loc[str(idx):str(idx_next)]

    return prev, next

d = get_geodict('GeoJSON files/*.geojson')
geos = list(d.keys())

data = pd.read_csv('CSVs/Resono_test.csv', index_col=0)

print('Loading model...')
model = xgb.Booster()
model.load_model("model.json")
print('Done!')


locations = data.Location.unique()
chordata = data[data['Time'] == get_time().strftime('%H:%M:%S')][['Location', 'Visits']]
chordata.set_index('Location')

style_function = lambda x: {'fillColor': '#ffffff',
                            'color':'#000000',
                            'fillOpacity': 0.0,
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

chor = folium.Choropleth(
    geo_data=d['all_parks'],
    name="choropleth",
    data=chordata,
    columns=["Location", "Visits"],
    key_on="properties.Naam",
    bins=6,
    fill_color="YlOrRd",
    fill_opacity=0.8,
    line_opacity=0.7,
    legend_name="Visitors",
    highlight=True,
)

chor.add_to(m)

fgp = folium.FeatureGroup(name="Parks")

print('Making predictions for the parks...')

for park in geos:
    g = d[park]
    name = g['features'][0]['properties']['Naam']
    if name in locations:
        gjson = folium.GeoJson(g,
                               name=name,
                               style_function=style_function,
                               highlight_function=highlight_function,
                              )
        data_loc = data[data['Location'] == name]
        prev, next = get_data(data_loc)
        preds = get_predictions(model, next, name)
        create_plot(prev, preds, name)
        html = get_html(prev.iloc[-1], preds.iloc[-1])
        test = folium.Html(html, script=True)
        popup = folium.Popup(test, max_width=400)
        gjson.add_child(popup)

        fgp.add_child(gjson)

fgp.add_to(m)
print('Done!')
m.save('index.html')
