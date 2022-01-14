import pandas as pd
import numpy as np
import folium
import json
import matplotlib.pyplot as plt
import seaborn as sns
import predict as h
# import train_model as h2
import xgboost as xgb
import os
import glob

IMG_HEIGHT = 300
IMG_WIDTH = 500

if not os.path.exists("images"):
    os.mkdir("images")

def get_html(name, width=IMG_WIDTH, height=IMG_HEIGHT):
    html = f'''
        <html>
            <body>
                <h1>{name}</h1><br>
                <img src="images/{name}.png" alt="{name}" width={width} height={height}>
            </body>
        </html>'''
    return html

def create_plot(data, file_name):
    sns.set_context("poster")
    plt.style.use('seaborn-poster')

    f, axes = plt.subplots()
    plt.plot(data['Predicted visitors'], label='Predicted visitors')
    plt.plot(data['Actual visitors'], label='Actual visitors')
    plt.setp(axes, xticks=np.linspace(0,96,5),
         xticklabels=['00:00:00','06:00:00','12:00:00','18:00:00', '00:00:00'])

    plt.xlabel('Location')
    plt.ylabel('Visitors')
    plt.legend(loc='upper left', prop={'size': 15})
    # plt.show()
    plt.savefig('images/'+file_name)

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

d = get_geodict('GeoJSON files/*.geojson')
geos = list(d.keys())

# print('Getting data...')
# data = h.get_data('CSVs/resono_2020_2022.csv')
data = pd.read_csv('CSVs/Resono_test.csv', index_col=0)
# print('Done!')

print('Loading model...')
model = xgb.Booster()
model.load_model("model.json")
print('Done!')

locations = data.Location.unique()
visits = list(np.random.randint(low=0, high=300, size=len(locations)))
chordata = pd.DataFrame(list(zip(locations, visits)),
               columns =['Location', 'Visitors'])


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
    columns=["Location", "Visitors"],
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
        preds = get_predictions(model, data_loc, name)
        create_plot(preds, name)
        html = get_html(name)
        test = folium.Html(html, script=True)
        popup = folium.Popup(test, max_width=2650)
        gjson.add_child(popup)

        fgp.add_child(gjson)

fgp.add_to(m)

m.save('index.html')
